import os, json, math, requests
import httpx
import asyncio
import datetime
import re 
from typing import List, Any, Dict, Optional, Tuple
import traceback
from itertools import permutations

from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.load import dumps, loads
from src.config import LLM, load_faiss_index, GMAPS_CLIENT

# 🚨 [중요] 사용자가 제공한 지역명 정규화 모듈 임포트
try:
    from src.region_cut_fuzz import normalize_region_name
except ImportError:
    def normalize_region_name(name): return name

# --- [1] LLM 체인 정의 (지역 추출, 설명 생성) ---

query_gen_prompt = PromptTemplate.from_template("""
역할: 당신은 '검색어 최적화 전문가'입니다.
목표: 사용자의 요청과 취향을 분석하여, 벡터 데이터베이스에서 가장 정확한 장소를 찾을 수 있는 **3개의 검색 쿼리**를 생성하세요.

[입력 정보]
- 여행지/지역: {target_region}
- 사용자 검색어: {query}
- 사용자 취향/정보: {user_info}
- 카테고리 필터: {category_filter}

[지침]
1. 사용자의 자연어 문장(취향)에서 **핵심 키워드(형용사, 명사)**만 추출하세요. (예: "조용한", "뷰맛집", "재즈")
2. 지역명과 핵심 키워드를 조합하여 검색어를 만드세요.
3. 다음 3가지 관점의 쿼리를 생성하세요:
   - 쿼리 1: 지역명 + 사용자 검색어 (기본 정확도 중심)
   - 쿼리 2: 지역명 + 사용자 검색어 + 취향 키워드 (구체적 니즈 중심)
   - 쿼리 3: 지역명 + 분위기/테마 키워드 (광범위 탐색)
4. 결과는 오직 쉼표(,)로 구분된 문자열로만 출력하세요. 다른 설명은 생략하세요.

[예시]
입력: 지역="서울", 검색어="카페", 취향="조용하고 작업하기 좋은 곳", 필터="카페"
출력: 서울 카페, 서울 조용한 작업하기 좋은 카페, 서울 스터디 카페 분위기
""")

query_gen_chain = query_gen_prompt | LLM | StrOutputParser()


# 1-1. 검색어에서 행정구역 추출 (LLM fallback용)
region_prompt = PromptTemplate.from_template("""
역할: 당신은 '지명 정규화 전문가'입니다.
목표: 사용자의 검색어("{query}")와 여행 목적지("{destination}")를 보고, 검색 대상이 되는 **정확한 행정구역 명칭** 하나만 출력하세요.

[규칙]
1. 검색어에 '해운대', '송도' 같은 구체적 지명이 있다면, 해당 지명의 **공식 행정구역명**을 찾으세요.
2. 검색어가 '맛집', '카페' 등 일반 명사뿐이라면, **여행 목적지("{destination}")**를 정규화해서 반환하세요.
3. **절대 추측하지 마세요.** 모르면 "{destination}"을 그대로 반환하세요.
4. 답변에는 군더더기 없이 **오직 지역명만** 출력하세요.

[예시]
- 입력: "해운대 맛집", 목적지: "부산" -> 출력: "부산광역시 해운대구"
- 입력: "성산일출봉", 목적지: "제주도" -> 출력: "제주특별자치도 서귀포시"
- 입력: "강남 점심", 목적지: "서울" -> 출력: "서울특별시 강남구"
- 입력: "맛집 추천", 목적지: "여수" -> 출력: "전라남도 여수시"
""")
region_chain = region_prompt | LLM | StrOutputParser()

# 1-2. 사용자 정보 기반 장소 추천사 생성 체인
desc_prompt = PromptTemplate.from_template("""
[상황]
사용자 정보: {user_info}
장소 이름: {place_name}
장소 특징/리뷰 요약: {place_data}

위 정보를 바탕으로, 이 장소가 **이 사용자에게 왜 좋은지** 매력적인 1~2줄의 추천사를 작성해줘.
- 반드시 한국어로 작성.
- 문장 끝은 '해요', '좋아요' 등으로 자연스럽게 마무리.
""")
desc_chain = desc_prompt | LLM | StrOutputParser()


# --- [2] 지리/거리 계산 헬퍼 함수 ---

async def get_coordinates(location_name: str):
    """지명/주소 -> 좌표 변환 (Google Maps API)"""
    if not GMAPS_CLIENT: return None, None
    try:
        # API 비용 절약을 위해 너무 긴 주소는 적당히 자르거나 처리할 수 있음
        res = await asyncio.to_thread(GMAPS_CLIENT.geocode, location_name, language='ko')
        if res:
            loc = res[0]['geometry']['location']
            return loc['lat'], loc['lng']
    except Exception as e:
        print(f"DEBUG: 좌표 변환 실패 ({location_name}): {e}")
    return None, None

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """두 좌표 간의 직선 거리(km) 계산 (Pure Python)"""
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except (ValueError, TypeError):
        return 9999.0

    R = 6371  # 지구 반지름 (km)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_distance_time(start_lat, start_lng, end_lat, end_lng, mode="driving"):
    """좌표 간 단순 직선 거리 및 예상 시간 추정"""
    dist = calculate_haversine_distance(start_lat, start_lng, end_lat, end_lng)
    
    speed = 4.0 if mode == "walking" else 30.0
    seconds = int((dist / speed) * 3600)
    
    if seconds < 3600: text = f"{seconds // 60}분"
    else: text = f"{seconds // 3600}시간 {(seconds % 3600) // 60}분"
    return dist, seconds, text

async def get_detailed_route(start_place: str, end_place: str, mode="transit", departure_time=None):
    """상세 경로 조회 (Google Maps Directions API)"""
    if not GMAPS_CLIENT: 
        print(f"DEBUG: ❌ GMAPS_CLIENT가 없습니다. (API Key 확인 필요)")
        return None
    if mode == "transit" and not departure_time: departure_time = datetime.datetime.now()
    if mode != "transit": departure_time = None

    try:
        print(f"DEBUG: 🗺️ 경로 검색 요청: {start_place} -> {end_place}")
        res = await asyncio.to_thread(
            GMAPS_CLIENT.directions, origin=start_place, destination=end_place,
            mode=mode, departure_time=departure_time, region="KR", language="ko"
        )
        if res:
            route = res[0]['legs'][0]
            steps_summary = []
            for step in route['steps']:
                tm = step['travel_mode']
                if tm == 'TRANSIT':
                    line = step.get('transit_details', {}).get('line', {})
                    name = line.get('short_name') or line.get('name') or "버스"
                    steps_summary.append(f"[{line.get('vehicle', {}).get('name', '대중교통')}] {name}")
                elif tm == 'WALKING': steps_summary.append("🚶 도보")
            
            if not steps_summary: steps_summary.append(f"이동 ({route['duration']['text']})")

            return {
                "mode": mode, "duration": route['duration']['text'],
                "duration_value": route['duration']['value'], "distance": route['distance']['text'],
                "steps": steps_summary,
                "start_location": route['start_location'], "end_location": route['end_location']
            }
    except Exception as e:
        print(f"DEBUG: ⚠️ 경로 검색 API 에러: {e}") # 에러 로그 출력
        return None
    
    # Fallback: 직선 거리 계산
    slat, slng = await get_coordinates(start_place)
    elat, elng = await get_coordinates(end_place)
    if slat and elat:
        dist, sec, txt = calculate_distance_time(slat, slng, elat, elng, mode)
        return {"mode": mode, "duration": txt, "duration_value": sec, "distance": f"{dist:.1f}km", "steps": ["직선거리"], "start_location": {"lat":slat, "lng":slng}, "end_location": {"lat":elat, "lng":elng}}
    return None

async def resolve_admin_region(query: str, destination: str) -> str:
    """
    [핵심 로직] "광안리" -> "부산광역시 수영구" 자동 변환기
    """
    if not GMAPS_CLIENT: 
        return normalize_region_name(destination)

    search_term = query
    if destination and destination not in query:
        search_term = f"{destination} {query}"
        
    print(f"DEBUG: 🗺️ 행정구역 식별 시도: '{search_term}'")

    try:
        geocode_res = await asyncio.to_thread(GMAPS_CLIENT.geocode, search_term, language='ko')
        
        if not geocode_res:
            return normalize_region_name(destination)

        loc = geocode_res[0]['geometry']['location']
        lat, lng = loc['lat'], loc['lng']
        
        reverse_res = await asyncio.to_thread(GMAPS_CLIENT.reverse_geocode, (lat, lng), language='ko')
        
        if not reverse_res:
            return normalize_region_name(destination)
            
        comps = reverse_res[0].get('address_components', [])
        level1 = "" 
        level2 = "" 
        
        for c in comps:
            types = c.get('types', [])
            if 'administrative_area_level_1' in types:
                level1 = c.get('long_name', '')
            elif 'sublocality_level_1' in types:
                level2 = c.get('long_name', '')
            elif 'locality' in types and not level2:
                level2 = c.get('long_name', '')

        extracted_region = f"{level1} {level2}".strip()
        
        if extracted_region:
            print(f"DEBUG: ✅ 변환 성공: '{query}' -> '{extracted_region}'")
            return extracted_region
        else:
            return normalize_region_name(destination)

    except Exception as e:
        print(f"DEBUG: 행정구역 변환 중 에러: {e}")
        return normalize_region_name(destination)


# --- [3] 핵심 검색 도구 (검색 + 필터링 + Fallback 로직) ---

async def _search_docs(query_str: str, k: int = 20):
    """Vector DB 검색 래퍼"""
    try:
        print(f"DEBUG: 🔍 벡터 DB 검색 시도: '{query_str}' (k={k})")
        db = load_faiss_index()
        if db is None:
            print("DEBUG: ❌ 벡터 DB 인스턴스 없음 (load_faiss_index returned None)")
            return []
        # If similarity_search is blocking/heavy, run in thread
        results = await asyncio.to_thread(db.similarity_search, query_str, k=k)
        print(f"DEBUG: 🔎 DB 검색 결과 개수: {len(results)}")
        return results
    except Exception as e:
        print(f"DEBUG: DB 검색 실패: {e}")
        return []

async def _filter_candidates(docs, target_region: str, exclude_places: List[str], category_filter: str):
    """
    메타데이터 필터링 (지역명 + 카테고리 + 제외 장소)
    - 더 관대하게 매칭: 지역 토큰 중 하나라도 주소/이름에 포함되면 허용
    - 모든 비교는 소문자 기준으로 수행
    """
    candidates = []

    # 안전한 defaults
    if exclude_places is None:
        exclude_places = []
    target_region = (target_region or "").strip()

    # 1. 지역명 필터 키워드 준비 (소문자)
    target_parts = [p.strip() for p in target_region.split() if p.strip()]
    refined_targets = [re.sub(r'(특별시|광역시|도|시|군|구)$', '', p).lower() for p in target_parts]
    if not refined_targets:
        refined_targets = [p.lower() for p in target_parts]

    print(f"DEBUG: ⚙️ 필터 적용 - 지역키워드:{refined_targets} / 카테고리:{category_filter}")

    for doc in docs:
        name = (doc.metadata.get('장소명') or doc.metadata.get('name') or '').strip()
        address = (doc.metadata.get('지역') or doc.metadata.get('road_address') or doc.metadata.get('address') or '').strip()
        doc_cat = (doc.metadata.get('카테고리') or doc.metadata.get('category') or '').strip()

        # 🚨 [핵심 수정] 메타데이터가 비어있으면 page_content에서 추출
        # 형식: "{장소명}은(는) {지역}에 위치한 {카테고리}입니다."
        if (not name or not address or not doc_cat) and hasattr(doc, 'page_content'):
            content = doc.page_content or ''
            try:
                # 예: "제주덕구 경기광주점은(는) 경기도 광주시에 위치한 식당 육류,고기요리입니다."
                if '은(는)' in content and '에 위치한' in content:
                    parts = content.split('은(는)')
                    if len(parts) >= 2:
                        if not name:
                            name = parts[0].strip()

                        location_part = parts[1].split('에 위치한')
                        if len(location_part) >= 2:
                            if not address:
                                address = location_part[0].strip()
                            if not doc_cat:
                                cat_part = location_part[1].split('입니다')[0].strip()
                                doc_cat = cat_part
            except:
                pass  # 파싱 실패 시 그냥 넘어감

        name_l = name.lower()
        address_l = address.lower()

        # A. 제외 장소 필터 (이름 기반)
        if name in exclude_places or name_l in [e.lower() for e in exclude_places]:
            continue

        # B. 카테고리 필터 (엄격 + 유연)
        if category_filter:
            cf = category_filter.lower()
            if cf in ("식당", "맛집"):
                if not any(x in doc_cat for x in ["식당", "맛집", "음식점"]):
                    continue
            elif cf == "카페":
                if not any(x in doc_cat for x in ["카페", "커피"]):
                    continue
            elif cf == "관광지":
                if not any(x in doc_cat for x in ["관광", "여행", "명소"]):
                    continue

        # C. 지역 텍스트 매칭 필터 (주소 기반으로만 매칭)
        is_match = False
        if not refined_targets:
            is_match = True
        else:
            # 지역 필터는 주소(address)만 확인 (장소명에 지역명이 포함된 경우 오매칭 방지)
            for token in refined_targets:
                if not token:
                    continue
                if token in address_l:  # 주소에서만 검색
                    is_match = True
                    break

        if is_match:
            candidates.append(doc)
            
    print(f"DEBUG: ⚙️ 필터링 후 후보 수: {len(candidates)}")
    return candidates

@tool
async def find_and_select_best_place(query: str,
                                    destination: str,
                                    anchor: str = "",
                                    exclude_places: List[str] = [],
                                    user_info: str = "", 
                                    category_filter: str = "") -> str:
    """
    [핵심 도구] 최적의 장소 1곳을 반환합니다 + 리뷰 정보 포함.
    """
    print(f"\n--- [DEBUG] find_and_select_best_place 호출 ---")
    
    # 1. 지역 및 기준점 설정 (개선: 여러 방식으로 resolve 시도하여 더 구체적인 영역 사용)
    target_region = ""
    try:
        if anchor:
            target_region = await resolve_admin_region(anchor, destination)
            print(f"DEBUG: Anchor 기반 target_region -> '{target_region}'")
        else:
            # 시도 1: 쿼리만으로 resolve (특정 지명 포함시 더 구체적으로 나올 수 있음)
            resolved_query_region = await resolve_admin_region(query, destination)
            print(f"DEBUG: resolved_query_region -> '{resolved_query_region}'")
            # 시도 2: destination + query (일반적으로 destination을 포함하면 검색 범위가 명확해짐)
            if destination:
                resolved_dest_query = await resolve_admin_region(f"{destination} {query}", destination)
            else:
                resolved_dest_query = resolved_query_region
            print(f"DEBUG: resolved_dest_query -> '{resolved_dest_query}'")

            # 우선순위 결정: 더 구체적인(더 많은 토큰을 가진) 지역명을 선택
            def region_specificity_score(region_str: str):
                if not region_str: return 0
                # tokens count including spaces (e.g., "서울특별시 강남구" -> 2)
                return len([p for p in region_str.split() if p.strip()])

            s_query = region_specificity_score(resolved_query_region)
            s_dest = region_specificity_score(resolved_dest_query)
            # 우선: resolved_query_region이 더 구체적이면 선택, 아니면 destination+query 결과 사용
            if s_query > s_dest:
                target_region = resolved_query_region
            else:
                target_region = resolved_dest_query
            # 마지막 보정: 비어 있으면 destination 사용
            if not target_region and destination:
                target_region = destination
            print(f"DEBUG: 선택된 target_region -> '{target_region}'")
    except Exception as e:
        print(f"DEBUG: resolve_admin_region 실패: {e}")
        target_region = destination or ""

    target_region = (target_region or "").strip()
    print(f"DEBUG: target_region resolved -> '{target_region}'")

    # 기준점(Anchor) 좌표 확보 (거리 계산용)
    center_place = anchor if anchor else target_region
    center_lat, center_lng = None, None
    if center_place:
        print(f"DEBUG: 📍 기준점 좌표 조회: '{center_place}'")
        try:
            center_lat, center_lng = await get_coordinates(center_place)
        except Exception as e:
            print(f"DEBUG: 좌표 조회 실패: {e}")

    try:
        # A. 쿼리 생성
        generated_queries_str = await query_gen_chain.ainvoke({
            "target_region": target_region,
            "query": query,
            "user_info": user_info,
            "category_filter": category_filter
        })
        # 쉼표로 분리하여 리스트화
        search_queries = [q.strip() for q in generated_queries_str.split(',') if q.strip()]
        print(f"DEBUG: 🧠 생성된 멀티 쿼리: {search_queries}")
        
    except Exception as e:
        print(f"DEBUG: 쿼리 생성 실패({e}) -> 기본 쿼리 사용")
        search_queries = [f"{target_region} {query} {category_filter}"]
    # B. 병렬 검색 실행 (모든 쿼리에 대해 동시에 검색)
    # 각 쿼리당 상위 50개씩 검색 (너무 많으면 느려지므로 조절)
    tasks = [_search_docs(q, k=50) for q in search_queries]
    results_list = await asyncio.gather(*tasks)
    
    # C. 결과 통합 및 중복 제거 (Dedup)
    seen_places = set()
    aggregated_docs = []
    
    for docs in results_list:
        for doc in docs:
            p_name = doc.metadata.get('장소명', '')
            # 이미 결과 목록에 있거나, 제외 목록에 있다면 스킵
            if p_name and p_name not in seen_places and p_name not in exclude_places:
                seen_places.add(p_name)
                aggregated_docs.append(doc)
    
    candidates = await _filter_candidates(aggregated_docs, target_region, exclude_places, category_filter)
    print(f"DEBUG: 🎯 필터링 후 후보군 수: {len(candidates)}")

    if not candidates:
        print(f"DEBUG: ⚠️ 1차 검색 결과 없음 -> 2차 검색(선호 제외, 거리/카테고리 중심) 전환")
        
        # user_info 제거하고 기본 쿼리로만 검색
        search_query_v2 = f"{query} {target_region} {category_filter}"
        print(f"DEBUG: 🔍 2차 검색 시도: '{search_query_v2}'")
        
        docs_v2 = await _search_docs(search_query_v2, k=30)
        candidates = await _filter_candidates(docs_v2, target_region, exclude_places, category_filter)
        print(f"DEBUG: 🎯 2차 후보군 수: {len(candidates)}")

        # 2차 검색 결과가 있다면, 이 중 "가장 가까운 곳"을 찾기 위해 좌표 변환 수행
        if candidates and center_lat and center_lng:
            print("DEBUG: 📏 후보군 상위 5개 거리 계산 및 최단거리 정렬 시작")
            
            # API 비용 절약을 위해 상위 5개만 좌표 변환
            top_n_candidates = candidates[:5]
            candidates_with_score = []
            
            for doc in top_n_candidates:
                addr = doc.metadata.get('지역', '').strip()

                # 🚨 메타데이터가 비어있으면 page_content에서 주소 추출
                if not addr and hasattr(doc, 'page_content'):
                    content = doc.page_content or ''
                    try:
                        if '은(는)' in content and '에 위치한' in content:
                            parts = content.split('은(는)')
                            if len(parts) >= 2:
                                location_part = parts[1].split('에 위치한')
                                if len(location_part) >= 2:
                                    addr = location_part[0].strip()
                    except:
                        pass

                p_lat, p_lng = await get_coordinates(addr)
                
                dist = 9999.0
                if p_lat and p_lng:
                    dist = calculate_haversine_distance(center_lat, center_lng, p_lat, p_lng)
                
                candidates_with_score.append((dist, doc))
            
            # 거리순 정렬 (오름차순)
            candidates_with_score.sort(key=lambda x: x[0])
            
            # 정렬된 순서대로 candidates 교체
            candidates = [x[1] for x in candidates_with_score]
            if candidates_with_score:
                 print(f"DEBUG: 🎯 최단 거리 장소 선정: {candidates_with_score[0][0]:.1f}km")

    if not candidates:
        print("DEBUG: ❌ 2차 검색까지 실패. 검색 결과 없음.")
        return json.dumps({"name": "추천 장소 없음", "type": "정보없음", "description": "조건에 맞는 장소를 찾지 못했습니다.", "reviews": []}, ensure_ascii=False)

    best_doc = candidates[0]
    best_name = best_doc.metadata.get('장소명', '').strip()
    best_address = best_doc.metadata.get('지역', '').strip()
    best_category = best_doc.metadata.get('카테고리', '').strip()

    # 🚨 [핵심 수정] 메타데이터가 비어있으면 page_content에서 추출
    if (not best_name or not best_address or not best_category) and hasattr(best_doc, 'page_content'):
        content = best_doc.page_content or ''
        try:
            # 형식: "{장소명}은(는) {지역}에 위치한 {카테고리}입니다."
            if '은(는)' in content and '에 위치한' in content:
                parts = content.split('은(는)')
                if len(parts) >= 2:
                    if not best_name:
                        best_name = parts[0].strip()

                    location_part = parts[1].split('에 위치한')
                    if len(location_part) >= 2:
                        if not best_address:
                            best_address = location_part[0].strip()
                        if not best_category:
                            cat_part = location_part[1].split('입니다')[0].strip()
                            best_category = cat_part
        except:
            pass

    # Fallback
    if not best_name:
        best_name = '이름미상'

    # 설명 생성
    description = await desc_chain.ainvoke({
        "user_info": user_info,
        "place_name": best_name,
        "place_data": best_doc.page_content[:400]
    })

    # ✨ [새로 추가] 리뷰 데이터 추출 (metadata나 page_content에서)
    reviews = []
    try:
        # 방법 1: metadata에서 직접 리뷰 추출 (있으면)
        if 'reviews' in best_doc.metadata:
            reviews_data = best_doc.metadata.get('reviews', [])
            if isinstance(reviews_data, list):
                reviews = reviews_data[:3]  # 상위 3개만 추출
            elif isinstance(reviews_data, str):
                # 문자열 형태라면 줄바꿈이나 구분자로 split
                reviews = [r.strip() for r in reviews_data.split('\n') if r.strip()][:3]
        
        # 방법 2: page_content에서 리뷰 키워드 찾기
        if not reviews and best_doc.page_content:
            content = best_doc.page_content
            # 리뷰 섹션이 있는지 확인 (예: "리뷰:" 이후 텍스트)
            if '리뷰' in content or 'review' in content.lower():
                # 간단하게 리뷰 섹션 후 첫 2-3문장 추출
                lines = content.split('\n')
                review_start = False
                temp_reviews = []
                for line in lines:
                    if '리뷰' in line or 'review' in line.lower():
                        review_start = True
                        continue
                    if review_start and line.strip():
                        temp_reviews.append(line.strip())
                        if len(temp_reviews) >= 2:
                            break
                reviews = temp_reviews
    except Exception as e:
        print(f"DEBUG: 리뷰 추출 중 에러: {e}")
        reviews = []

    # 리뷰가 없으면 빈 리스트로 설정
    if not reviews:
        reviews = []

    result_data = {
        "name": best_name,
        "type": best_category if best_category else '장소',
        "description": description.strip(),
        "address": best_address,
        "reviews": reviews,  # ✨ [새로 추가] 리뷰 필드
        "coordinates": None
    }
    
    print(f"✅ 최종 추천: {best_name} / 리뷰 개수: {len(reviews)}")
    return json.dumps(result_data, ensure_ascii=False)



@tool
async def plan_itinerary_timeline(itinerary: List[Dict]) -> str:
    """
    [일정 정리 도구] 일정 리스트를 받아 시간순 타임라인 생성
    """
    print(f"\n--- [DEBUG] plan_itinerary_timeline 호출 ---")
    places_only = [item for item in itinerary if item.get('type') != 'move']
    
    try:
        from src.scheduler.smart_scheduler import SmartScheduler
        scheduler = SmartScheduler(start_time_str="10:00")
        
        days = sorted(list(set(item.get('day', 1) for item in places_only)))
        final_timeline = []
        
        for day in days:
            day_items = [item for item in places_only if item.get('day', 1) == day]
            day_schedule = await scheduler.plan_day(day_items)
            
            for item in day_schedule:
                item['day'] = day
                if item.get('type') == 'move':
                    detail = item.get('transport_detail', '')
                    min_val = item.get('duration_min', 0)
                    item['duration_text'] = f"약 {min_val}분 ({detail})" if detail else f"약 {min_val}분 (이동)"
                final_timeline.append(item)
                
        return json.dumps(final_timeline, ensure_ascii=False)

    except Exception as e:
        print(f"ERROR: 스케줄링 실패: {e}")
        traceback.print_exc()
        return json.dumps(itinerary, ensure_ascii=False)

def _solve_tsp(duration_matrix, start_fixed, n):
    """TSP 알고리즘"""
    min_duration = float('inf')
    best_order_indices = []
    
    indices = list(range(n))
    if start_fixed: indices = list(range(1, n))

    if len(indices) > 8:
        current = 0
        unvisited = set(indices)
        path = [0]
        cost = 0
        while unvisited:
            nxt = min(unvisited, key=lambda i: duration_matrix[current][i])
            cost += duration_matrix[current][nxt]
            path.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return path, cost

    for p in permutations(indices):
        current_indices = [0] + list(p) if start_fixed else list(p)
        current_dur = sum(duration_matrix[current_indices[i]][current_indices[i+1]] for i in range(len(current_indices)-1))
        if current_dur < min_duration:
            min_duration = current_dur
            best_order_indices = current_indices
            
    return best_order_indices, min_duration

@tool
async def optimize_and_get_routes(places: List[str], start_location: str = "") -> str:
    """최적 경로(순서) 계산"""
    if not GMAPS_CLIENT: return "API 키 없음"
    all_places = [start_location] + places if start_location else places
    if len(all_places) < 2: return "장소 부족"

    try:
        matrix = await asyncio.to_thread(
            GMAPS_CLIENT.distance_matrix, origins=all_places, destinations=all_places, mode="transit"
        )
        dur_matrix = []
        for row in matrix['rows']:
            vals = [el.get('duration', {}).get('value', 99999) for el in row['elements']]
            dur_matrix.append(vals)
            
        best_indices, _ = await asyncio.to_thread(_solve_tsp, dur_matrix, bool(start_location), len(all_places))
        optimized = [all_places[i] for i in best_indices]
        
        return json.dumps({"optimized_order": optimized}, ensure_ascii=False)
        
    except Exception as e:
        return f"최적화 실패: {e}"

@tool
def get_weather_forecast(destination: str, dates: str) -> str:
    """
    도시명(destination)으로 위도/경도를 조회하고, 그 좌표로 5일 예보를 조회하여,
    사용자가 요청한 날짜(dates)의 날씨만 요약해 반환합니다. (3단계 날짜 파싱 적용)
    """
    API_KEY = os.getenv("OWM_API_KEY")
    if not API_KEY:
        return "오류: OWM_API_KEY가 .env 파일에 설정되지 않았습니다."

    # 1단계: Geocoding
    geo_url = "https://api.openweathermap.org/geo/1.0/direct"
    geo_params = {'q': f"{destination},KR", 'limit': 1, 'appid': API_KEY}
    lat, lon = None, None
    try:
        response = requests.get(geo_url, params=geo_params, timeout=5)
        response.raise_for_status()
        geo_data = response.json()
        if geo_data:
            lat = geo_data[0]['lat']
            lon = geo_data[0]['lon']
        else:
            return f"오류: '{destination}'의 좌표(Geocoding)를 찾을 수 없습니다."
    except Exception as e:
        return f"오류: Geocoding API 호출 중 문제 발생: {e}"

    # 2단계: Forecast
    forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
    forecast_params = {'lat': lat, 'lon': lon, 'appid': API_KEY, 'units': 'metric', 'lang': 'kr'}
    forecasts = None
    try:
        response = requests.get(forecast_url, params=forecast_params, timeout=10)
        response.raise_for_status()
        forecast_data = response.json()
        forecasts = forecast_data.get('list', [])
    except Exception as e:
        return f"오류: Forecast API 호출 중 문제 발생: {e}"
    if not forecasts:
        return "오류: Forecast API에서 'list' 데이터를 찾을 수 없습니다."

    # 3단계: 날짜 필터링 (3-Step 파싱 로직)
    target_date_str = ""
    today = datetime.datetime.now()
    
    try:
        # 1. 'YYYY년 M월 D일' (공백 O)
        target_date_obj = datetime.datetime.strptime(dates, "%Y년 %m월 %d일")
        target_date_str = target_date_obj.strftime("%Y-%m-%d")
    except ValueError:
        try:
            # 2. 'YYYY년MM월DD일' (공백 X)
            target_date_obj = datetime.datetime.strptime(dates, "%Y년%m월%d일")
            target_date_str = target_date_obj.strftime("%Y-%m-%d")
        except ValueError:
            try:
                # 3. 'M월 D일' (연도 없음)
                target_date_obj = datetime.datetime.strptime(dates, "%m월 %d일")
                target_date_obj = target_date_obj.replace(year=today.year)
                target_date_str = target_date_obj.strftime("%Y-%m-%d")
            except ValueError:
                 # 4. 모든 형식 실패 -> 키워드 검색
                 if "주말" in dates or "토요일" in dates:
                     days_until_saturday = (5 - today.weekday() + 7) % 7
                     saturday = today + datetime.timedelta(days=days_until_saturday)
                     target_date_str = saturday.strftime("%Y-%m-%d")
                 elif "내일" in dates:
                     tomorrow = today + datetime.timedelta(days=1)
                     target_date_str = tomorrow.strftime("%Y-%m-%d")
                 else: 
                     tomorrow = today + datetime.timedelta(days=1)
                     target_date_str = tomorrow.strftime("%Y-%m-%d")
    
    # 4단계: 결과 가공
    output_str = f"[{destination} ({target_date_str}) 날씨 예보 (OWM)]\n"
    found = False
    for forecast in forecasts:
        if forecast['dt_txt'].startswith(target_date_str):
            time_utc = forecast['dt_txt'].split(' ')[1][:5]
            temp = forecast['main']['temp'] 
            desc = forecast['weather'][0]['description']
            output_str += f"- {time_utc} (UTC): {temp:.1f}℃, {desc}\n"
            found = True
    
    if not found:
        return f"정보: {target_date_str} 날짜의 예보를 찾을 수 없습니다. (OWM은 5일치만 제공)"
    
    return output_str


@tool
def confirm_and_download_pdf():
    """최종 승인 및 PDF 다운로드 활성화"""
    return "PDF 다운로드 승인됨"

@tool
async def delete_place(place_name: str) -> str:
    """일정에서 특정 장소를 삭제합니다."""
    return json.dumps({"action": "delete", "place_name": place_name}, ensure_ascii=False)

@tool
async def replace_place(old_place_name: str, query: str, destination: str) -> str:
    """일정 교체 도구"""
    return json.dumps({"action": "replace", "old": old_place_name, "new_query": query}, ensure_ascii=False)


# --- 도구 등록 ---
TOOLS = [
    find_and_select_best_place,
    plan_itinerary_timeline,
    optimize_and_get_routes,
    get_weather_forecast,
    delete_place,
    replace_place,
    confirm_and_download_pdf
]
AVAILABLE_TOOLS = {tool.name: tool for tool in TOOLS}