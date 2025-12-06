# src/tools.py

import os, json, math
import httpx
import asyncio
import datetime
import re 
from typing import List, Any, Dict, Optional
import traceback
from itertools import permutations

from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.load import dumps, loads
from src.config import LLM, DB_INSTANCE, GMAPS_CLIENT 

# 🚨 [중요] 사용자가 제공한 지역명 정규화 모듈 임포트
try:
    from src.region_cut_fuzz import normalize_region_name
except ImportError:
    def normalize_region_name(name): return name

# --- [신규] 1. 지역명 추출 및 개인화 설명 체인 ---

# 1-1. 검색어에서 행정구역 추출 (LLM fallback용)
region_prompt = PromptTemplate.from_template("""
사용자의 검색어: "{query}"
현재 여행 목적지: "{destination}"

이 검색어가 가리키는 정확한 행정구역(City, District)을 추출해.
- 예: "해운대 맛집" -> "부산광역시 해운대구"
- 예: "문경새재" -> "경상북도 문경시"
- 예: "근처 카페" -> "{destination}" (목적지 따라감)

답변은 군더더기 없이 **오직 지역명만** 출력해.
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


# --- 2. 지리 정보 헬퍼 함수 ---

async def get_coordinates(location_name: str):
    """지명 -> 좌표 변환"""
    if not GMAPS_CLIENT: return None, None
    try:
        res = await asyncio.to_thread(GMAPS_CLIENT.geocode, location_name, language='ko')
        if not res:
            normalized = normalize_region_name(location_name)
            if normalized != location_name:
                res = await asyncio.to_thread(GMAPS_CLIENT.geocode, normalized, language='ko')
        
        if res:
            loc = res[0]['geometry']['location']
            return loc['lat'], loc['lng']
    except Exception as e:
        print(f"DEBUG: 좌표 변환 실패 ({location_name}): {e}")
    return None, None

def calculate_distance_time(start_lat, start_lng, end_lat, end_lng, mode="driving"):
    """직선 거리 기반 시간 추정"""
    R = 6371
    d_lat = math.radians(end_lat - start_lat)
    d_lng = math.radians(end_lng - start_lng)
    a = math.sin(d_lat/2)**2 + math.cos(math.radians(start_lat)) * math.cos(math.radians(end_lat)) * math.sin(d_lng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    dist = R * c
    
    speed = 4.0 if mode == "walking" else 30.0
    seconds = int((dist / speed) * 3600)
    
    if seconds < 3600: text = f"{seconds // 60}분"
    else: text = f"{seconds // 3600}시간 {(seconds % 3600) // 60}분"
    return dist, seconds, text

async def get_detailed_route(start_place: str, end_place: str, mode="transit", departure_time=None):
    """상세 경로 조회 (Google Maps Directions API)"""
    if not GMAPS_CLIENT: return None
    if mode == "transit" and not departure_time: departure_time = datetime.datetime.now()
    if mode != "transit": departure_time = None

    try:
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
        # print(f"DEBUG: API 경로 조회 실패: {e}")
        pass
    
    # Fallback
    slat, slng = await get_coordinates(start_place)
    elat, elng = await get_coordinates(end_place)
    if slat and elat:
        dist, sec, txt = calculate_distance_time(slat, slng, elat, elng, mode)
        return {"mode": mode, "duration": txt, "duration_value": sec, "distance": f"{dist:.1f}km", "steps": ["직선거리"], "start_location": {"lat":slat, "lng":slng}, "end_location": {"lat":elat, "lng":elng}}
    return None

# --- [핵심 복구] 행정구역 변환 함수 (ImportError 해결 대상) ---
async def resolve_admin_region(query: str, destination: str) -> str:
    """
    [핵심 로직] "광안리" -> "부산광역시 수영구" 자동 변환기
    Google Maps API를 사용하여 비공식 지명을 공식 행정구역으로 변환합니다.
    """
    # 1. API 클라이언트 확인
    if not GMAPS_CLIENT: 
        return normalize_region_name(destination)

    # 2. 검색어 보정
    search_term = query
    if destination and destination not in query:
        search_term = f"{destination} {query}"
        
    print(f"DEBUG: 🗺️ 행정구역 식별 시도: '{search_term}'")

    try:
        # 3. Geocoding
        geocode_res = await asyncio.to_thread(GMAPS_CLIENT.geocode, search_term, language='ko')
        
        if not geocode_res:
            return normalize_region_name(destination)

        loc = geocode_res[0]['geometry']['location']
        lat, lng = loc['lat'], loc['lng']
        
        # 4. Reverse Geocoding
        reverse_res = await asyncio.to_thread(GMAPS_CLIENT.reverse_geocode, (lat, lng), language='ko')
        
        if not reverse_res:
            return normalize_region_name(destination)
            
        # 5. 행정구역 파싱
        comps = reverse_res[0].get('address_components', [])
        level1 = "" # 광역
        level2 = "" # 기초
        
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


# --- 3. 핵심 도구 (Tools) ---

@tool
async def find_and_select_best_place(query: str, destination: str, anchor: str = "", exclude_places: List[str] = [], user_info: str = "") -> str:
    """
    [핵심 도구] 장소를 검색하고 최적의 1곳을 반환합니다.
    """
    print(f"\n--- [DEBUG] find_and_select_best_place 호출 ---")
    print(f"QUERY: {query} / ANCHOR: {anchor} / DEST: {destination}")
    
    # ---------------------------------------------------------
    # [수정된 로직] 앵커 우선 변환 정책
    # ---------------------------------------------------------
    target_region = ""
    
    # 1. 앵커(구체적 장소/지역)가 있다면 -> 앵커를 행정구역으로 변환 (예: "광안리" -> "부산 수영구")
    if anchor:
        print(f"DEBUG: ⚓️ 앵커 기반 지역 변환 시도: '{anchor}'")
        target_region = await resolve_admin_region(anchor, destination)
    
    # 2. 앵커가 없다면 -> 쿼리나 목적지를 기반으로 변환 (예: "부산 맛집" -> "부산광역시")
    else:
        print(f"DEBUG: 🔍 쿼리/목적지 기반 지역 변환 시도")
        target_input = query if destination in query else f"{destination} {query}"
        target_region = await resolve_admin_region(target_input, destination)

    target_region = target_region.strip()
    print(f"DEBUG: 🎯 확정 타겟 지역: '{target_region}'")

    # ---------------------------------------------------------
    # Vector DB 검색
    # ---------------------------------------------------------
    # 검색어 구성: "부산광역시 수영구" + "오션뷰 카페"
    # 이렇게 해야 "수영구"에 있는 "오션뷰 카페"만 나옵니다.
    search_query = f"{target_region} {query}"
    
    try:
        # k=20으로 넉넉하게 가져옴
        docs = await DB_INSTANCE.asimilarity_search(search_query, k=20)
    except Exception as e:
        print(f"DEBUG: Vector Store 검색 실패: {e}")
        return "검색 시스템 오류가 발생했습니다."

    # ---------------------------------------------------------
    # 필터링 및 후보 선정 (기존 로직 유지)
    # ---------------------------------------------------------
    candidates = []
    target_parts = target_region.split()
    
    refined_targets = []
    for part in target_parts:
        clean_part = re.sub(r'(특별시|광역시|특별자치시|특별자치도|도|시|군|구)$', '', part)
        if len(clean_part) >= 2: refined_targets.append(clean_part)
            
    if not refined_targets: refined_targets = target_parts

    print(f"DEBUG: ⚙️ 필터 키워드: {refined_targets}")

    for doc in docs:
        name = doc.metadata.get('장소명', '이름미상')
        address = doc.metadata.get('지역', '')
        
        if name in exclude_places: continue
        
        # 교차 검증
        is_match = False
        if all(k in address for k in refined_targets): is_match = True
        elif refined_targets and refined_targets[-1] in address: is_match = True
            
        if is_match: candidates.append(doc)

    # Fallback (필터 실패 시 완화)
    if not candidates:
        print("DEBUG: ⚠️ 엄격 매칭 실패. 검색 상위 결과 사용.")
        candidates = docs[:3] 

    # 최적 장소 선정
    best_doc = candidates[0]
    best_name = best_doc.metadata.get('장소명')
    best_address = best_doc.metadata.get('지역')
    
    description = await desc_chain.ainvoke({
        "user_info": user_info,
        "place_name": best_name,
        "place_data": best_doc.page_content[:400]
    })

    result_data = {
        "name": best_name,
        "type": best_doc.metadata.get('카테고리', '장소'), 
        "description": description.strip(),
        "address": best_address,
        "coordinates": None  
    }
    
    print(f"✅ 최종 추천: {best_name}")
    return json.dumps(result_data, ensure_ascii=False)


@tool
async def plan_itinerary_timeline(itinerary: List[Dict]) -> str:
    """
    [일정 정리 도구]
    일정 리스트를 받아 시간순 타임라인을 생성하고, 상세 교통편 정보를 포함합니다.
    """
    print(f"\n--- [DEBUG] plan_itinerary_timeline 호출 ---")
    places_only = [item for item in itinerary if item.get('type') != 'move']
    
    # 순환 참조 방지를 위해 함수 내부 import
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


# --- [복구] TSP 기반 경로 최적화 도구 ---
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
async def get_weather_forecast(destination: str, dates: str) -> str:
    """날씨 조회 도구"""
    return f"[{destination}] 날씨 정보: 맑음, 기온 20도 (API 연동 필요)" 

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