import os, json
import requests
import datetime
import re 
from typing import List, Any 

from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.load import dumps, loads
from src.config import LLM, load_faiss_index, GMAPS_CLIENT

from itertools import permutations



def normalize_message_to_str(message: Any) -> str:
    """LLM / LangChain 메시지나 content를 항상 str로 변환."""
    if message is None:
        return ""

    # BaseMessage 같은 객체면 .content 다시 태워서 처리
    if hasattr(message, "content"):
        return normalize_message_to_str(message.content)

    # 이미 str이면 그대로
    if isinstance(message, str):
        return message

    # 멀티파트: [{"type": "text", "text": "..."}, ...]
    if isinstance(message, list):
        parts = []
        for part in message:
            if isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    parts.append(str(part["text"]))
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)

    # dict인 경우 JSON 문자열로
    if isinstance(message, dict):
        try:
            return json.dumps(message, ensure_ascii=False)
        except TypeError:
            return str(message)

    # 나머지는 일단 문자열 캐스팅
    return str(message)

# --- RAG 헬퍼 함수 ---

def format_docs(docs):
    """검색된 Document 객체를 LLM 프롬프트용 문자열로 변환합니다."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- RAG 체인 구성 ---

# 2. RAG-Fusion용 쿼리 생성 프롬프트
template = """
당신은 AI 여행/맛집 추천 조수입니다. 
주어진 사용자 질문에 대해 전국 관광지와 식당 리뷰 데이터를 기반으로 관련 정보를 찾을 수 있도록 다섯 가지 다른 버전을 생성하세요.
각 질문은 새 줄로 구분하여 제공하세요. 원본 질문: {question}
"""
prompt_perspectives = ChatPromptTemplate.from_template(template)


# 3. LLM의 쿼리 생성 결과를 정제하는 헬퍼 함수
def clean_generated_queries(text: str) -> List[str]:
    """LLM이 생성한 쿼리 문자열에서 실제 쿼리만 정리하여 리스트로 반환합니다."""
    queries = []
    for line in text.split("\n"):
        cleaned_line = re.sub(r"^\d+[:.]\s*", "", line).strip()
        if not cleaned_line:
            continue
        if cleaned_line.startswith("다음은") or cleaned_line.startswith("원본 질문"):
            continue
        queries.append(cleaned_line)
    return queries

# 4. 쿼리 생성 체인
generate_queries = (
    prompt_perspectives
    | LLM
    | StrOutputParser()
    | clean_generated_queries 
)

# 6. RAG 후보 목록 생성 프롬프트
final_prompt = ChatPromptTemplate.from_template(
    """당신은 AI 여행 가이드의 검색 조수입니다.
제공된 맥락(리뷰 데이터)을 참고하여, 사용자 질문에 대한 '후보 장소 목록'을 생성하세요.

지침:
1.  결과는 반드시 '후보 목록' 형식이어야 합니다.
2.  최대 5개의 후보를 제시하세요.
3.  각 후보는 [이름]과 [핵심 특징(한 줄 요약)]만 포함해야 합니다.
4.  친절한 인사말이나 서론, 결론을 붙이지 마세요. 오직 목록만 반환하세요.

--- 맥락 ---
{context}

--- 사용자 질문 ---
{question}

--- 후보 목록 (이 형식 준수) ---
1. [장소 이름]: [특징 요약]
2. [장소 이름]: [특징 요약]
3. [장소 이름]: [특징 요약]
4. [장소 이름]: [특징 요약]
5. [장소 이름]: [특징 요약]
"""
)

# 7. RAG 최종 생성 체인
final_generation_chain = (
    final_prompt
    | LLM
    | StrOutputParser()
)

# --- 4. 에이전트가 사용할 '도구(Tools)' 정의 ---

@tool
def search_attractions_and_reviews(query: str) -> str:
    """
    사용자 쿼리를 5개로 확장(및 정제)하고,
    '각 쿼리별 Top-1' 결과를 결합하여 후보 목록을 검색합니다.
    """
    print(f"\n--- [DEBUG] search_attractions_and_reviews 호출됨 ---") # 👈 [추가]
    print(f"DEBUG: RAG 원본 사용자 쿼리: {query}") # 👈 [추가]

    try:
        DB = load_faiss_index() # 캐시된 DB 로드
        FAISS_RETRIEVER = DB.as_retriever(search_type="similarity", search_kwargs={'k': 1})
        retrieval_only_chain = FAISS_RETRIEVER.map() # 리트리버 체인 동적 생성
    except Exception as e:
        print(f"!!!!!!!!!! [DEBUG] FAISS 인덱스 로드 실패 !!!!!!!!!!")
        print(f"DEBUG: Error details: {e}")
        return "오류: RAG 벡터 데이터베이스를 로드하는 데 실패했습니다."
    
    # 1. 5개 쿼리 생성 및 정제
    generated_queries = generate_queries.invoke(query)
    
    # 👈 [추가] RAG-Fusion을 위해 생성된 쿼리 목록 확인
    print(f"DEBUG: RAG-Fusion 생성 쿼리 (최대 5개): {generated_queries}")

    # 2. RAG 병렬 검색 (각 쿼리당 k=1)
    parallel_search_results = retrieval_only_chain.invoke(generated_queries)
    
    # 👈 [추가] FAISS 벡터DB가 반환한 원본 검색 결과 (Document 리스트의 리스트)
    print(f"DEBUG: FAISS 원본 검색 결과 (Raw Docs): {parallel_search_results}")

    # 3. Top-1 결과 결합 (중복 제거)
    top_1_docs = []
    seen_content = set()
    for doc_list in parallel_search_results:
        if doc_list:
            doc = doc_list[0]
            if doc.page_content not in seen_content:
                top_1_docs.append(doc)
                seen_content.add(doc.page_content)
    
    # 4. LLM 요약 (최종 후보 목록 생성)
    context_str = format_docs(top_1_docs)
    
    # 👈 [추가] 요약 LLM에 전달할 최종 맥락(context) 확인
    print(f"DEBUG: 요약 LLM에 전달할 최종 Context:\n{context_str[:500]}...") # (너무 길 수 있으니 500자만 출력)

    # (만약 검색 결과가 아예 없다면 LLM을 호출할 필요 없이 바로 반환)
    if not context_str:
        print("DEBUG: FAISS 검색 결과가 없어 빈 문자열을 반환합니다.") # 👈 [추가]
        return "오류: RAG 검색 결과가 없습니다. (벡터DB에 관련 내용 없음)"

    input_for_final_chain = {"context": context_str, "question": query}
    
    final_result = final_generation_chain.invoke(input_for_final_chain)
    
    print(f"DEBUG: 최종 반환 (후보 목록):\n{final_result}") # 👈 [추가]
    return final_result

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
def optimize_and_get_routes(places: List[str]) -> str:
    """
    (수정됨) 여러 장소(places)의 최적 방문 순서를 'distance_matrix' API로 계산하고,
    '최적 순서'와 '예상 총 이동 시간'만 반환합니다. (상세 경로X)
    """
    if not GMAPS_CLIENT:
        return "오류: Google Maps API 키가 설정되지 않아 경로를 조회할 수 없습니다."
    
    if not places or len(places) < 2:
        return "오류: 경로를 최적화하려면 2개 이상의 장소가 필요합니다."

    # 👈 [디버그] 함수명 변경 식별
    print(f"\n--- [DEBUG] optimize_and_get_routes (v2 - 상세경로 제외) 호출됨 ---") 
    print(f"DEBUG: Input places: {places}")

    # --- 1단계: Distance Matrix API 호출 ---
    now = datetime.datetime.now()
    try:
        print("DEBUG: Distance Matrix API 호출 시도...")
        matrix_result = GMAPS_CLIENT.distance_matrix(origins=places,
                                                     destinations=places,
                                                     mode="transit",
                                                     departure_time=now)
        print("DEBUG: Distance Matrix API 호출 성공.")
    except Exception as e:
        print(f"!!!!!!!!!! [DEBUG] optimize_and_get_routes (Matrix API) 예외 발생 !!!!!!!!!!")
        print(f"DEBUG: Error details: {e}")
        return f"오류: Google Distance Matrix API 호출 중 문제 발생: {e}"

    # --- 2단계: 경로 최적화 (단순화된 TSP) ---
    try:
        print("DEBUG: Distance Matrix 결과 파싱 및 최적화 시작...")
        duration_matrix = []
        
        for i, row in enumerate(matrix_result['rows']):
            duration_row = []
            for j, el in enumerate(row['elements']):
                if el['status'] == 'OK':
                    duration_val = el['duration']['value']
                    duration_row.append(duration_val)
                    # (로그가 너무 길어질 수 있으므로 소요 시간 개별 출력은 주석 처리)
                    # print(f"DEBUG: [ {places[i]} -> {places[j]} ] 소요 시간: {duration_val} 초")
                else:
                    print(f"DEBUG: [ {places[i]} -> {places[j]} ] 구간 경로 없음 (Status: {el['status']})")
                    duration_row.append(float('inf')) 
            duration_matrix.append(duration_row)
        
        print(f"DEBUG: 완성된 Duration Matrix (초): {duration_matrix}")

        min_duration = float('inf')
        best_order_indices = []
        other_indices = list(range(1, len(places))) 
        
        for p in permutations(other_indices):
            current_order_indices = [0] + list(p) 
            current_duration = 0
            for i in range(len(current_order_indices) - 1):
                origin_idx = current_order_indices[i]
                dest_idx = current_order_indices[i+1]
                current_duration += duration_matrix[origin_idx][dest_idx]
            if current_duration < min_duration:
                min_duration = current_duration
                best_order_indices = current_order_indices

        if min_duration == float('inf'):
            print("DEBUG: 최적화 실패 (모든 경로에 유효한 값이 없어 'inf'만 존재)")
            return "오류: 장소 간의 유효한 대중교통 경로를 찾을 수 없어 최적화에 실패했습니다."
            
        optimized_places = [places[i] for i in best_order_indices]
        print(f"DEBUG: 최적화된 순서: {optimized_places}")

    except KeyError as e:
        print(f"!!!!!!!!!! [DEBUG] optimize_and_get_routes (Matrix 파싱) 예외 발생 !!!!!!!!!!")
        print(f"DEBUG: Error details: KeyError {e}")
        return f"오류: Distance Matrix 결과 파싱 중 문제 발생: {e}"
    except Exception as e:
        print(f"!!!!!!!!!! [DEBUG] optimize_and_get_routes (최적화 로직) 예외 발생 !!!!!!!!!!")
        print(f"DEBUG: Error details: {e}")
        return f"오류: 경로 최적화 로직 중 알 수 없는 문제 발생: {e}"

    # --- 3단계: (수정됨) 상세 경로 없이 결과 요약 ---
    
    # 👈 [수정] 3단계(상세 경로 조회 루프)를 삭제하고, 2단계의 결과로만 요약본을 생성합니다.
    output_str = f"--- 🗺️ 최적 경로 제안 (총 {len(optimized_places)}곳) ---\n"
    output_str += f"계산된 최적 순서: {' → '.join(optimized_places)}\n"
    output_str += f"예상 총 이동 시간(대중교통): 약 {min_duration // 60} 분\n"
    output_str += "(참고: '총 이동 시간'은 장소 간 이동 시간의 합이며, 장소에서 머무는 시간은 제외된 수치입니다.)"

    print("DEBUG: optimize_and_get_routes (v2) 성공적으로 완료. (상세 경로 제외)")
    return output_str

# 에이전트가 사용할 도구 목록
TOOLS = [search_attractions_and_reviews, get_weather_forecast, optimize_and_get_routes] # 👈 [수정]
AVAILABLE_TOOLS = {tool.name: tool for tool in TOOLS}