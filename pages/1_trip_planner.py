# pages/1_trip_planner.py

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.graph_flow import build_graph, AgentState 
import re
from datetime import datetime

# PDF 생성을 위한 라이브러리 임포트
from fpdf import FPDF
from src.tools import get_detailed_route

# --- 1. 페이지 직접 접근 방지 ---
if not st.session_state.get("preferences_collected", False):
    st.error("⚠️ 먼저 '여행 정보 입력' 페이지에서 정보를 입력하고 저장해주세요.")
    if st.button("정보 입력 페이지로 돌아가기"):
        st.switch_page("pages/1_📝_여행_정보_입력.py")
    st.stop()

# --- PDF 생성 함수 ---
def create_itinerary_pdf(itinerary, destination, dates, weather, final_routes, total_days, route_details=None, start_location=None):
    """세션 상태 정보를 바탕으로 여행 계획 PDF를 생성합니다. (경로 정보 포함)"""
    pdf = FPDF()
    pdf.add_page()

    # 폰트 설정
    try:
        pdf.add_font('NanumGothic', '', 'NanumGothic.ttf', uni=True)
        pdf.add_font('NanumGothic', 'B', 'NanumGothicBold.ttf', uni=True)
        pdf.set_font('NanumGothic', '', 12)
    except RuntimeError:
        print("PDF ERROR: 폰트 파일을 찾을 수 없습니다.")
        return None

    # 1. 표지
    pdf.set_font_size(24)
    pdf.cell(0, 20, f"{destination} 여행 계획", ln=True, align='C')
    pdf.set_font_size(16)
    pdf.cell(0, 10, f"기간: {dates}", ln=True, align='C')
    if start_location:
        pdf.set_font_size(12)
        pdf.cell(0, 8, f"출발지/숙소: {start_location}", ln=True, align='C')
    pdf.ln(20)

    # 2. 일차별 계획
    # 원본 순서를 유지하면서 day와 인덱스로 정렬 (같은 day 내 순서 보장)
    sorted_itinerary = sorted(enumerate(itinerary), key=lambda x: (x[1]['day'], x[0]))
    sorted_itinerary = [item[1] for item in sorted_itinerary]  # 인덱스 제거

    # 첫 일차를 위한 새 페이지
    pdf.add_page()

    for day_num in range(1, total_days + 1):
        # 첫 일차가 아니면 여유 공간 추가 (페이지는 자동으로 넘어감)
        if day_num > 1:
            pdf.ln(15)  # 일차 사이 여유 공간

        pdf.set_font_size(18)
        pdf.cell(0, 15, f"Day {day_num}", ln=True)

        places_today = [item for item in sorted_itinerary if item['day'] == day_num]

        if not places_today:
            pdf.set_font_size(12)
            pdf.cell(0, 10, "  - 계획된 장소가 없습니다.", ln=True)
            pdf.ln(10)  # 빈 일차 후 여유 공간
            continue

        for i, item in enumerate(places_today):
            # 장소 이름
            pdf.set_font('NanumGothic', 'B', 12)
            pdf.cell(0, 8, f"  - [{item.get('type', '장소')}] {item.get('name', '이름 없음')}", ln=True)

            # 설명
            if item.get('description'):
                pdf.set_font('NanumGothic', '', 10)
                pdf.set_x(15)
                pdf.multi_cell(0, 5, f"    └ {item['description']}")
                pdf.ln(2)

            # [추가됨] 다음 장소로 가는 경로 정보 출력
            if i < len(places_today) - 1 and route_details:
                # 저장할 때 썼던 키와 동일한 규칙으로 찾기 (DayN_0, DayN_1 ...)
                route_key = f"Day{day_num}_{i}"
                info = route_details.get(route_key)

                if info:
                    pdf.set_text_color(100, 100, 100) # 회색
                    pdf.set_font('NanumGothic', '', 9)
                    # "⬇️ [BUS] 143번 (약 20분)" 형태로 출력
                    step_summary = info['steps'][0] if info['steps'] else "이동"
                    pdf.set_x(15)
                    pdf.cell(0, 6, f"      ⬇️ {step_summary} ({info['duration']})", ln=True)
                    pdf.set_text_color(0, 0, 0) # 다시 검정
                    pdf.ln(2)

        # 일차별 구분선과 메모 공간
        pdf.ln(10)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(5)
        pdf.set_font_size(14)
        pdf.cell(0, 10, "메모:", ln=True)
        pdf.ln(20)  # 메모 공간 (페이지 넘김용 40에서 20으로 줄임)

    # 3. 종합 정보
    pdf.add_page()
    pdf.set_font_size(18)
    pdf.cell(0, 15, "종합 정보", ln=True)
    
    pdf.set_font_size(14)
    pdf.cell(0, 10, "[날씨 정보]", ln=True)
    pdf.set_font_size(10)
    pdf.multi_cell(0, 5, weather)
    pdf.ln(10)

    pdf.set_font_size(14)
    pdf.cell(0, 10, "[최적 경로 요약]", ln=True)
    pdf.set_font_size(10)
    pdf.multi_cell(0, 5, final_routes)
    pdf.ln(10)
    
    return bytes(pdf.output())

# --- 2. 페이지 설정 및 AI 에이전트 로딩 ---
st.set_page_config(page_title="AI 여행 플래너", layout="centered")
st.title("💬 AI 여행 플래너")
st.caption(f"'{st.session_state.get('destination', '알 수 없는 목적지')}' 여행 계획을 시작합니다.")

# --- 좌측 사이드바 가이드 추가 ---
with st.sidebar:
    # ===== 1. 현재 여행 정보 =====
    st.header("📍 현재 여행 정보")

    st.markdown(f"**목적지:** {st.session_state.get('destination', '-')}")
    if st.session_state.get('start_location'):
        st.markdown(f"**출발지:** {st.session_state.get('start_location', '-')}")
    st.markdown(f"**여행 기간:** {st.session_state.get('dates', '-')}")

    st.markdown("---")

    # ===== 2. 사용 가이드 =====
    st.header("💡 사용 가이드")

    st.markdown("""
    **기본 질문 예시**
    - "다음 날 계획을 알려줘"
    - "맛집 추가해줘"
    - "카페 추천해줘"
    - "1일차 계획 다시 알려줘"

    **장소 추가/변경**
    - "[지역명] 관광지 추가해줘"
    - "실내 활동으로 바꿔줘"
    - "사진 찍기 좋은 곳 추천해줘"

    **계획 수정**
    - 날씨에 맞는 대안 요청
    - 이동 시간을 고려한 재배치
    - 특정 테마의 장소 추천

    **완료 후**
    - PDF 다운로드로 상세 일정 저장
    - 이동 경로 및 소요시간 포함
    """)

@st.cache_resource
def get_graph_app():
    return build_graph()

APP = get_graph_app()

# --- 3. 세션 상태 초기화 (안전장치) ---
if "messages" not in st.session_state: st.session_state.messages = []
if "itinerary" not in st.session_state: st.session_state.itinerary = []
if "current_planning_day" not in st.session_state: st.session_state.current_planning_day = 1
if "total_days" not in st.session_state: st.session_state.total_days = 1
if "activity_level" not in st.session_state: st.session_state.activity_level = 3
if "current_weather" not in st.session_state: st.session_state.current_weather = ""
if "destination" not in st.session_state: st.session_state.destination = ""
if "start_location" not in st.session_state: st.session_state.start_location = ""
if "dates" not in st.session_state: st.session_state.dates = ""
if "preference" not in st.session_state: st.session_state.preference = ""
if "show_pdf_button" not in st.session_state: st.session_state.show_pdf_button = False

# --- 4. 자동 첫 메시지 생성 ---
if not st.session_state.messages:
    # 출발지 정보가 있으면 포함
    start_location_text = ""
    if st.session_state.start_location:
        start_location_text = f"\n    - **출발지/숙소:** {st.session_state.start_location}"

    initial_prompt = f"""
    안녕하세요! 방금 입력한 정보를 바탕으로 여행 계획을 시작해주세요.

    ### 입력된 여행 정보 요약
    - **목적지:** {st.session_state.destination}{start_location_text}
    - **여행 기간:** {st.session_state.dates} (총 {st.session_state.total_days}일)
    - **하루 활동량(목표 장소 수):** {st.session_state.activity_level}곳
    - **나의 여행 스타일 및 요청사항:**
    {st.session_state.preference}

    이제 위 정보를 바탕으로 1일차 계획 추천을 시작해주세요.
    """
    st.session_state.messages.append(HumanMessage(content=initial_prompt))

# --- 5. 상태 업데이트 파싱 로직 ---
def update_state_from_message(message_text):
    # [안전장치] message_text가 문자열이 아닌 경우 처리
    if not isinstance(message_text, str):
        if isinstance(message_text, list):
            text_parts = []
            for item in message_text:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
            message_text = "\n".join(text_parts)
        else:
            message_text = str(message_text)
            
    if not message_text:
        return

    match_plan = re.search(r"'(.*?)'을/를 (\d+)일차 (관광지|식당|카페) 계획에 추가합니다", message_text)
    if match_plan:
        place_name, day, place_type = match_plan.groups()
        new_item = {'day': int(day), 'type': place_type, 'name': place_name}
        if new_item not in st.session_state.itinerary:
            st.session_state.itinerary.append(new_item)

    if "[STATE_UPDATE: increment_day=True]" in message_text:
        st.session_state.current_planning_day += 1

    if "[STATE_UPDATE: show_pdf_button=True]" in message_text:
        st.session_state.show_pdf_button = True

    match_state = re.search(r"\[STATE_UPDATE:\s*(.*?)\]", message_text, re.DOTALL)
    if match_state:
        for key, value in re.findall(r'(\w+)\s*=\s*"(.*?)"', match_state.group(1)):
            if hasattr(st.session_state, key):
                if key in ["total_days", "activity_level", "current_planning_day"]:
                    try: value = int(value)
                    except ValueError: pass
                setattr(st.session_state, key, value)


# --- 6. UI 및 메인 실행 로직 ---
def run_ai_agent():
    inputs = {
        "messages": st.session_state.messages,
        "itinerary": st.session_state.itinerary,
        "destination": st.session_state.destination,
        "dates": st.session_state.dates,
        "preference": st.session_state.preference,
        "total_days": st.session_state.total_days,
        "activity_level": st.session_state.activity_level,
        "current_planning_day": st.session_state.current_planning_day,
        "current_weather": st.session_state.current_weather,
        "show_pdf_button": st.session_state.show_pdf_button,
    }

    response = APP.invoke(inputs)

    st.session_state.messages = response.get('messages', st.session_state.messages)
    st.session_state.itinerary = response.get('itinerary', st.session_state.itinerary)
    if response.get('current_weather'):
        st.session_state.current_weather = response['current_weather']

    if response.get('show_pdf_button'):
        st.session_state.show_pdf_button = True

    final_message = st.session_state.messages[-1] if st.session_state.messages else None
    if isinstance(final_message, AIMessage) and final_message.content:
        update_state_from_message(final_message.content)

# 이전 대화 기록 UI 출력
for msg in st.session_state.messages:
    content_to_display = msg.content
    
    if not isinstance(content_to_display, str):
        if isinstance(content_to_display, list):
            text_parts = []
            for item in content_to_display:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
            content_to_display = "\n".join(text_parts)
        else:
            content_to_display = str(content_to_display)

    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(content_to_display)
    elif isinstance(msg, AIMessage) and content_to_display:
        cleaned_text = re.sub(
            r"\[FINAL_ITINERARY_JSON\].*?\[/FINAL_ITINERARY_JSON\]", 
            "", 
            content_to_display, 
            flags=re.DOTALL
        )
        cleaned_text = re.sub(
            r"\[(STATE_UPDATE|PLAN_ADD):.*?\]", 
            "", 
            cleaned_text, 
            flags=re.DOTALL
        )
        display_text = cleaned_text.strip()
        if display_text:
            st.chat_message("assistant").markdown(display_text)

# --- [수정된] PDF 다운로드 및 경로 분석 섹션 ---
if st.session_state.get("show_pdf_button", False):
    
    # 1. 상세 경로 계산 버튼 (PDF 위에 배치)
    if st.session_state.itinerary:
        st.markdown("---")
        st.subheader("🗺️ 실시간 이동 경로 분석")
        
        # 경로 정보 저장소 초기화
        if "route_details" not in st.session_state:
            st.session_state.route_details = {} 

        # [계산 로직] 버튼 클릭 시
        if st.button("🚀 상세 이동 경로 및 소요시간 계산하기"):
            with st.spinner("구글 지도에서 실시간 교통 정보를 가져오는 중입니다..."):
                # [핵심 수정] 날짜별로 장소를 분류해야 인덱스(i)를 0부터 다시 셀 수 있음
                places_by_day = {}
                # 원본 순서를 유지하면서 day와 인덱스로 정렬 (같은 day 내 순서 보장)
                sorted_all = sorted(enumerate(st.session_state.itinerary), key=lambda x: (x[1]['day'], x[0]))
                sorted_all = [item[1] for item in sorted_all]
                for item in sorted_all:
                    d = item['day']
                    if d not in places_by_day: places_by_day[d] = []
                    places_by_day[d].append(item)
                
                temp_routes = {}
                
                # 날짜별 루프
                for day_num, places in places_by_day.items():
                    for i in range(len(places) - 1):
                        start = places[i]
                        end = places[i+1]
                        
                        # 키 생성 규칙: Day{날짜}_{순번} (예: Day2_0)
                        # 이렇게 해야 PDF 함수 및 아래 표시 로직과 번지수가 맞음
                        route_key = f"Day{day_num}_{i}"
                        
                        # tools.py 함수 호출
                        route_info = get_detailed_route(
                            start['name'], 
                            end['name'], 
                            mode="transit"
                        )
                        
                        if route_info:
                            temp_routes[route_key] = route_info
                
                st.session_state.route_details = temp_routes
                st.success("경로 분석 완료! 아래 PDF를 다운로드하면 이동 정보가 포함됩니다.")
                st.rerun() # 화면 갱신

        # [표시 로직] 계산된 경로가 있으면 화면에 보여주기
        if st.session_state.get("route_details"):
            # 원본 순서를 유지하면서 day와 인덱스로 정렬 (같은 day 내 순서 보장)
            sorted_all = sorted(enumerate(st.session_state.itinerary), key=lambda x: (x[1]['day'], x[0]))
            sorted_all = [item[1] for item in sorted_all]

            # [핵심 수정] 표시할 때도 날짜별로 분류해서 키를 찾아야 함
            places_by_day_display = {}
            for item in sorted_all:
                d = item['day']
                if d not in places_by_day_display: places_by_day_display[d] = []
                places_by_day_display[d].append(item)

            for day_num, places in places_by_day_display.items():
                # 날짜별 이동 경로 표시
                for i in range(len(places) - 1):
                    start = places[i]
                    end = places[i+1]
                    
                    # 키 생성 (위 계산 로직과 동일)
                    key = f"Day{day_num}_{i}"
                    
                    info = st.session_state.route_details.get(key)
                    
                    if info:
                        steps_str = " -> ".join(info['steps']) if info['steps'] else "도보/이동"
                        with st.expander(f"📍 Day {day_num} | {start['name']} ➡️ {end['name']} ({info['duration']})"):
                            st.write(f"**총 거리:** {info['distance']}")
                            st.info(f"**이동 경로:** {steps_str}")
                    else:
                        with st.expander(f"Day {day_num} | {start['name']} ➡️ {end['name']}"):
                            st.caption("경로 정보를 불러올 수 없습니다.")

    # 2. PDF 생성 및 다운로드 로직
    final_routes_text = ""
    for msg in reversed(st.session_state.messages):
        if isinstance(msg, AIMessage) and "최적 경로 제안" in msg.content:
            final_routes_text = re.sub(r"\[(STATE_UPDATE|PLAN_ADD):.*?\]", "", msg.content, flags=re.DOTALL).strip()
            break 
    
    if not final_routes_text:
        final_routes_text = "최적 경로가 아직 계산되지 않았습니다."

    pdf_bytes = create_itinerary_pdf(
        itinerary=st.session_state.itinerary,
        destination=st.session_state.destination,
        dates=st.session_state.dates,
        weather=st.session_state.current_weather,
        final_routes=final_routes_text,
        total_days=st.session_state.total_days,
        route_details=st.session_state.get("route_details"),
        start_location=st.session_state.get("start_location")
    )
    
    if pdf_bytes:
        st.download_button(
            label="📄 여행 계획 PDF 다운로드 (이동 경로 포함)",
            data=pdf_bytes,
            file_name=f"{st.session_state.destination}_여행계획_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
    else:
        st.error("PDF 파일 생성 실패: 폰트 파일을 확인해주세요.")

# 최초 실행 또는 사용자 입력 시 AI 호출
if 'last_message_count' not in st.session_state:
    st.session_state.last_message_count = 0

if len(st.session_state.messages) == 1 and st.session_state.last_message_count == 0:
    with st.chat_message("assistant"):
        with st.spinner("AI 전문가 팀이 회의 중입니다..."):
            run_ai_agent()
    st.session_state.last_message_count = len(st.session_state.messages)
    st.rerun()

# 사용자 입력 처리
if user_input := st.chat_input(f"'{st.session_state.destination}' 여행에 대해 더 물어보세요"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("AI 전문가 팀이 회의 중입니다..."):
            run_ai_agent()

    st.session_state.last_message_count = len(st.session_state.messages)
    st.rerun()