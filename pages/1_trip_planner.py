# pages/1_trip_planner.py

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.graph_flow import build_graph
import re
import json
from datetime import datetime
from fpdf import FPDF
import time

# --- 1. 헬퍼 함수: 무조건 안전한 문자열로 변환 ---
def normalize_to_string(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 멀티모달 리스트 처리 [{'text': '...'}]
        texts = []
        for item in content:
            if isinstance(item, dict):
                texts.append(str(item.get('text', '')))
            else:
                texts.append(str(item))
        return "\n".join(texts)
    return str(content)

# --- 2. PDF 생성 함수 ---
# pages/1_trip_planner.py

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

def create_itinerary_pdf(itinerary, destination, dates, weather, final_routes, total_days):
    """
    여행 계획 PDF 생성 함수 (SmartScheduler 이동 경로 반영 & 최신 FPDF2 문법 적용)
    """
    pdf = FPDF()
    pdf.add_page()
    
    # 1. 폰트 로드 (NanumGothic)
    font_path = 'NanumGothic.ttf'
    bold_font_path = 'NanumGothicBold.ttf'
    
    # 폰트 파일 존재 여부 확인 및 등록
    has_korean_font = False
    try:
        if os.path.exists(font_path):
            pdf.add_font('NanumGothic', '', font_path)
            if os.path.exists(bold_font_path):
                pdf.add_font('NanumGothic', 'B', bold_font_path)
            else:
                pdf.add_font('NanumGothic', 'B', font_path) # 볼드 없으면 일반으로 대체
            
            pdf.set_font('NanumGothic', '', 12)
            has_korean_font = True
        else:
            print("⚠️ [PDF 생성] 폰트 파일이 없습니다. 기본 폰트(Arial)를 사용합니다.")
            pdf.set_font('Arial', '', 12)
    except Exception as e:
        print(f"⚠️ [PDF 생성] 폰트 로드 에러: {e}")
        return None

    # 2. 헤더 (여행지 및 기간)
    pdf.set_font_size(24)
    # ln=True -> new_x=XPos.LMARGIN, new_y=YPos.NEXT 로 변경
    pdf.cell(0, 20, text=f"{destination} 여행 계획", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    pdf.set_font_size(12)
    pdf.cell(0, 10, text=f"기간: {dates} | 날씨: {weather if weather else '정보 없음'}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)

    # 안전한 정렬 (day 키 기준)
    try:
        sorted_itinerary = sorted(itinerary, key=lambda x: int(x.get('day', 1)))
    except:
        sorted_itinerary = itinerary

    # 3. 일차별 상세 일정 작성
    for day_num in range(1, total_days + 1):
        # 날짜 헤더
        pdf.set_font_size(18)
        if has_korean_font: pdf.set_font('NanumGothic', 'B', 18)
        
        pdf.cell(0, 15, text=f"Day {day_num}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # 폰트 원복
        pdf.set_font_size(11)
        if has_korean_font: pdf.set_font('NanumGothic', '', 11)

        # 해당 날짜 아이템 필터링
        items_today = [item for item in sorted_itinerary if int(item.get('day', 1)) == day_num]
        
        if not items_today:
            pdf.cell(0, 10, text="  - 계획된 일정이 없습니다.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
            continue

        for item in items_today:
            item_type = item.get('type', 'activity')

            # --- [Case A] 이동 정보 (SmartScheduler가 생성한 'move') ---
            if item_type == 'move':
                # 이동은 회색으로 작게 표시하여 시각적 구분
                pdf.set_text_color(100, 100, 100) # Gray
                pdf.set_font_size(10)

                start_t = item.get('start', '')
                end_t = item.get('end', '')
                duration = item.get('duration_text', '')
                transport = item.get('transport', '이동')

                # 🚨 [수정] 이모지 대신 화살표 문자 사용 (폰트 호환성)
                move_text = f"      |  {start_t} ~ {end_t} ({duration}) : {transport}"
                pdf.cell(0, 8, text=move_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

                # 색상 및 크기 원복
                pdf.set_text_color(0, 0, 0) # Black
                pdf.set_font_size(11)

            # --- [Case B] 장소 방문 (activity/식당/관광지 등) ---
            else:
                # 시간 정보
                start_t = item.get('start', '')
                end_t = item.get('end', '')
                time_info = f"[{start_t}-{end_t}]" if start_t else "[시간 미정]"
                
                place_name = item.get('name', '이름 없음')
                category = item.get('category', item.get('type', '장소'))
                
                # 제목 라인 (볼드)
                if has_korean_font: pdf.set_font('NanumGothic', 'B', 12)
                
                main_text = f"  ● {time_info} {place_name} ({category})"
                pdf.cell(0, 8, text=main_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                
                # 설명 라인 (일반)
                if item.get('description'):
                    if has_korean_font: pdf.set_font('NanumGothic', '', 10)
                    
                    # 들여쓰기 후 설명 출력
                    pdf.set_x(20) 
                    pdf.multi_cell(0, 5, text=f"{item['description']}")
                    pdf.ln(2)

        pdf.ln(10) # 날짜 간 간격

    return bytes(pdf.output())


# --- 3. 페이지 설정 ---
st.set_page_config(page_title="AI 여행 플래너", layout="centered")
st.title("💬 AI 여행 플래너")

if "preferences_collected" not in st.session_state:
    st.warning("⚠️ 정보 입력 페이지에서 먼저 여행 정보를 입력해주세요.")
    if st.button("돌아가기"):
        st.switch_page("pages/1_📝_여행_정보_입력.py")
    st.stop()

# 세션 초기화
if "messages" not in st.session_state: st.session_state.messages = []
if "itinerary" not in st.session_state: st.session_state.itinerary = []
if "current_planning_day" not in st.session_state: st.session_state.current_planning_day = 1
if "total_days" not in st.session_state: st.session_state.total_days = 1
if "show_pdf_button" not in st.session_state: st.session_state.show_pdf_button = False
if "destination" not in st.session_state: st.session_state.destination = ""
if "current_weather" not in st.session_state: st.session_state.current_weather = ""

@st.cache_resource
def get_graph_app():
    return build_graph()

APP = get_graph_app()

# --- 4. 상태 업데이트 로직 (안전장치 적용) ---
def update_state_from_message(message_content):
    # 🚨 [핵심] 입력값을 무조건 문자열로 변환
    message_text = normalize_to_string(message_content)

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

# --- 5. AI 에이전트 실행 ---
def run_ai_agent():
    config = {"configurable": {"thread_id": "streamlit_user"}}
    
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
    
    with st.spinner("AI가 생각 중입니다..."):
        response = APP.invoke(inputs, config=config)
    
    st.session_state.messages = response.get('messages', st.session_state.messages)
    st.session_state.itinerary = response.get('itinerary', st.session_state.itinerary)
    
    if response.get('current_weather'):
        st.session_state.current_weather = response['current_weather']
    
    if response.get('show_pdf_button'):
        st.session_state.show_pdf_button = True

    # 마지막 메시지 처리
    if st.session_state.messages:
        final_msg = st.session_state.messages[-1]
        if isinstance(final_msg, AIMessage):
            update_state_from_message(final_msg.content)

            if "[STATE_UPDATE: show_pdf_button=True]" in normalize_to_string(final_msg.content):
                st.rerun()

# --- 6. 초기 실행 ---
if not st.session_state.messages:
    initial_prompt = f"""
    안녕하세요! 방금 입력한 정보를 바탕으로 여행 계획을 시작해주세요.
    - 목적지: {st.session_state.destination}
    - 여행 기간: {st.session_state.dates}
    - 나의 여행 스타일: {st.session_state.preference}
    
    이제 위 정보를 바탕으로 1일차 계획 추천을 시작해주세요.
    """
    st.session_state.messages.append(HumanMessage(content=initial_prompt))
    run_ai_agent() # 첫 실행
    st.rerun()

# --- 7. 화면 출력 ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage):
        # 안전한 변환
        safe_content = normalize_to_string(msg.content)
        
        # 태그 제거
        cleaned_text = re.sub(r"\[FINAL_ITINERARY_JSON\].*?\[/FINAL_ITINERARY_JSON\]", "", safe_content, flags=re.DOTALL)
        cleaned_text = re.sub(r"\[(STATE_UPDATE|PLAN_ADD):.*?\]", "", cleaned_text, flags=re.DOTALL)
        
        if cleaned_text.strip():
            st.chat_message("assistant").markdown(cleaned_text.strip())

# --- 8. PDF 다운로드 ---
if st.session_state.show_pdf_button:
    # 경로 정보 안전하게 추출
    final_routes_text = "경로 정보 없음"
    for msg in reversed(st.session_state.messages):
        if isinstance(msg, AIMessage):
            c_str = normalize_to_string(msg.content)
            if "최적 경로" in c_str:
                final_routes_text = re.sub(r"\[.*?\]", "", c_str).strip()
                break
                
    pdf_bytes = create_itinerary_pdf(
        st.session_state.itinerary,
        st.session_state.destination,
        st.session_state.dates,
        st.session_state.current_weather,
        final_routes_text,
        st.session_state.total_days
    )
    if pdf_bytes:
        st.download_button(
            label="📄 여행 계획 PDF 다운로드",
            data=pdf_bytes,
            file_name=f"{st.session_state.destination}_여행계획.pdf",
            mime="application/pdf"
        )

# --- 9. 사용자 입력 ---
if user_input := st.chat_input("메시지를 입력하세요..."):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").markdown(user_input)
    run_ai_agent()
    st.rerun()