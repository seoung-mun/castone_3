import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.graph_flow import build_graph
import re
import asyncio
from datetime import datetime
from fpdf import FPDF
import time
import os
from fpdf.enums import XPos, YPos

# --- 1. 헬퍼 함수 ---
def normalize_to_string(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                texts.append(str(item.get('text', '')))
            else:
                texts.append(str(item))
        return "\n".join(texts)
    return str(content)

# --- 2. PDF 생성 함수 ---
def create_itinerary_pdf(itinerary, destination, dates, weather, final_routes, total_days):
    pdf = FPDF()
    pdf.add_page()
    
    # 폰트 설정 (한글 깨짐 방지)
    # 폰트 파일이 프로젝트 루트에 있어야 합니다. 없으면 Arial(한글 미지원)로 동작
    font_path = 'NanumGothic.ttf'
    bold_font_path = 'NanumGothicBold.ttf'
    
    has_korean_font = False
    try:
        if os.path.exists(font_path):
            pdf.add_font('NanumGothic', '', font_path)
            if os.path.exists(bold_font_path):
                pdf.add_font('NanumGothic', 'B', bold_font_path)
            else:
                pdf.add_font('NanumGothic', 'B', font_path)
            
            pdf.set_font('NanumGothic', '', 12)
            has_korean_font = True
        else:
            # 폰트 없을 시 영문 기본 폰트
            pdf.set_font('Arial', '', 12)
    except Exception as e:
        print(f"⚠️ [PDF 생성] 폰트 로드 에러: {e}")
        pdf.set_font('Arial', '', 12)

    # 타이틀
    pdf.set_font_size(24)
    pdf.cell(0, 20, text=f"{destination} 여행 계획", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    # 날짜
    pdf.set_font_size(12)
    pdf.cell(0, 10, text=f"기간: {dates}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    # 날씨
    if weather and weather.strip() and weather != '정보 없음':
        pdf.set_font_size(10)
        pdf.multi_cell(0, 5, text=f"날씨: {weather}", align='C')

    pdf.ln(10)

    # 일정 정렬
    try:
        sorted_itinerary = sorted(itinerary, key=lambda x: (int(x.get('day', 1)), x.get('start', '00:00')))
    except:
        sorted_itinerary = itinerary

    # 일자별 출력
    for day_num in range(1, total_days + 1):
        pdf.set_font_size(18)
        if has_korean_font: pdf.set_font('NanumGothic', 'B', 18)
        
        pdf.cell(0, 15, text=f"Day {day_num}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.set_font_size(11)
        if has_korean_font: pdf.set_font('NanumGothic', '', 11)

        items_today = [item for item in sorted_itinerary if int(item.get('day', 1)) == day_num]
        
        if not items_today:
            pdf.cell(0, 10, text="  - 계획된 일정이 없습니다.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
            continue

        for item in items_today:
            item_type = item.get('type', 'activity')

            # 이동(Move) 항목
            if item_type == 'move':
                pdf.set_text_color(100, 100, 100) # 회색
                pdf.set_font_size(10)
                move_text = f"      |  {item.get('start', '')} ~ {item.get('end', '')} ({item.get('duration_text', '')}) : {item.get('transport', '이동')}"
                pdf.cell(0, 8, text=move_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_text_color(0, 0, 0) # 검정색 복구
                pdf.set_font_size(11)
            
            # 장소(Activity) 항목
            else:
                time_info = f"[{item.get('start', '시간 미정')}-{item.get('end', '')}]" if item.get('start') else "[시간 미정]"
                
                if has_korean_font: pdf.set_font('NanumGothic', 'B', 12)
                main_text = f"  ● {time_info} {item.get('name', '이름 없음')} ({item.get('category', item_type)})"
                pdf.cell(0, 8, text=main_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                
                # 설명
                if item.get('description'):
                    if has_korean_font: pdf.set_font('NanumGothic', '', 10)
                    pdf.set_x(20) # 들여쓰기
                    pdf.multi_cell(0, 5, text=f"{item['description']}")
                    pdf.ln(2)
        pdf.ln(10)

    return bytes(pdf.output())

# --- 3. 페이지 설정 및 세션 초기화 ---
st.set_page_config(page_title="AI 여행 플래너", layout="centered")
st.title("💬 AI 여행 플래너")

with st.sidebar:
    st.header("질문 가이드")
    st.markdown("""
    - "근처 관광지 추천해줘"
    - "맛집 알려줘"
    - "일정 수정하고 싶어"
    - "경로 최적화해줘"
    - "PDF로 만들어줘"
    """)

# 필수 정보 체크
if "preferences_collected" not in st.session_state:
    st.warning("⚠️ 정보 입력 페이지에서 먼저 여행 정보를 입력해주세요.")
    # 로컬 테스트용 임시 버튼 (실제 배포시 제거 가능)
    if st.button("테스트용 임시 데이터 로드"):
        st.session_state.destination = "부산 해운대"
        st.session_state.dates = "2025-12-06 (1일)"
        st.session_state.total_days = 1
        st.session_state.activity_level = 3
        st.session_state.preference = "맛집 탐방"
        st.session_state.group_type = "친구"
        st.session_state.preferences_collected = True
        st.rerun()
    st.stop()

# 세션 상태 초기화
if "messages" not in st.session_state: st.session_state.messages = []
if "itinerary" not in st.session_state: st.session_state.itinerary = []
if "show_pdf_button" not in st.session_state: st.session_state.show_pdf_button = False
if "current_weather" not in st.session_state: st.session_state.current_weather = ""
if "current_anchor" not in st.session_state: st.session_state.current_anchor = ""
if "dialog_stage" not in st.session_state: st.session_state.dialog_stage = "planning"

# [수정] Asyncio 이벤트 루프 관리
# 세션 전체에서 단일 이벤트 루프를 사용하도록 설정
if "event_loop" not in st.session_state:
    st.session_state.event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.event_loop)
loop = st.session_state.event_loop

# --- 4. 그래프 로드 ---
def get_graph_app():
    return build_graph()

APP = get_graph_app()

# --- 5. AI 에이전트 실행 로직 (비동기 처리) ---
async def run_ai_agent():
    thread_id = st.session_state.session_id if 'session_id' in st.session_state else "streamlit_user"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    
    current_state = {
        "messages": st.session_state.messages,
        "itinerary": st.session_state.itinerary,
        "destination": st.session_state.get('destination', ''),
        "dates": st.session_state.get('dates', ''),
        "group_type": st.session_state.get('group_type', '정보없음'),
        "style": st.session_state.get('preference', ''),
        "preference": st.session_state.get('preference', ''),
        "total_days": st.session_state.get('total_days', 1),
        "activity_level": st.session_state.get('activity_level', 3),
        "current_weather": st.session_state.get('current_weather', ''),
        "show_pdf_button": st.session_state.get('show_pdf_button', False),
        "current_anchor": st.session_state.get('current_anchor', st.session_state.get('destination', '')),
        "dialog_stage": st.session_state.get("dialog_stage", "planning")
    }
    
    with st.spinner("AI가 여행 계획을 생성/수정 중입니다..."):
        response = await APP.ainvoke(current_state, config=config)

    st.session_state.messages = response.get('messages', [])
    st.session_state.itinerary = response.get('itinerary', [])
    st.session_state.current_weather = response.get('current_weather', '')
    st.session_state.show_pdf_button = response.get('show_pdf_button', False)
    st.session_state.current_anchor = response.get('current_anchor', '')
    
    if 'dialog_stage' in response:
        st.session_state.dialog_stage = response['dialog_stage']

# --- 6. 초기 실행 트리거 ---
if not st.session_state.messages:
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(time.time())

    initial_prompt = f"""
    안녕하세요! 아래 정보로 여행 계획을 세워주세요.
    - 목적지: {st.session_state.get('destination')}
    - 일정: {st.session_state.get('dates')} (총 {st.session_state.get('total_days')}일)
    - 활동량: 하루 {st.session_state.get('activity_level')}곳
    - 스타일: {st.session_state.get('preference')}
    - 동행: {st.session_state.get('group_type')}
    
    날씨 확인 후, 1일차 일정부터 바로 시작해주세요.
    """
    st.session_state.messages.append(HumanMessage(content=initial_prompt))
    
    # [수정] 공유된 이벤트 루프 사용
    loop.run_until_complete(run_ai_agent())
    st.rerun()

# --- 7. 채팅 화면 출력 ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        content_str = normalize_to_string(msg.content)
        if content_str.strip():
            clean_content = re.sub(r"\[(ADD|REPLACE|DELETE)_PLACE\].*?\[/\1_PLACE\]", "", msg.content, flags=re.DOTALL)
            if "FINISH" in clean_content and len(clean_content) < 10:
                continue
            if clean_content.strip():
                st.chat_message("assistant").markdown(clean_content)

# --- 8. PDF 다운로드 버튼 ---
if st.session_state.show_pdf_button:
    pdf_bytes = create_itinerary_pdf(
        st.session_state.itinerary,
        st.session_state.destination,
        st.session_state.dates,
        st.session_state.current_weather,
        "", 
        st.session_state.total_days
    )
    if pdf_bytes:
        st.download_button(
            label="📄 여행 계획 PDF 다운로드",
            data=pdf_bytes,
            file_name=f"{st.session_state.destination}_여행계획.pdf",
            mime="application/pdf"
        )

# --- 9. 사용자 입력 처리 ---
if user_input := st.chat_input("메시지를 입력하세요..."):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").markdown(user_input)
    
    # [수정] 공유된 이벤트 루프 사용
    loop.run_until_complete(run_ai_agent())
    st.rerun()