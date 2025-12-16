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
from src.config import load_faiss_index

with st.spinner("여행 데이터를 불러오는 중입니다..."):
    DB = load_faiss_index()
    print("1_trip_planner 페이지에서 DB 로드 확인 완료")

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

def translate_category_to_korean(category):
    """카테고리를 한글로 변환"""
    if not category:
        return "활동"

    category_lower = str(category).lower()

    if 'activity' in category_lower:
        return "활동"
    elif 'restaurant' in category_lower or 'food' in category_lower:
        return "식당"
    elif 'cafe' in category_lower or 'coffee' in category_lower:
        return "카페"
    elif 'tourist' in category_lower or 'attraction' in category_lower:
        return "관광지"
    elif 'move' in category_lower or 'transport' in category_lower:
        return "이동"

    return category

def _normalize_itinerary_for_pdf(itinerary, total_days=None):
    norm = []
    for item in itinerary:
        if not isinstance(item, dict):
            continue
        it = item.copy()
        
        day = it.get('day', 1)
        if isinstance(day, str):
            m = re.search(r'(\d+)', day)
            try:
                day = int(m.group(1)) if m else 1
            except:
                day = 1
        else:
            try: day = int(day)
            except: day = 1
            
        if total_days:
            try:
                td = int(total_days)
                if day < 1: day = 1
                if day > td: day = td
            except: pass
            
        it['day'] = day
        if 'description' not in it: it['description'] = it.get('description', '')
        if 'type' not in it and 'category' in it: it['type'] = it.get('category')
        if 'name' not in it: it['name'] = it.get('장소명', it.get('name', '이름 없음'))
        if 'reviews' not in it: it['reviews'] = [] 
        
        norm.append(it)
    return norm

def create_itinerary_pdf(itinerary, destination, dates, weather, final_routes, total_days, start_location=None):
    pdf = FPDF()
    pdf.add_page()

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
            pdf.set_font('Arial', '', 12)
    except Exception as e:
        print(f" [PDF 생성] 폰트 로드 에러: {e}")
        pdf.set_font('Arial', '', 12)

    pdf.set_font_size(24)
    pdf.cell(0, 20, text=f"{destination} 여행 계획", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    pdf.set_font_size(12)
    pdf.cell(0, 10, text=f"기간: {dates}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    if start_location:
        pdf.set_font_size(11)
        pdf.cell(0, 8, text=f"출발지/숙소: {start_location}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    if weather and weather.strip() and weather != '정보 없음':
        pdf.set_font_size(10)
        pdf.multi_cell(0, 5, text=f"날씨: {weather}", align='C')

    pdf.ln(10)

    normalized_itinerary = _normalize_itinerary_for_pdf(itinerary, total_days)
    try:
        sorted_itinerary = sorted(enumerate(normalized_itinerary), key=lambda x: (int(x[1].get('day', 1)), x[1].get('start', '00:00') or '00:00', x[0]))
        sorted_itinerary = [item[1] for item in sorted_itinerary]
    except:
        sorted_itinerary = normalized_itinerary

    for day_num in range(1, total_days + 1):
        if day_num > 1: pdf.ln(15) 

        pdf.set_font_size(18)
        if has_korean_font: pdf.set_font('NanumGothic', 'B', 18)
        pdf.cell(0, 15, text=f"Day {day_num}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font_size(11)
        if has_korean_font: pdf.set_font('NanumGothic', '', 11)

        items_today = [item for item in sorted_itinerary if int(item.get('day', 1)) == day_num]

        if not items_today:
            pdf.cell(0, 10, text="  - 계획된 일정이 없습니다.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(10)
            continue

        for item in items_today:
            item_type = item.get('type', 'activity')

            if item_type == 'move':
                pdf.set_text_color(100, 100, 100)
                pdf.set_font_size(10)
                move_text = f"      |  {item.get('start', '')} ~ {item.get('end', '')} ({item.get('duration_text', '')}) : {item.get('transport', '이동')}"
                pdf.cell(0, 8, text=move_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_text_color(0, 0, 0)
                pdf.set_font_size(11)

            else:
                time_info = f"[{item.get('start', '시간 미정')}-{item.get('end', '')}]" if item.get('start') else "[시간 미정]"

                if has_korean_font: pdf.set_font('NanumGothic', 'B', 12)
                category_text = translate_category_to_korean(item.get('category', item_type))
                main_text = f"  ● {time_info} {item.get('name', '이름 없음')} ({category_text})"
                pdf.cell(0, 8, text=main_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

                if item.get('description'):
                    if has_korean_font: pdf.set_font('NanumGothic', '', 10)
                    pdf.set_x(20)
                    pdf.multi_cell(0, 5, text=f"{item['description']}")
                    pdf.ln(2)

                reviews = item.get('reviews', [])
                if reviews and isinstance(reviews, list):
                    if has_korean_font: pdf.set_font('NanumGothic', '', 9)
                    pdf.set_x(20)
                    for review in reviews:
                        if isinstance(review, str): review_text = review
                        elif isinstance(review, dict): review_text = review.get('text', str(review))
                        else: review_text = str(review)
                        pdf.multi_cell(0, 4, text=f"  • {review_text}")
                    pdf.ln(2)

        pdf.ln(10)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(5)
        pdf.set_font_size(14)
        if has_korean_font: pdf.set_font('NanumGothic', '', 14)
        pdf.cell(0, 10, text="메모:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(20)

    return bytes(pdf.output())

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

if "preferences_collected" not in st.session_state:
    st.warning("⚠️ 정보 입력 페이지에서 먼저 여행 정보를 입력해주세요.")
    if st.button("테스트용 임시 데이터 로드"):
        st.session_state.destination = "부산 해운대"
        st.session_state.dates = "2025-12-06 (1일)"
        st.session_state.total_days = 1
        st.session_state.preference = "맛집 탐방"
        st.session_state.group_type = "친구"
        st.session_state.preferences_collected = True
        st.rerun()
    st.stop()

if "messages" not in st.session_state: st.session_state.messages = []
if "itinerary" not in st.session_state: st.session_state.itinerary = []
if "show_pdf_button" not in st.session_state: st.session_state.show_pdf_button = False
if "current_weather" not in st.session_state: st.session_state.current_weather = ""
if "current_anchor" not in st.session_state: st.session_state.current_anchor = ""
if "dialog_stage" not in st.session_state: st.session_state.dialog_stage = "planning"
if "last_deleted_spot" not in st.session_state: st.session_state.last_deleted_spot = None
if "ban_list" not in st.session_state: st.session_state.ban_list = []

if "event_loop" not in st.session_state:
    st.session_state.event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.event_loop)
loop = st.session_state.event_loop

def get_graph_app():
    return build_graph()

APP = get_graph_app()

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
        "current_weather": st.session_state.get('current_weather', ''),
        "show_pdf_button": st.session_state.get('show_pdf_button', False),
        "current_anchor": st.session_state.get('current_anchor', st.session_state.get('destination', '')),
        "dialog_stage": st.session_state.get("dialog_stage", "planning"),
        "last_deleted_spot": st.session_state.get("last_deleted_spot"),
        "ban_list": st.session_state.get("ban_list", [])
    }
    
    with st.spinner("AI가 여행 계획을 생성/수정 중입니다..."):
        response = await APP.ainvoke(current_state, config=config)

    st.session_state.messages = response.get('messages', [])
    st.session_state.itinerary = response.get('itinerary', [])
    
    try:
        st.session_state.itinerary = _normalize_itinerary_for_pdf(st.session_state.itinerary, st.session_state.get('total_days', None))
    except Exception as e:
        print("itinerary 정규화 실패:", e)

    st.session_state.current_weather = response.get('current_weather', '')
    st.session_state.show_pdf_button = response.get('show_pdf_button', False)
    st.session_state.current_anchor = response.get('current_anchor', '')
    
    if 'dialog_stage' in response:
        st.session_state.dialog_stage = response['dialog_stage']

    if 'last_deleted_spot' in response:
        st.session_state.last_deleted_spot = response['last_deleted_spot']

    if 'ban_list' in response:
        st.session_state.ban_list = response['ban_list']

# --- 6. 초기 실행 트리거 ---
if not st.session_state.messages:
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(time.time())

    initial_prompt = f"""
    안녕하세요! 아래 정보로 여행 계획을 세워주세요.
    - 목적지: {st.session_state.get('destination')}
    - 일정: {st.session_state.get('dates')} (총 {st.session_state.get('total_days')}일)
    - 스타일: {st.session_state.get('preference')}
    - 동행: {st.session_state.get('group_type')}
    
    날씨 확인 후, 1일차 일정부터 바로 시작해주세요.
    """
    st.session_state.messages.append(HumanMessage(content=initial_prompt))
    
    loop.run_until_complete(run_ai_agent())
    st.rerun()

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        content_str = normalize_to_string(msg.content)
        if content_str.strip():
            clean_content = re.sub(r"\[(ADD|REPLACE|DELETE)_PLACE\].*?\[/\1_PLACE\]", "", content_str, flags=re.DOTALL)
            if "FINISH" in clean_content and len(clean_content) < 10:
                continue
            if clean_content.strip():
                st.chat_message("assistant").markdown(clean_content)

if st.session_state.show_pdf_button:
    weather_info = st.session_state.get('current_weather', '날씨 정보 없음')
    
    with st.expander("📊 PDF 생성 데이터 확인 (Debug)", expanded=False):
        st.write("데이터 구조 검증 중...")
        st.json(st.session_state.itinerary)

    pdf_bytes = create_itinerary_pdf(
        st.session_state.itinerary,
        st.session_state.destination,
        st.session_state.dates,
        weather_info,
        "", 
        st.session_state.total_days,
        st.session_state.get("start_location") 
    )
    
    if pdf_bytes:
        st.success("✅ PDF 생성 완료!")
        st.download_button(
            label="📄 여행 계획 PDF 다운로드",
            data=pdf_bytes,
            file_name=f"{st.session_state.destination}_여행계획.pdf",
            mime="application/pdf"
        )
    else:
        st.error("PDF 생성 실패")

# --- 9. 사용자 입력 처리 ---
if user_input := st.chat_input("메시지를 입력하세요..."):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").markdown(user_input)
    loop.run_until_complete(run_ai_agent())
    st.rerun()