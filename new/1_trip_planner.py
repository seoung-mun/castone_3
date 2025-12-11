# pages/2_💬_AI_여행_플래너.py

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.graph_flow import build_graph, AgentState # 사용자님의 프로젝트 구조에 맞게 수정
import re
from datetime import datetime

# PDF 생성을 위한 라이브러리 임포트
from fpdf import FPDF

# --- 1. 페이지 직접 접근 방지 ---
if not st.session_state.get("preferences_collected", False):
    st.error("⚠️ 먼저 '여행 정보 입력' 페이지에서 정보를 입력하고 저장해주세요.")
    if st.button("정보 입력 페이지로 돌아가기"):
        st.switch_page("pages/1_📝_여행_정보_입력.py")
    st.stop()

# --- PDF 생성 함수 ---
def create_itinerary_pdf(itinerary, destination, dates, weather, final_routes, total_days):
    """세션 상태 정보를 바탕으로 여행 계획 PDF를 생성합니다."""
    pdf = FPDF()
    pdf.add_page()

    # !!! 중요: 한글 폰트 설정 !!!
    try:
        pdf.add_font('NanumGothic', '', 'NanumGothic.ttf', uni=True)
        pdf.set_font('NanumGothic', '', 12)
    except RuntimeError:
        print("PDF ERROR: 한글 폰트 파일('NanumGothic.ttf')을 찾을 수 없습니다. 프로젝트 폴더에 폰트 파일을 추가해주세요.")
        return None

    # Normalize itinerary days to ensure reliable grouping by day
    normalized_itinerary = _normalize_itinerary_for_pdf(itinerary, total_days)

    # 1. 표지
    pdf.set_font_size(24)
    pdf.cell(0, 20, f"{destination} 여행 계획", ln=True, align='C')
    pdf.set_font_size(16)
    pdf.cell(0, 10, f"기간: {dates}", ln=True, align='C')
    pdf.ln(20)

    # 2. 일차별 계획
    sorted_itinerary = sorted(normalized_itinerary, key=lambda x: int(x.get('day', 1)))

    for day_num in range(1, int(total_days) + 1):
        pdf.add_page()
        pdf.set_font_size(18)
        pdf.cell(0, 15, f"Day {day_num}", ln=True)
        
        places_today = [item for item in sorted_itinerary if int(item.get('day', 1)) == day_num]
        
        if not places_today:
            pdf.set_font_size(12)
            pdf.cell(0, 10, "  - 계획된 장소가 없습니다.", ln=True)
            continue

        for item in places_today:
            # 장소 이름 출력 (조금 더 굵게)
            pdf.set_font('NanumGothic', 'B', 12) # 'B' for Bold
            pdf.cell(0, 8, f"  - [{item.get('type', '장소')}] {item.get('name', '이름 없음')}", ln=True)
            
            # 설명이 있다면, 작은 글씨로 예쁘게 출력
            if item.get('description'):
                pdf.set_font('NanumGothic', '', 10) # 일반, 작은 폰트
                pdf.set_x(15) # 살짝 들여쓰기
                pdf.multi_cell(0, 5, f"    └ {item['description']}")
                pdf.ln(2) # 설명 뒤에 약간의 간격 추가
        
        pdf.ln(10)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(5)
        pdf.set_font_size(14)
        pdf.cell(0, 10, "메모:", ln=True)
        pdf.ln(40)

    # 3. 종합 정보 페이지
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
    
    # [수정된 부분] bytearray를 Streamlit이 요구하는 bytes 타입으로 변환합니다.
    return bytes(pdf.output())

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
            try:
                day = int(day)
            except:
                day = 1
        if total_days:
            try:
                td = int(total_days)
                if day < 1: day = 1
                if day > td: day = td
            except:
                pass
        it['day'] = day
        if 'description' not in it: it['description'] = it.get('description', '')
        if 'type' not in it: it['type'] = it.get('type', '장소')
        if 'name' not in it: it['name'] = it.get('name', '이름 없음')
        # ✨ [추가] reviews 필드 기본값 설정
        if 'reviews' not in it: it['reviews'] = []
        norm.append(it)
    return norm

# --- 2. 페이지 설정 및 AI 에이전트 로딩 ---
st.set_page_config(page_title="AI 여행 플래너", layout="centered")
st.title("💬 AI 여행 플래너")
st.caption(f"'{st.session_state.get('destination', '알 수 없는 목적지')}' 여행 계획을 시작합니다.")

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
if "dates" not in st.session_state: st.session_state.dates = ""
if "preference" not in st.session_state: st.session_state.preference = ""
if "show_pdf_button" not in st.session_state: st.session_state.show_pdf_button = False

# --- 4. 자동 첫 메시지 생성 ---
if not st.session_state.messages:
    initial_prompt = f"""
    안녕하세요! 방금 입력한 정보를 바탕으로 여행 계획을 시작해주세요.

    ### 입력된 여행 정보 요약
    - **목적지:** {st.session_state.destination}
    - **여행 기간:** {st.session_state.dates} (총 {st.session_state.total_days}일)
    - **하루 활동량(목표 장소 수):** {st.session_state.activity_level}곳
    - **나의 여행 스타일 및 요청사항:**
    {st.session_state.preference}
    
    이제 위 정보를 바탕으로 1일차 계획 추천을 시작해주세요.
    """
    st.session_state.messages.append(HumanMessage(content=initial_prompt))

# --- 5. 상태 업데이트 파싱 로직 ---
def update_state_from_message(message_text: str):
    # NOTE: itinerary 조작은 백엔드(Supervisor/Agent)가 반환한 itinerary를 신뢰합니다.
    # 이 함수는 오직 단순 상태값만 업데이트합니다 (current_planning_day, show_pdf_button, 기타 STATE_UPDATE 태그).

    # 사용자가 '다음 날' 또는 유사 표현을 말하면 current_planning_day 증가
    if re.search(r"(다음 ?날|다음 ?일|next ?day)", message_text, re.IGNORECASE):
        try:
            st.session_state.current_planning_day = int(st.session_state.get("current_planning_day", 1)) + 1
        except Exception:
            st.session_state.current_planning_day = st.session_state.get("current_planning_day", 1)

    # 기존의 상태 업데이트 태그 파싱: total_days, activity_level 등 기본 변수만 갱신
    if "[STATE_UPDATE: show_pdf_button=True]" in message_text:
        st.session_state.show_pdf_button = True

    match_state = re.search(r"\[STATE_UPDATE:\s*(.*?)\]", message_text, re.DOTALL)
    if match_state:
        for key, value in re.findall(r'(\w+)\s*=\s*"(.*?)"', match_state.group(1)):
            # 안전하게 허용된 키만 처리 (단순 스칼라 값)
            if key in ["total_days", "activity_level", "current_planning_day", "destination", "dates"]:
                try:
                    if key in ["total_days", "activity_level", "current_planning_day"]:
                        value_cast = int(value)
                    else:
                        value_cast = value
                    st.session_state[key] = value_cast
                except Exception:
                    # 실패 시 문자열 그대로 저장(비파괴)
                    st.session_state[key] = value
            else:
                # 기타 복잡한 구조(itinerary 등)는 무시 — 백엔드가 담당
                continue

    if "[STATE_UPDATE: increment_day=True]" in message_text:
        st.session_state.current_planning_day += 1

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
    # 응답으로 받은 itinerary 강제 정규화 (day 정수화 등)
    try:
        st.session_state.itinerary = _normalize_itinerary_for_pdf(st.session_state.itinerary, st.session_state.get('total_days', None))
        # 디버깅: day가 1로 몰려 있는 항목이 있는지 체크
        bad = [i for i in st.session_state.itinerary if int(i.get('day',1)) < 1]
        if bad:
            print("NORMALIZE WARNING: day 값 비정상 항목 있음:", bad)
    except Exception as e:
        print("DEBUG: 세션 itinerary 정규화 실패:", e)

    if response.get('current_weather'):
        st.session_state.current_weather = response['current_weather']
    
    if response.get('show_pdf_button'):
        st.session_state.show_pdf_button = True

    final_message = st.session_state.messages[-1] if st.session_state.messages else None
    if isinstance(final_message, AIMessage) and final_message.content:
        update_state_from_message(final_message.content)

# 이전 대화 기록 UI 출력
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        display_text = re.sub(r"\[(STATE_UPDATE|PLAN_ADD):.*?\]", "", msg.content, flags=re.DOTALL).strip()
        if display_text:
            st.chat_message("assistant").markdown(display_text)

# PDF 다운로드 버튼 표시 로직
if st.session_state.get("show_pdf_button", False):
    final_routes_text = ""
    for msg in reversed(st.session_state.messages):
        if isinstance(msg, AIMessage) and "최적 경로 제안" in msg.content:
            final_routes_text = re.sub(r"\[(STATE_UPDATE|PLAN_ADD):.*?\]", "", msg.content, flags=re.DOTALL).strip()
            break 
    
    if not final_routes_text:
        final_routes_text = "최적 경로가 아직 계산되지 않았습니다."

    # ✨ [새로 추가] PDF 생성 전 데이터 검증
    st.write("### 🔍 PDF 생성 데이터 검증")
    
    with st.expander("📊 데이터 상세 확인 (클릭하여 펼치기)", expanded=False):
        itinerary_data = st.session_state.itinerary
        st.write(f"**총 항목 수:** {len(itinerary_data)}")
        
        # Day별 분류
        day_groups = {}
        for idx, item in enumerate(itinerary_data):
            day = int(item.get('day', 1))
            if day not in day_groups:
                day_groups[day] = []
            day_groups[day].append((idx, item))
        
        # Day별 세부 정보
        for day in sorted(day_groups.keys()):
            items = day_groups[day]
            st.write(f"**📅 Day {day}:** {len(items)}개 항목")
            
            for idx, item in items:
                start = item.get('start', '없음')
                end = item.get('end', '없음')
                item_type = item.get('type', '미지정')
                name = item.get('name', '이름없음')
                
                time_valid = "✅" if (start != '없음' and end != '없음') else "⚠️"
                st.write(f"  {idx}. [{item_type}] {name} {time_valid}")
                st.write(f"     └ {start} ~ {end}")
                
                # Description
                desc = item.get('description', '')
                st.write(f"     └ 설명: {desc[:50]}..." if len(desc) > 50 else f"     └ 설명: {desc or '(없음)'}")
                
                # Reviews
                reviews = item.get('reviews', [])
                if reviews:
                    st.write(f"     └ 리뷰 ({len(reviews)}개): {reviews[0][:50]}...")
                else:
                    st.write(f"     └ 리뷰: (없음)")
        
        # 메타데이터
        st.write(f"\n**메타데이터:**")
        st.write(f"- 목적지: {st.session_state.destination}")
        st.write(f"- 날짜: {st.session_state.dates}")
        st.write(f"- 총 일수: {st.session_state.total_days}")

    pdf_bytes = create_itinerary_pdf(
        itinerary=st.session_state.itinerary,
        destination=st.session_state.destination,
        dates=st.session_state.dates,
        weather=st.session_state.current_weather,
        final_routes=final_routes_text,
        total_days=st.session_state.total_days
    )
    
    if pdf_bytes:
        st.success("✅ PDF 생성 완료!")
        st.download_button(
            label="📄 여행 계획 PDF 다운로드",
            data=pdf_bytes,
            file_name=f"{st.session_state.destination}_여행계획_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
    else:
        st.error("❌ PDF 파일 생성 실패")

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