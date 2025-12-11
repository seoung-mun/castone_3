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
    print("DEBUG: 1_trip_planner 페이지에서 DB 로드 확인 완료")
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

# --- helper: normalize itinerary for PDF output ---
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
        if 'type' not in it and 'category' in it: it['type'] = it.get('category')
        if 'name' not in it: it['name'] = it.get('장소명', it.get('name', '이름 없음'))
        # ✨ [새로 추가] reviews 필드 기본값 설정
        if 'reviews' not in it: it['reviews'] = []
        norm.append(it)
    return norm

# --- 2. PDF 생성 함수 ---
def create_itinerary_pdf(itinerary, destination, dates, weather, final_routes, total_days, start_location=None):
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

    # 출발지
    if start_location:
        pdf.set_font_size(11)
        pdf.cell(0, 8, text=f"출발지/숙소: {start_location}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    # 날씨
    if weather and weather.strip() and weather != '정보 없음':
        pdf.set_font_size(10)
        pdf.multi_cell(0, 5, text=f"날씨: {weather}", align='C')

    pdf.ln(10)

    # Normalize items first to ensure 'day' is int for sorting
    normalized_itinerary = _normalize_itinerary_for_pdf(itinerary, total_days)

    # 일정 정렬 (원본 순서 유지하면서 day와 인덱스로 정렬)
    try:
        sorted_itinerary = sorted(enumerate(normalized_itinerary), key=lambda x: (int(x[1].get('day', 1)), x[1].get('start', '00:00') or '00:00', x[0]))
        sorted_itinerary = [item[1] for item in sorted_itinerary]  # 인덱스 제거
    except Exception:
        # As a safe fallback, use normalized list directly
        sorted_itinerary = normalized_itinerary

    # 일자별 출력
    for day_num in range(1, total_days + 1):
        # 2일차부터는 여유 공간 추가 (페이지는 자동으로 넘어감)
        if day_num > 1:
            pdf.ln(15)  # 일차 사이 여유 공간

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

                # ✨ [새로 추가] 리뷰 섹션
                reviews = item.get('reviews', [])
                if reviews and isinstance(reviews, list):
                    if has_korean_font: pdf.set_font('NanumGothic', '', 9)
                    pdf.set_x(20)
                    for review in reviews:
                        # 리뷰 항목이 문자열이라면 그대로, dict라면 포매팅
                        if isinstance(review, str):
                            review_text = review
                        elif isinstance(review, dict):
                            review_text = review.get('text', str(review))
                        else:
                            review_text = str(review)
                        pdf.multi_cell(0, 4, text=f"  • {review_text}")
                    pdf.ln(2)

        # 일차별 구분선과 메모 공간
        pdf.ln(10)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(5)
        pdf.set_font_size(14)
        if has_korean_font: pdf.set_font('NanumGothic', '', 14)
        pdf.cell(0, 10, text="메모:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(20)  # 메모 공간

    return bytes(pdf.output())

# --- 3. 페이지 설정 및 세션 초기화 ---
st.set_page_config(page_title="AI 여행 플래너", layout="centered")
st.title("💬 AI 여행 플래너")

with st.sidebar:
    st.markdown("---")
    st.header("🛠️ 개발자 테스트 도구")
    
    # 체크박스를 켰을 때만 테스트 버튼이 보이게 설정
    if st.checkbox("PDF 생성 테스트 모드 켜기"):
        
        # 1. 테스트용 가짜 데이터 생성 (실제 JSON 구조에 맞춤)
        if st.button("🧪 테스트 데이터 로드 및 PDF 생성"):
            
            # (1) 메타데이터
            mock_destination = "서울 성수동"
            mock_dates = "2025-12-16 ~ 2025-12-19 (4일간)"
            mock_total_days = 4
            mock_weather = "맑음, 기온 12~18도 / 통풍이 잘 되는 옷 추천"
            mock_routes = "[최적 경로 요약]\n성수동 일대에서 각 명소들을 효율적으로 연결한 경로입니다. 대중교통(버스, 지하철) 이용으로 이동 시간을 최소화했습니다."
            
            # (2) 실제 JSON 구조: day(정수), type, name, description, start, end
            mock_itinerary = [
                # ===== Day 1 =====
                {
                    "day": 1,
                    "type": "식당",
                    "name": "카멜 성수점",
                    "description": "신선한 재료로 만든 정통 한식 요리를 즐길 수 있는 식당입니다.",
                    "start": "10:00",
                    "end": "11:30"
                },
                {
                    "day": 1,
                    "type": "move",
                    "transport": "버스 2412",
                    "duration_text": "약 22분",
                    "start": "11:30",
                    "end": "11:52"
                },
                {
                    "day": 1,
                    "type": "카페",
                    "name": "성수동대림창고갤러리",
                    "description": "예술 감성과 함께 편안함을 느낄 수 있는 갤러리 카페입니다.",
                    "start": "11:52",
                    "end": "13:22"
                },
                {
                    "day": 1,
                    "type": "move",
                    "transport": "버스 성동13",
                    "duration_text": "약 34분",
                    "start": "13:22",
                    "end": "13:56"
                },
                {
                    "day": 1,
                    "type": "관광지",
                    "name": "서울숲",
                    "description": "도시 속 자연을 만끽할 수 있는 광활한 공원으로 산책하기 좋습니다.",
                    "start": "13:56",
                    "end": "15:26"
                },
                {
                    "day": 1,
                    "type": "move",
                    "transport": "버스 6013",
                    "duration_text": "약 20분",
                    "start": "15:26",
                    "end": "15:47"
                },
                {
                    "day": 1,
                    "type": "식당",
                    "name": "글로우 성수",
                    "description": "세련된 분위기에서 건강식 메뉴를 제공하는 레스토랑입니다.",
                    "start": "15:47",
                    "end": "17:17"
                },

                # ===== Day 2 =====
                {
                    "day": 2,
                    "type": "관광지",
                    "name": "서울숲 가족마당",
                    "description": "가족 단위로 즐길 수 있는 넓은 잔디 공간과 포토존이 있습니다.",
                    "start": "10:00",
                    "end": "11:30"
                },
                {
                    "day": 2,
                    "type": "move",
                    "transport": "지하철 2호선",
                    "duration_text": "약 22분",
                    "start": "11:30",
                    "end": "11:52"
                },
                {
                    "day": 2,
                    "type": "식당",
                    "name": "아키야마 성수본점",
                    "description": "프리미엄 돈까스와 다양한 일식 요리로 유명한 고급 음식점입니다.",
                    "start": "11:52",
                    "end": "13:22"
                },
                {
                    "day": 2,
                    "type": "move",
                    "transport": "지하철 2호선 + 버스 270",
                    "duration_text": "약 38분",
                    "start": "13:22",
                    "end": "13:32"  # 실제로는 14:00 정도지만 표기 간소화
                },
                {
                    "day": 2,
                    "type": "카페",
                    "name": "바이닐 성수",
                    "description": "레트로한 감성과 아늑한 분위기가 매력적인 독립 카페입니다.",
                    "start": "14:00",
                    "end": "15:30"
                },
                {
                    "day": 2,
                    "type": "move",
                    "transport": "버스 심야A21",
                    "duration_text": "약 14분",
                    "start": "15:30",
                    "end": "15:45"
                },
                {
                    "day": 2,
                    "type": "관광지",
                    "name": "홍대선원",
                    "description": "홍대의 문화와 예술을 체험할 수 있는 갤러리와 전시 공간입니다.",
                    "start": "15:45",
                    "end": "17:15"
                },
                {
                    "day": 2,
                    "type": "move",
                    "transport": "버스",
                    "duration_text": "약 41분",
                    "start": "17:15",
                    "end": "17:56"
                },
                {
                    "day": 2,
                    "type": "식당",
                    "name": "사조미가",
                    "description": "신선한 회와 정통 일식 코스를 제공하는 프리미엄 식당입니다.",
                    "start": "18:00",
                    "end": "19:30"
                },

                # ===== Day 3 =====
                {
                    "day": 3,
                    "type": "카페",
                    "name": "앤트러사이트 연희점",
                    "description": "감성적인 인테리어와 정성스러운 음료로 유명한 브런치 카페입니다.",
                    "start": "10:00",
                    "end": "11:30"
                },
                {
                    "day": 3,
                    "type": "move",
                    "transport": "버스 N62 + 버스 6010",
                    "duration_text": "약 70분",
                    "start": "11:30",
                    "end": "12:40"
                },
                {
                    "day": 3,
                    "type": "식당",
                    "name": "은성보쌈",
                    "description": "풍미 있는 보쌈과 다양한 반찬으로 알려진 전통 한식당입니다.",
                    "start": "12:40",
                    "end": "14:10"
                },
                {
                    "day": 3,
                    "type": "move",
                    "transport": "지하철 3호선",
                    "duration_text": "약 17분",
                    "start": "14:10",
                    "end": "14:27"
                },
                {
                    "day": 3,
                    "type": "카페",
                    "name": "호텔수선화",
                    "description": "우아한 분위기와 프리미엄 디저트로 오후 시간을 즐길 수 있습니다.",
                    "start": "14:27",
                    "end": "15:57"
                },
                {
                    "day": 3,
                    "type": "move",
                    "transport": "버스 261",
                    "duration_text": "약 22분",
                    "start": "15:57",
                    "end": "16:19"
                },
                {
                    "day": 3,
                    "type": "관광지",
                    "name": "서울로7017",
                    "description": "옛 고가도로를 공원으로 재탄생시킨 핫플레이스로 야경이 아름답습니다.",
                    "start": "16:19",
                    "end": "17:49"
                },
                {
                    "day": 3,
                    "type": "move",
                    "transport": "버스 463",
                    "duration_text": "약 33분",
                    "start": "17:49",
                    "end": "18:22"
                },
                {
                    "day": 3,
                    "type": "식당",
                    "name": "유래회관",
                    "description": "신선한 회와 다양한 해산물 요리로 저녁을 우아하게 마무리할 수 있습니다.",
                    "start": "18:22",
                    "end": "19:52"
                },

                # ===== Day 4 =====
                {
                    "day": 4,
                    "type": "관광지",
                    "name": "성수동구두테마공원",
                    "description": "서울의 신발 산업 역사를 배우고 다양한 구두와 패션 제품을 볼 수 있는 공간입니다.",
                    "start": "10:00",
                    "end": "12:00"
                },
            ]
            
            # (3) 세션 상태 강제 업데이트
            st.session_state.destination = mock_destination
            st.session_state.dates = mock_dates
            st.session_state.itinerary = mock_itinerary
            st.session_state.total_days = mock_total_days
            st.session_state.current_weather = mock_weather

            # (4) 정규화 후 PDF 생성
            normalized_mock = _normalize_itinerary_for_pdf(mock_itinerary, mock_total_days)
            
            try:
                pdf_data = create_itinerary_pdf(
                    itinerary=normalized_mock,
                    destination=mock_destination,
                    dates=mock_dates,
                    weather=mock_weather,
                    final_routes=mock_routes,
                    total_days=mock_total_days
                )
                
                # (5) 다운로드 버튼 생성
                if pdf_data:
                    st.success("✅ 테스트 PDF 생성 완료!")
                    st.download_button(
                        label="📥 테스트 PDF 다운로드",
                        data=pdf_data,
                        file_name=f"test_itinerary_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error("❌ PDF 생성 실패 (데이터 없음)")
            except Exception as e:
                st.error(f"❌ 에러 발생: {e}")
                st.write(f"상세: {str(e)}")
    
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

# 필수 정보 체크
if "preferences_collected" not in st.session_state:
    st.warning("⚠️ 정보 입력 페이지에서 먼저 여행 정보를 입력해주세요.")
    # 로컬 테스트용 임시 버튼 (실제 배포시 제거 가능)
    if st.button("테스트용 임시 데이터 로드"):
        st.session_state.destination = "부산 해운대"
        st.session_state.dates = "2025-12-06 (1일)"
        st.session_state.total_days = 1
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
        "current_weather": st.session_state.get('current_weather', ''),
        "show_pdf_button": st.session_state.get('show_pdf_button', False),
        "current_anchor": st.session_state.get('current_anchor', st.session_state.get('destination', '')),
        "dialog_stage": st.session_state.get("dialog_stage", "planning")
    }
    
    with st.spinner("AI가 여행 계획을 생성/수정 중입니다..."):
        response = await APP.ainvoke(current_state, config=config)

    st.session_state.messages = response.get('messages', [])
    st.session_state.itinerary = response.get('itinerary', [])
    # 강제 정규화: day 정수형 및 기본 키 보장
    try:
        st.session_state.itinerary = _normalize_itinerary_for_pdf(st.session_state.itinerary, st.session_state.get('total_days', None))
    except Exception as e:
        print("DEBUG: 페이지측 itinerary 정규화 실패:", e)

    st.session_state.current_weather = response.get('current_weather', '')
    st.session_state.show_pdf_button = response.get('show_pdf_button', False)
    st.session_state.current_anchor = response.get('current_anchor', '')
    
    if 'dialog_stage' in response:
        st.session_state.dialog_stage = response['dialog_stage']

# --- 6. 초기 실행 트리거 ---
if not st.session_state.messages:
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(time.time())

    # 출발지 정보가 있으면 포함
    start_location_text = ""
    if st.session_state.get('start_location'):
        start_location_text = f"\n    - 출발지/숙소: {st.session_state.get('start_location')}"

    initial_prompt = f"""
    안녕하세요! 아래 정보로 여행 계획을 세워주세요.
    - 목적지: {st.session_state.get('destination')}{start_location_text}
    - 일정: {st.session_state.get('dates')} (총 {st.session_state.get('total_days')}일)
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
            clean_content = re.sub(r"\[(ADD|REPLACE|DELETE)_PLACE\].*?\[/\1_PLACE\]", "", content_str, flags=re.DOTALL)
            if "FINISH" in clean_content and len(clean_content) < 10:
                continue
            if clean_content.strip():
                st.chat_message("assistant").markdown(clean_content)

# --- 8. PDF 다운로드 버튼 ---
if st.session_state.show_pdf_button:
    weather_info = st.session_state.get('current_weather', '날씨 정보 없음')
    
    # ✨ [새로 추가] PDF 생성 전 데이터 검증
    st.write("### 🔍 PDF 생성 데이터 검증")
    
    with st.expander("📊 데이터 상세 확인 (클릭하여 펼치기)", expanded=False):
        # 1. Itinerary 구조 검증
        st.write("#### 1️⃣ Itinerary 구조 검증")
        
        itinerary_data = st.session_state.itinerary
        st.write(f"**총 항목 수:** {len(itinerary_data)}")
        
        # Day별 분류
        day_groups = {}
        for idx, item in enumerate(itinerary_data):
            day = int(item.get('day', 1))
            if day not in day_groups:
                day_groups[day] = []
            day_groups[day].append((idx, item))
        
        for day in sorted(day_groups.keys()):
            items = day_groups[day]
            st.write(f"**Day {day}:** {len(items)}개 항목")
            
            for idx, item in items:
                # 시간 정보 검증
                start = item.get('start', '없음')
                end = item.get('end', '없음')
                item_type = item.get('type', '미지정')
                name = item.get('name', '이름없음')
                
                # 시간 유효성 검사
                time_valid = "✅" if (start != '없음' and end != '없음' and start < end) else "❌"
                
                st.write(f"  {idx}. [{item_type}] {name} {time_valid}")
                st.write(f"     └ 시간: {start} ~ {end}")
                
                # Description 확인
                description = item.get('description', '')
                if description:
                    st.write(f"     └ 설명: {description[:60]}..." if len(description) > 60 else f"     └ 설명: {description}")
                else:
                    st.write(f"     └ 설명: (없음)")
                
                # Reviews 확인
                reviews = item.get('reviews', [])
                if reviews:
                    st.write(f"     └ 리뷰 ({len(reviews)}개):")
                    for rev in reviews:
                        st.write(f"        • {rev[:70]}..." if len(rev) > 70 else f"        • {rev}")
                else:
                    st.write(f"     └ 리뷰: (없음)")
        
        # 2. 메타데이터 검증
        st.write("#### 2️⃣ 메타데이터 검증")
        st.write(f"**목적지:** {st.session_state.destination}")
        st.write(f"**날짜:** {st.session_state.dates}")
        st.write(f"**총 일수:** {st.session_state.total_days}")
        st.write(f"**날씨:** {weather_info[:100]}..." if len(weather_info) > 100 else f"**날씨:** {weather_info}")
        
        # 3. 시간 순서 검증
        st.write("#### 3️⃣ 시간 순서 검증 (각 Day별)")
        
        for day in sorted(day_groups.keys()):
            items = day_groups[day]
            # 활동만 필터 (move 제외 또는 포함)
            activity_items = [item for _, item in items if item.get('type') != 'move']
            
            if activity_items:
                times = [item.get('start', '00:00') for item in activity_items]
                is_sorted = all(times[i] <= times[i+1] for i in range(len(times)-1))
                status = "✅ 정렬됨" if is_sorted else "❌ 정렬 안 됨"
                
                st.write(f"**Day {day}:** {status}")
                for item in activity_items:
                    st.write(f"  - {item.get('start', '?')} ~ {item.get('end', '?')}: {item.get('name', '?')}")
        
        # 4. 정규화 후 상태 확인
        st.write("#### 4️⃣ 정규화 후 상태")
        normalized = _normalize_itinerary_for_pdf(itinerary_data, st.session_state.total_days)
        st.write(f"**정규화 후 항목 수:** {len(normalized)}")
        
        # 모든 day가 정수인지 확인
        all_days_int = all(isinstance(item.get('day'), int) for item in normalized)
        st.write(f"**모든 day가 정수:** {'✅ Yes' if all_days_int else '❌ No'}")
        
        # 모든 필수 필드 확인
        missing_fields = []
        for idx, item in enumerate(normalized):
            if not item.get('name'):
                missing_fields.append(f"항목{idx}: name 없음")
            if not item.get('type'):
                missing_fields.append(f"항목{idx}: type 없음")
            if 'day' not in item:
                missing_fields.append(f"항목{idx}: day 없음")
            if 'reviews' not in item:
                missing_fields.append(f"항목{idx}: reviews 없음")
        
        if missing_fields:
            st.write(f"**필드 누락:** ❌")
            for field in missing_fields:
                st.write(f"  - {field}")
        else:
            st.write(f"**필드 누락:** ✅ None")
    
    # PDF 생성
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
        st.error("❌ PDF 생성 실패")

# --- 9. 사용자 입력 처리 ---
if user_input := st.chat_input("메시지를 입력하세요..."):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").markdown(user_input)
    
    # [수정] 공유된 이벤트 루프 사용
    loop.run_until_complete(run_ai_agent())
    st.rerun()