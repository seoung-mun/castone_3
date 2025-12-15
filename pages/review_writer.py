import streamlit as st
import pandas as pd
import os
from src.rag_updater import update_vector_db_if_needed # 👈 3단계에서 만든 업데이터 임포트

st.set_page_config(page_title="리뷰 작성기", page_icon="✍️")
st.title("✍️ 여행지 리뷰 작성")
st.caption("여러분의 리뷰가 10개 이상 쌓이면 AI 에이전트의 지식에 반영됩니다.")

# 임시 리뷰 저장 파일
NEW_REVIEWS_FILE = "new_reviews.csv"

# 1. 리뷰 입력을 위한 UI (data_clean.ipynb 컬럼 참고)
with st.form("review_form", clear_on_submit=True):
    # data_clean.ipynb의 '장소', '지역', '카테고리_통합', '평점', '리뷰' 컬럼
    place_name = st.text_input("장소 이름", placeholder="예: 경복궁, 해운대 해수욕장")
    region = st.text_input("지역", placeholder="예: 서울 종로구, 부산 해운대구")
    category = st.text_input("카테고리", placeholder="예: 관광지, 식당 한식")
    rating = st.slider("평점 (1-5)", 1, 5, 3)
    review_text = st.text_area("리뷰 내용", placeholder="방문 경험을 자세히 적어주세요...")
    
    submitted = st.form_submit_button("리뷰 제출하기")

# 2. 폼 제출 시 로직
if submitted:
    if not all([place_name, region, category, review_text]):
        st.error("모든 항목을 입력해주세요.")
    else:
        # 2-1. 새 리뷰 데이터 생성 (data_clean.ipynb/data_process.ipynb 컬럼명과 일치)
        new_review_data = {
            "지역": [region],
            "장소명": [place_name],
            "카테고리_통합": [category],
            "리뷰": [review_text],
            "평점": [rating]
        }
        new_df = pd.DataFrame(new_review_data)
        
        # 2-2. 기존 new_reviews.csv에 추가 (없으면 새로 생성)
        try:
            if not os.path.exists(NEW_REVIEWS_FILE):
                new_df.to_csv(NEW_REVIEWS_FILE, index=False, encoding="utf-8-sig")
            else:
                new_df.to_csv(NEW_REVIEWS_FILE, mode='a', header=False, index=False, encoding="utf-8-sig")
                
            st.success(f"'{place_name}' 리뷰가 성공적으로 저장되었습니다. 감사합니다!")
            
            # 2-3. (핵심) DB 업데이트 트리거
            update_status = update_vector_db_if_needed(NEW_REVIEWS_FILE)
            print(update_status) # 콘솔에 상태 출력

        except Exception as e:
            st.error(f"리뷰 저장 중 오류 발생: {e}")

# 3. 현재 누적된 리뷰 수 표시
try:
    if os.path.exists(NEW_REVIEWS_FILE):
        df = pd.read_csv(NEW_REVIEWS_FILE)
        st.info(f"현재 누적된 리뷰: {len(df)}개 (10개 이상 시 DB에 자동 반영)")
    else:
        st.info("현재 누적된 리뷰: 0개 (10개 이상 시 DB에 자동 반영)")
except pd.errors.EmptyDataError:
    st.info("현재 누적된 리뷰: 0개 (10개 이상 시 DB에 자동 반영)")