
import pandas as pd
import re
import emoji
import streamlit as st
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from src.config import review_faiss 

# --- 1. 전처리 함수 ---
def clean_review(text):
    text = str(text) 
    text = re.sub(r'\s+', ' ', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    text = text.strip()
    return text

def chunk_text_with_overlap(text, chunk_size=500, overlap=50):
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0: start = 0
        if start >= len(text): break
    return chunks

# --- 2. [신규] 기존 DB에서 주소 찾기 헬퍼 ---
def find_address_from_db(db, place_name):
    """
    기존 FAISS DB에서 장소명으로 검색하여 '상세 주소'를 가져옵니다.
    """
    if not db: return ""
    
    try:
        # 장소명으로 유사도 검색 (상위 1개만)
        results = db.similarity_search(place_name, k=1)
        if results:
            doc = results[0]
   
            existing_name = doc.metadata.get("장소명", "")
            
            if place_name in existing_name or existing_name in place_name:
                address = doc.metadata.get("상세 주소", "")
                if address:
                    print(f"   [Smart Fill] '{place_name}'의 주소를 DB에서 찾았습니다: {address}")
                    return address
    except Exception as e:
        print(f"DEBUG: 주소 검색 중 오류: {e}")
    
    return ""

# --- 3. 문서화 함수 (수정됨: DB 주소 조회 로직 추가) ---
def create_documents_from_df(df, existing_db=None):
    """
    DataFrame -> Document 변환
    * existing_db: 주소 조회를 위해 전달받은 기존 FAISS DB 객체
    """
    docs = []
    for _, row in df.iterrows():
        cleaned_review = clean_review(row.get("리뷰", "")) 
        chunks = chunk_text_with_overlap(cleaned_review, chunk_size=500, overlap=20)
        
        # 컬럼 매핑
        place_name = row.get("장소명") if pd.notna(row.get("장소명")) else row.get("장소", "장소미상")
        category = row.get("카테고리_통합") if pd.notna(row.get("카테고리_통합")) else row.get("카테고리", "기타")
        rating = row.get("평점") if pd.notna(row.get("평점")) else row.get("별점", "0")
        
    
        address = row.get("상세 주소") if pd.notna(row.get("상세 주소")) else ""
        
        if not address and existing_db:
            address = find_address_from_db(existing_db, place_name)

        for chunk in chunks:
            if len(chunk) <= 5: continue

            combined_text = (
                f"지역: {row.get('지역', '')} | "
                f"장소명: {place_name} | "
                f"카테고리: {category} | "
                f"리뷰: {chunk}"
            )
            
            doc = Document(
                page_content=combined_text,
                metadata={
                    "지역": str(row.get("지역", "")),
                    "카테고리": str(category),
                    "장소명": str(place_name),
                    "별점": str(rating),
                    "상세 주소": str(address),  # 찾아낸 주소가 들어감
                    "리뷰": str(row.get("리뷰", "")[:100])
                }
            )
            docs.append(doc)
    return docs

# --- 4. 벡터 DB 업데이트 함수 (수정됨: DB 먼저 로드) ---
def update_vector_db_if_needed(new_reviews_file="new_reviews.csv"):
    try:
        df = pd.read_csv(new_reviews_file)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return "업데이트할 리뷰가 없습니다."

    if len(df) < 10:
        return f"리뷰 {len(df)}개 누적됨. (10개 이상이어야 업데이트)"

    st.toast(f"리뷰 {len(df)}개 DB 업데이트 시작...")
    print(f"--- [RAG Updater] 리뷰 {len(df)}개 DB 업데이트 시작 ---")

    try:
        # 1. 임베딩 모델 로드
        embeddings = HuggingFaceEmbeddings(
            model_name="upskyy/bge-m3-korean",
            model_kwargs={"device": "cpu"}
        )
        
        # 2. [순서 변경] 기존 DB를 먼저 로드 (검색용)
        existing_db = None
        if os.path.exists(review_faiss):
            try:
                existing_db = FAISS.load_local(
                    review_faiss, embeddings, allow_dangerous_deserialization=True
                )
                print("[RAG Updater] 기존 DB 로드 완료 (주소 검색용)")
            except Exception as e:
                print(f"[RAG Updater] 기존 DB 로드 실패: {e}")

        # 3. 문서 생성 (여기서 existing_db를 넘겨줘서 주소를 찾게 함)
        new_docs = create_documents_from_df(df, existing_db=existing_db)
        
        if not new_docs:
            os.remove(new_reviews_file) 
            return "유효한 문서 없음"

        print(f"[RAG Updater] {len(new_docs)}개의 새 문서 생성 완료")

        # 4. DB에 추가 (existing_db가 있으면 거기에 추가, 없으면 새로 생성)
        if existing_db:
            existing_db.add_documents(new_docs)
            db_to_save = existing_db
        else:
            print("[RAG Updater] 기존 DB가 없어 새로 생성합니다.")
            db_to_save = FAISS.from_documents(new_docs, embeddings)

        # 5. 저장 및 정리
        db_to_save.save_local(review_faiss)
        st.cache_resource.clear()
        os.remove(new_reviews_file)
        
        print("[RAG Updater] 업데이트 완료 및 저장됨.")
        st.toast("벡터 DB 업데이트 완료!", icon="🎉")
        return "벡터 DB 업데이트 완료!"

    except Exception as e:
        print(f"DEBUG: Critical Error: {e}")
        return f"오류: {e}"