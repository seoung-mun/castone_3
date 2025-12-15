# src/config.py

import os
import streamlit as st 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import googlemaps

# --- 1. 환경 변수 및 기본 설정 로드 ---
load_faiss_index_start_time = None # 디버깅용 (선택사항)

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
# review_faiss 경로 설정 (상위 폴더 기준)
review_faiss = os.path.join(os.path.dirname(current_dir), "review_faiss") 

LLM = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.0)

GMAPS_API_KEY = os.getenv("GMAPS_API_KEY")
GMAPS_CLIENT = None
if GMAPS_API_KEY:
    GMAPS_CLIENT = googlemaps.Client(key=GMAPS_API_KEY)
else:
    print("경고: .env 파일에 GMAPS_API_KEY가 설정되지 않았습니다.")


# --- 2. RAG FAISS 인덱스 로드 함수 (Lazy Import 적용) ---
@st.cache_resource(show_spinner=False)
def load_faiss_index():
    """FAISS 인덱스를 로드합니다."""
    print("DEBUG: 🐢 무거운 라이브러리(Langchain, FAISS) 로딩 시작 (함수 내부)...")
    
    # 🚨 [핵심 수정] 무거운 임포트를 함수 안으로 이동!
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    
    print("DEBUG: 🚀 임베딩 모델 및 FAISS 인덱스 로딩 중...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="upskyy/bge-m3-korean",
        model_kwargs={'device': 'cpu'}, # GPU가 있다면 'cuda'
        encode_kwargs={'normalize_embeddings': True}
    )
    
    try:
        DB = FAISS.load_local(review_faiss, embeddings, allow_dangerous_deserialization=True)
        print("DEBUG: ✅ Vector DB(Faiss) 로딩 완료!")
        return DB
    except Exception as e:
        print(f"DEBUG: ❌ FAISS 로드 실패: {e}")
        return None# src/config.py

import os
import streamlit as st 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import googlemaps

# --- 1. 환경 변수 및 기본 설정 로드 ---
load_faiss_index_start_time = None # 디버깅용 (선택사항)

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
# review_faiss 경로 설정 (상위 폴더 기준)
review_faiss = os.path.join(os.path.dirname(current_dir), "review_faiss") 

LLM = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.0)

GMAPS_API_KEY = os.getenv("GMAPS_API_KEY")
GMAPS_CLIENT = None
if GMAPS_API_KEY:
    GMAPS_CLIENT = googlemaps.Client(key=GMAPS_API_KEY)
else:
    print("경고: .env 파일에 GMAPS_API_KEY가 설정되지 않았습니다.")


# --- 2. RAG FAISS 인덱스 로드 함수 (Lazy Import 적용) ---
@st.cache_resource(show_spinner=False)
def load_faiss_index():
    """FAISS 인덱스를 로드합니다."""
    print("DEBUG: 🐢 무거운 라이브러리(Langchain, FAISS) 로딩 시작 (함수 내부)...")
    
    # 🚨 [핵심 수정] 무거운 임포트를 함수 안으로 이동!
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    
    print("DEBUG: 🚀 임베딩 모델 및 FAISS 인덱스 로딩 중...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="upskyy/bge-m3-korean",
        model_kwargs={'device': 'cpu'}, # GPU가 있다면 'cuda'
        encode_kwargs={'normalize_embeddings': False}
    )
    
    try:
        DB = FAISS.load_local(review_faiss, embeddings, allow_dangerous_deserialization=True)
        print("DEBUG: ✅ Vector DB(Faiss) 로딩 완료!")
        return DB
    except Exception as e:
        print(f"DEBUG: ❌ FAISS 로드 실패: {e}")
        return None