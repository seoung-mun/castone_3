import os
import streamlit as st # 👈 [추가]
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import googlemaps

# --- 1. 환경 변수 및 기본 설정 로드 ---
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
review_faiss = os.path.join(os.path.dirname(current_dir), "review_faiss") 

LLM = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.0)

GMAPS_API_KEY = os.getenv("GMAPS_API_KEY")
GMAPS_CLIENT = None
if GMAPS_API_KEY:
    GMAPS_CLIENT = googlemaps.Client(key=GMAPS_API_KEY)
else:
    print("경고: .env 파일에 GMAPS_API_KEY가 설정되지 않았습니다.")


# --- 2. RAG FAISS 인덱스 로드 함수 ---

@st.cache_resource # 👈 [추가]
def load_faiss_index():
    """FAISS 인덱스를 로드합니다."""
    embeddings = HuggingFaceEmbeddings(
        model_name="upskyy/bge-m3-korean",
        model_kwargs={"device": "cpu"}
    )
    load_db = FAISS.load_local(
        review_faiss, embeddings, allow_dangerous_deserialization=True
    )
    return load_db