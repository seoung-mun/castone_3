
import os
import streamlit as st 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import googlemaps

load_faiss_index_start_time = None 

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


@st.cache_resource(show_spinner=False)
def load_faiss_index():
    """FAISS 인덱스를 로드합니다."""
    print(" 무거운 라이브러리(Langchain, FAISS) 로딩 시작 (함수 내부)...")
    
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    
    print("임베딩 모델 및 FAISS 인덱스 로딩 중...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="upskyy/bge-m3-korean",
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    try:
        DB = FAISS.load_local(review_faiss, embeddings, allow_dangerous_deserialization=True)
        print("Vector DB(Faiss) 로딩 완료!")
        return DB
    except Exception as e:
        print(f"FAISS 로드 실패: {e}")
        return None

import os
import streamlit as st 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import googlemaps

load_faiss_index_start_time = None 

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


@st.cache_resource(show_spinner=False)
def load_faiss_index():
    """FAISS 인덱스를 로드합니다."""
    print(" 무거운 라이브러리(Langchain, FAISS) 로딩 시작 (함수 내부)...")
    
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    
    print("임베딩 모델 및 FAISS 인덱스 로딩 중...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="upskyy/bge-m3-korean",
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': False}
    )
    
    try:
        DB = FAISS.load_local(review_faiss, embeddings, allow_dangerous_deserialization=True)
        print("Vector DB(Faiss) 로딩 완료!")
        return DB
    except Exception as e:
        print(f"FAISS 로드 실패: {e}")
        return None