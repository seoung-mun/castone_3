# convert_faiss_format.py

import os
import pickle
import faiss

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS as LangchainFAISS
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW_FAISS_DIR = "/Users/seoungmun/Documents/work/3-2/project/travle_agent4/review-faiss"  # 필요하면 경로 수정

def load_metadata_list(path: str):
    with open(path, "rb") as f:
        metadata_list = pickle.load(f)
    return metadata_list

def ensure_documents(metadata_list):
    """metadata_list를 LangChain Document 리스트로 맞춰주기."""
    docs = []
    for item in metadata_list:
        if isinstance(item, Document):
            # 이미 Document면 그대로 사용
            docs.append(item)

        elif isinstance(item, dict):
            # 1) page_content로 쓸 텍스트 결정
            text = (
                item.get("page_content")          # 혹시 이미 있다면 우선
                or item.get("text_for_embedding") # 네 데이터에 있는 필드
                or item.get("리뷰")               # 그래도 없으면 리뷰만이라도
                or ""
            )

            # 2) 나머지는 metadata로 사용 (page_content용 필드는 빼줌)
            metadata = dict(item)
            for k in ["page_content", "text_for_embedding"]:
                metadata.pop(k, None)

            docs.append(Document(page_content=text, metadata=metadata))

        else:
            # dict도 아니고 Document도 아니면 그냥 문자열로 캐스팅
            docs.append(Document(page_content=str(item), metadata={}))

    return docs
def main():
    # 1) faiss.index / metadata_list.pkl 경로
    faiss_index_path = os.path.join(REVIEW_FAISS_DIR, "faiss.index")
    metadata_path = os.path.join(REVIEW_FAISS_DIR, "metadata_list.pkl")

    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"faiss.index not found: {faiss_index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata_list.pkl not found: {metadata_path}")

    print("▶ Loading FAISS index...")
    index = faiss.read_index(faiss_index_path)

    print("▶ Loading metadata_list.pkl...")
    metadata_list = load_metadata_list(metadata_path)
    docs = ensure_documents(metadata_list)

    print(f"▶ Documents loaded: {len(docs)}")

    # 2) Docstore / id 매핑 구성
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}

    # 3) embedding_function은 굳이 로드 안 해도 됨 (나중에 load_local에서 다시 넣을 거라)
    faiss_store = LangchainFAISS(
        embedding_function=None,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    # 4) LangChain 표준 포맷으로 저장
    print("▶ Saving as LangChain FAISS format (index.faiss, index.pkl)...")
    faiss_store.save_local(REVIEW_FAISS_DIR)
    print("✅ Done!  Now you can use FAISS.load_local(review_faiss, embeddings, ...)")

if __name__ == "__main__":
    main()
