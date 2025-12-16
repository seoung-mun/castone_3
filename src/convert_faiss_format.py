
import os
import pickle
import faiss

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS as LangchainFAISS
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW_FAISS_DIR = "/Users/seoungmun/Documents/work/3-2/project/travle_agent4/review_faiss"  
def load_metadata_list(path: str):
    with open(path, "rb") as f:
        metadata_list = pickle.load(f)
    return metadata_list

def ensure_documents(metadata_list):
    """metadata_list를 LangChain Document 리스트로 맞춰주기."""
    docs = []
    for item in metadata_list:
        if isinstance(item, Document):
            docs.append(item)

        elif isinstance(item, dict):
            text = (
                item.get("page_content")          
                or item.get("text_for_embedding") 
                or item.get("리뷰")               
                or ""
            )

            metadata = dict(item)
            for k in ["page_content", "text_for_embedding"]:
                metadata.pop(k, None)

            docs.append(Document(page_content=text, metadata=metadata))

        else:
            docs.append(Document(page_content=str(item), metadata={}))

    return docs
def main():
    faiss_index_path = os.path.join(REVIEW_FAISS_DIR, "faiss.index")
    metadata_path = os.path.join(REVIEW_FAISS_DIR, "metadata_list.pkl")

    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"faiss.index not found: {faiss_index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata_list.pkl not found: {metadata_path}")

    print(" Loading FAISS index...")
    index = faiss.read_index(faiss_index_path)

    print("Loading metadata_list.pkl...")
    metadata_list = load_metadata_list(metadata_path)
    docs = ensure_documents(metadata_list)

    print(f" Documents loaded: {len(docs)}")

    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}

    faiss_store = LangchainFAISS(
        embedding_function=None,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    print("Saving as LangChain FAISS format (index.faiss, index.pkl)...")
    faiss_store.save_local(REVIEW_FAISS_DIR)
    print(" Done!  Now you can use FAISS.load_local(review_faiss, embeddings, ...)")

if __name__ == "__main__":
    main()
