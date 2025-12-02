# src/check_metadata.py

import os
import pickle

REVIEW_DIR = "/Users/seoungmun/Documents/work/3-2/project/travle_agent4/review_faiss"
META_PATH = os.path.join(REVIEW_DIR, "metadata_list.pkl")

if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"metadata_list.pkl not found: {META_PATH}")

with open(META_PATH, "rb") as f:
    metadata_list = pickle.load(f)

print("✅ metadata_list 전체 타입:", type(metadata_list))
print("✅ metadata_list 전체 개수:", len(metadata_list))
print("✅ 첫 번째 항목 타입:", type(metadata_list[0]))
print("\n✅ 첫 번째 항목 전체 내용 ↓↓↓\n")
print(metadata_list[0])
