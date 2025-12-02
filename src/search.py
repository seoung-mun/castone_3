# search_utils.py
"""
FAISS 기반 리뷰 검색 + 메타데이터(지역) 필터링 유틸 모듈.

- FAISS similarity 검색
- 사용자 쿼리에서 허용 지역 추출 (parse_regions_from_query)
- 문서 메타데이터의 특정 필드(예: '광역시/도', '지역') 기준 필터링
"""

from typing import List, Tuple, Set

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


from langchain.schema import BaseRetriever
from langchain_core.documents import Document
from typing import List, Set, Optional

from src.region_cut_fuzz import (
    parse_regions_from_query,
    filter_docs_by_region,
)


def get_allowed_regions_from_query(
    query: str,
    *,
    fuzzy_threshold: int = 87,
) -> Set[str]:
    """
    사용자 자연어 쿼리에서 허용 지역(광역시/도 등)을 fuzzy 방식으로 추출.

    Args:
        query: 사용자 입력 문장.
        fuzzy_threshold: 지역명 매칭에 사용할 임계값 (높을수록 더 엄격).

    Returns:
        허용 지역 문자열들의 집합(set).
    """
    return parse_regions_from_query(
        query,
        fuzzy=True,
        fuzzy_threshold=fuzzy_threshold,
    )


def apply_region_filter(
    docs: List[Document],
    allowed_regions: Set[str],
    *,
    field: str = "광역시/도",
    drop_unknown: bool = True,
) -> List[Document]:
    """
    Document 리스트에 대해 메타데이터의 특정 필드 기준으로 지역 필터 적용.

    Args:
        docs: 검색된 Document 리스트.
        allowed_regions: 허용할 지역 문자열 집합.
        field: 메타데이터에서 지역이 저장된 필드 이름 (예: '광역시/도', '지역').
        drop_unknown: 메타데이터에 해당 필드가 없는 문서를 버릴지 여부.

    Returns:
        허용 지역에 해당하는 문서만 남긴 새로운 리스트.
    """
    if not allowed_regions:
        # 허용 지역 정보가 없으면 필터를 적용하지 않고 그대로 반환
        return docs

    return filter_docs_by_region(
        docs,
        allowed_regions,
        field=field,
        drop_unknown=drop_unknown,
    )


def search_with_region_filter(
    db: FAISS,
    query: str,
    *,
    k: int = 10,
    region_field: str = "광역시/도",
    fuzzy_threshold: int = 87,
) -> Tuple[List[Document], List[str]]:
    """
    1) FAISS similarity 검색으로 리뷰 데이터에서 상위 k개 문서를 찾고,
    2) 문서 메타데이터의 지역 필드(예: '광역시/도', '지역')로 한 번 더 필터링.

    Args:
        db: LangChain FAISS VectorStore 인스턴스.
        query: 사용자 자연어 쿼리.
        k: FAISS similarity 검색에서 가져올 문서 수.
        region_field: 메타데이터에 저장된 지역 필드 이름.
        fuzzy_threshold: 쿼리 → 허용 지역 추출 시 사용할 fuzzy 매칭 임계값.

    Returns:
        (필터링된 문서 리스트, 허용 지역 리스트)
    """
    # 1. 쿼리에서 허용 지역 파싱
    allowed_regions: Set[str] = get_allowed_regions_from_query(
        query,
        fuzzy_threshold=fuzzy_threshold,
    )

    # 2. 순수 similarity 기반 검색 (지역 제한 X)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    docs: List[Document] = retriever.get_relevant_documents(query)

    # 3. 메타데이터 기반 지역 필터
    docs_filtered = apply_region_filter(
        docs,
        allowed_regions,
        field=region_field,
        drop_unknown=True,
    )

    # 리스트 형태로도 돌려주기 (프롬프트에 넣을 때 사용)
    return docs_filtered, sorted(list(allowed_regions))


class RegionFilteringRetriever(BaseRetriever):
    """
    LangChain 호환 Retriever:
    - 입력: query(str)
    - 출력: List[Document]
    - 내부: FAISS 검색 → 메타데이터 지역 필터링
    """

    def __init__(self, db, k: int = 10, region_field: str = "광역시/도", fuzzy_threshold: int = 87):
        super().__init__()
        self.db = db
        self.k = k
        self.region_field = region_field
        self.fuzzy_threshold = fuzzy_threshold

    def _get_allowed_regions(self, query: str) -> Set[str]:
        return parse_regions_from_query(
            query, fuzzy=True, fuzzy_threshold=self.fuzzy_threshold
        )

    def _apply_region_filter(self, docs: List[Document], allowed: Set[str]) -> List[Document]:
        if not allowed:
            return docs
        return filter_docs_by_region(docs, allowed, field=self.region_field, drop_unknown=True)

    def get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # 1. 지역 파싱
        allowed = self._get_allowed_regions(query)

        # 2. FAISS 검색
        retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )
        docs = retriever.get_relevant_documents(query)

        # 3. 지역 필터링
        docs = self._apply_region_filter(docs, allowed)
        return docs