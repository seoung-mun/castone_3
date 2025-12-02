
import re
from typing import List, Set, Dict
from langchain.schema import Document

CANON: List[str] = [
    "서울","부산","대구","인천","광주","대전","울산","세종",
    "경기","강원","충북","충남","전북","전남","경북","경남","제주"
]

# 흔한 별칭/영문/오타 경향
ALIASES: Dict[str, Set[str]] = {
    "서울": {"서울시","서울특별시","seoul","서울특"},
    "부산": {"부산시","부산광역시","busan"},
    "대구": {"대구시","대구광역시","daegu"},
    "인천": {"인천시","인천광역시","incheon"},
    "광주": {"광주시","광주광역시","gwangju"},
    "대전": {"대전시","대전광역시","daejeon"},
    "울산": {"울산시","울산광역시","ulsan"},
    "세종": {"세종시","세종특별자치시","sejong"},
    "경기": {"경기도","gyeonggi"},
    "강원": {"강원도","gangwon"},
    "충북": {"충청북도","chungbuk"},
    "충남": {"충청남도","chungnam"},
    "전북": {"전라북도","jeonbuk"},
    "전남": {"전라남도","jeonnam"},
    "경북": {"경상북도","gyeongbuk"},
    "경남": {"경상남도","gyeongnam"},
    "제주": {"제주도","jeju"},
}

# 권역/집합 토큰 → 광역시/도 집합 확장
MACROS: Dict[str, Set[str]] = {
    "수도권": {"서울","경기","인천"},
    "부울경": {"부산","울산","경남"},
    "영남": {"부산","대구","울산","경북","경남"},
    "호남": {"광주","전북","전남"},
    "충청권": {"대전","세종","충북","충남"},
    "강원권": {"강원"},
    "제주권": {"제주"},
    "서울근교": {"서울","경기","인천"},
    "수도": {"서울"},   # 자주 나오는 축약/오타 방지용
}

SEP = re.compile(r"[,\|/·\-]")  # 흔한 구분자

def _tokenize_query(q: str) -> List[str]:
    q = SEP.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q.split()

def parse_regions_from_query(query: str, fuzzy: bool = True, fuzzy_threshold: int = 85) -> Set[str]:
    """쿼리에서 광역시/도 집합을 뽑는다. (정확일치 → 별칭/권역 → 퍼지매칭 순)"""
    q = query.lower()
    tokens = _tokenize_query(q)

    found: Set[str] = set()

    # 1) 정확 일치(한글 그대로 들어온 경우)
    for c in CANON:
        if c in query:
            found.add(c)

    # 2) 권역/매크로
    for macro, expands in MACROS.items():
        if macro in query:
            found |= expands

    # 3) 별칭 일치
    for canon, alset in ALIASES.items():
        if any(a.lower() in q for a in alset):
            found.add(canon)

    # 4) 퍼지 매칭(선택)
    if fuzzy:
        try:
            from rapidfuzz import process, fuzz
            # 후보군: 토큰들 각각을 CANON과 비교
            for t in tokens:
                # 한글/영문/숫자 혼합 토큰에서 길이 2 미만은 스킵
                if len(t) < 2: 
                    continue
                cand, score, _ = process.extractOne(
                    t, CANON, scorer=fuzz.WRatio  # 종합 점수
                )
                if score >= fuzzy_threshold:
                    found.add(cand)
        except Exception:
            pass  # rapidfuzz 미설치면 건너뜀

    return found

def filter_docs_by_region(docs: List[Document], allowed: Set[str], field: str = "광역시/도",
                          drop_unknown: bool = True) -> List[Document]:
    """리트리버 결과를 광역시/도로 컷. 결과 메타데이터는 정규화되어 있다고 가정."""
    if not allowed:
        return docs
    out = []
    for d in docs:
        reg = (d.metadata or {}).get(field)
        if reg in allowed:
            out.append(d)
        elif reg in (None, "", "nan"):
            if not drop_unknown:
                out.append(d)
    return out
