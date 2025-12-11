import sys, os
from pathlib import Path

# Ensure project root is on sys.path so 'src' imports work even when running this script from /scripts
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # /.../castone_3
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Sanity check and helpful warning
if not (PROJECT_ROOT / "src").exists():
    print(f"WARNING: Could not find 'src' under project root: {PROJECT_ROOT}. Imports may fail.")

import asyncio
import json
import time
import statistics
import os
from pathlib import Path
from typing import List, Dict, Any
import re

import matplotlib.pyplot as plt
import difflib

from src.graph_flow import build_graph
from src.config import load_faiss_index, LLM  # <-- LLM import 추가
from src.tools import find_and_select_best_place
from langchain_core.messages import HumanMessage

# --- 0. 간소화된 시나리오 정의 (10개) ---
TEST_SCENARIOS = [
    {"name": "서울_강남_카페", "destination": "서울특별자치시 강남구"},
    {"name": "광주_충장로_시장", "destination": "광주광역시 동구 충장로"}
]

# --- [변경] 검색 시도 전역 상수 추가: 전체 시도 제한 ---
MAX_SEARCH_ATTEMPTS = 20  # 전체 검색 시도 상한 (기본 10회)

# --- 검색 쿼리 패턴 / 사용자 키워드 (간단화) ---
SEARCH_KEYWORDS = ["가족", "혼자", "커플", "친구", "데이트"]
QUERY_PATTERNS = [
    "{kw}와 가기 좋은 {dest} 추천",
    "{dest}에서 {kw}에게 인기있는 장소 추천",
    "{dest} {kw} 추천 장소",
]

# Output locations
GRAPHS_DIR = Path(__file__).parent / "graphs"
OUTPUT_JSON = Path(__file__).parent / "service_eval_results.json"

# Ensure graph directory exists
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# --- similarity caching (현 프로세스 내 간단 캐시) ---
_similarity_cache: Dict[tuple, float] = {}

# --- compute similarity via LLM (Gemini) with fallback ---
async def compute_similarity_llm(a: str, b: str, weight_seq: float = 0.6, weight_token: float = 0.4) -> float:
    """
    LLM 기반 유사도 산출:
    - LLM(예: Gemini)을 호출하여 0.0 ~ 1.0 사이의 숫자만 반환하도록 지시합니다.
    - 실패 시 local_compute_similarity로 폴백.
    """
    # Normalize keys for caching
    key = (str(a or "").strip().lower(), str(b or "").strip().lower())
    if key in _similarity_cache:
        return _similarity_cache[key]

    # If both empty -> 1.0, if one empty -> 0.0
    if (not key[0] and not key[1]):
        _similarity_cache[key] = 1.0
        return 1.0
    if not key[0] or not key[1]:
        _similarity_cache[key] = 0.0
        return 0.0

    # Prompt: LLM이 정확히 숫자만 반환하도록 강하게 지시합니다.
    prompt = f"""
You are a helpful assistant that must compute the similarity between two place names.
Given:
A: "{a}"
B: "{b}"

Return only a single numeric value between 0.0 and 1.0 (inclusive) representing how similar A and B are.
- Output MUST contain nothing else except the number.
- Use as much precision as reasonable, up to 3 decimal places (e.g. 0.754).
- Think about semantic similarity and typical variations (spacing, abbreviations, minor subtitle differences).
Examples:
A: "해운대해수욕장", B: "해운대 해수욕장"  -> 0.95
A: "충장로", B: "충장로거리" -> 0.9

Now compute and output the numeric similarity for the provided A and B.
"""

    try:
        # `LLM` should be configured to call Gemini. Use async invoke and parse the returned text.
        # We wrap the prompt as a HumanMessage since LLM might expect messages
        llm_input = HumanMessage(content=prompt)
        llm_resp = await LLM.ainvoke({"messages": [llm_input]})
        resp_text = str(llm_resp)
        # Extract first numeric token between 0 and 1
        m = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", resp_text)
        if m:
            score = float(m.group(1))
            score = max(0.0, min(1.0, score))
            _similarity_cache[key] = score
            return score
        else:
            # fallback
            raise ValueError("LLM returned no numeric similarity")
    except Exception as e:
        # Fallback to local method if LLM fails/unreliable
        seq_score = difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
        # Jaccard token-based score
        a_tokens = set(re.sub(r'[^0-9a-z가-힣\s]', '', str(a).lower()).split())
        b_tokens = set(re.sub(r'[^0-9a-z가-힣\s]', '', str(b).lower()).split())
        jaccard = 0.0
        if a_tokens or b_tokens:
            inter = a_tokens.intersection(b_tokens)
            union = a_tokens.union(b_tokens)
            jaccard = len(inter) / len(union) if union else 0.0
        fallback_score = weight_seq * seq_score + weight_token * jaccard
        fallback_score = max(0.0, min(1.0, fallback_score))
        _similarity_cache[key] = fallback_score
        print(f"[SIMILARITY - FALLBACK] {a} vs {b} -> {fallback_score:.3f} (reason: {e})")
        return fallback_score

# --- Helper: normalize & DB 존재 확인 (유사도 기반, 개선) ---
def normalize_text_variants(s: str):
    """원본 + 공백 제거 + 단어 토크나이즈 등 변형들 생성"""
    if not s:
        return [""]
    s = str(s).strip()
    variants = set()
    # 기본
    variants.add(s.strip())
    # 소문자, 공백 제거
    simple = re.sub(r'\s+', '', s).lower()
    variants.add(simple)
    # 특수문자 제거
    alpha = re.sub(r'[^0-9a-z가-힣]', '', s).lower()
    variants.add(alpha)
    # 어떤 경우엔 '거리','구' 등 접미사 제거한 버전도 시도 (너무 과도하지 않게)
    for suf in ['구', '시', '동', '읍', '리']:
        if s.endswith(suf):
            variants.add(s[:-len(suf)].strip())
            variants.add(re.sub(r'\s+', '', s[:-len(suf)].strip()).lower())
    return list(variants)

# --- Helper: DB 존재 확인 (async, LLM similarity check) ---
async def place_exists_in_db(db, place_name: str, threshold: float = 0.75, k=5):
    """
    Search FAISS with name variants. For each unique candidate, call compute_similarity_llm to score.
    Return (exists: bool, best_match_name, best_score, top_candidates).
    """
    if not db or not place_name:
        return False, "", 0.0, []

    all_candidates = {}
    try:
        # variants generation (reuse previous normalize variants)
        def normalize_text_variants(s: str):
            if not s:
                return [""]
            s = str(s).strip()
            variants = set()
            variants.add(s.strip())
            variants.add(re.sub(r'\s+', '', s).lower())  # no spaces
            variants.add(re.sub(r'[^0-9a-z가-힣]', '', s).lower())  # only alnum
            for suf in ['구', '시', '동', '읍', '리']:
                if s.endswith(suf):
                    variants.add(s[:-len(suf)].strip())
                    variants.add(re.sub(r'\s+', '', s[:-len(suf)].strip()).lower())
            return list(variants)

        variants = normalize_text_variants(place_name)

        for v in variants:
            try:
                docs = db.similarity_search(v, k=k)
            except Exception:
                docs = []
            for d in docs:
                meta_name = (d.metadata.get('장소명') or d.metadata.get('name') or '').strip()
                if not meta_name:
                    content = getattr(d, 'page_content', '') or ''
                    if '은(는)' in content and '에 위치' in content:
                        try:
                            meta_name = content.split('은(는)')[0].strip()
                        except:
                            pass
                if not meta_name:
                    continue
                # compute LLM similarity (meta_name vs returned place_name)
                score = await compute_similarity_llm(meta_name, place_name)
                prev = all_candidates.get(meta_name)
                if prev is None or score > prev:
                    all_candidates[meta_name] = score
    except Exception as e:
        print(f"[DB CHECK] place_exists_in_db internal error: {e}")
        return False, "", 0.0, []

    if not all_candidates:
        return False, "", 0.0, []

    sorted_candidates = sorted(all_candidates.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_score = sorted_candidates[0]
    top = [{"name": name, "score": round(score, 3)} for name, score in sorted_candidates[:5]]
    exists = best_score >= threshold
    return exists, best_name, best_score, top

# --- EVALUATION: 기준 계산 ---
def compute_plan_validity(itinerary: List[Dict[str, Any]], total_days: int, activity_level: int) -> Dict[str, Any]:
    # day coverage: fraction of days that have at least one activity
    days_with_items = {}
    total_items = len(itinerary)
    items_with_time = 0
    category_counts = {}
    for it in itinerary:
        d = int(it.get('day', 1))
        days_with_items.setdefault(d, 0)
        days_with_items[d] += 1
        if it.get('start') and it.get('end'):
            items_with_time += 1
        typ = (it.get('type') or it.get('category') or "").lower()
        category_counts[typ] = category_counts.get(typ, 0) + 1

    day_coverage_rate = len(days_with_items) / total_days if total_days else 0
    # fraction of days meeting activity_level
    days_meet_activity = len([d for d, cnt in days_with_items.items() if cnt >= activity_level])
    activity_coverage_rate = days_meet_activity / total_days if total_days else 0
    time_info_rate = items_with_time / total_items if total_items > 0 else 0

    return {
        "total_items": total_items,
        "day_coverage_rate": day_coverage_rate,
        "activity_coverage_rate": activity_coverage_rate,
        "time_info_rate": time_info_rate,
        "category_counts": category_counts
    }

# --- Helper: sanitize objects for JSON serialization ---
def sanitize(obj):
    """Recursively convert objects like HumanMessage/AIMessage/ToolMessage into serializable structures."""
    if isinstance(obj, HumanMessage):
        try:
            return obj.content
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    # If it's a simple type, return as-is; otherwise convert to string
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        return str(obj)
    except Exception:
        return None

# --- Evaluate single scenario asynchronously ---
async def evaluate_scenario(APP, db, scenario: Dict[str, Any]) -> Dict[str, Any]:
    """단일 시나리오에 대해 E2E 평가 수행"""
    # Build input state similar to app
    messages = [HumanMessage(content=m) for m in scenario["messages"]]
    state = {
        "messages": messages,
        "itinerary": [],
        "destination": scenario["destination"],
        "dates": scenario["dates"],
        "preference": scenario["preference"],
        "total_days": scenario["total_days"],
        "activity_level": scenario["activity_level"],
        "current_planning_day": 1,
        "current_weather": "",
        "show_pdf_button": False
    }

    # Measure invoke latency
    t0 = time.time()
    response = await APP.ainvoke(state, config={"configurable": {"thread_id": "eval"}, "recursion_limit": 50})
    elapsed = time.time() - t0

    # Extract itinerary and other fields
    itinerary = response.get("itinerary", [])
    weather = response.get("current_weather", "")
    show_pdf = response.get("show_pdf_button", False)

    total_items = len(itinerary)
    hallucinations = 0
    missing_descriptions = 0
    category_matches = 0
    day_distrib = {}
    for item in itinerary:
        name = item.get("name", "") or ""
        typ = item.get("type", "") or item.get("category", "")
        desc = item.get("description", "")
        day = int(item.get("day", 1)) if item.get("day") else 1
        day_distrib[day] = day_distrib.get(day, 0) + 1

        if not name:
            hallucinations += 1
        else:
            try:
                exists, _, _, _ = await place_exists_in_db(db, name)
            except Exception as e:
                print(f"DEBUG: place_exists_in_db 호출 실패: {e}")
                exists = False
            if not exists:
                hallucinations += 1
        if not desc or desc.strip() == "":
            missing_descriptions += 1
        pref_ok = False
        pref_text = scenario["preference"].lower() if scenario.get("preference") else ""
        if ("카페" in typ or "카페" in pref_text) or ("식당" in typ or "맛집" in pref_text):
            pref_ok = True
            category_matches += 1

    hallucination_rate = (hallucinations / total_items) if total_items else 0.0
    description_missing_rate = (missing_descriptions / total_items) if total_items else 0.0
    category_match_rate = (category_matches / total_items) if total_items else 0.0

    # sanitize itinerary & response for JSON
    sanitized_itinerary = sanitize(itinerary)
    sanitized_response = sanitize({k: v for k, v in response.items() if k != "itinerary"})

    summary = {
        "scenario_name": scenario["name"],
        "elapsed_sec": elapsed,
        "total_items": total_items,
        "hallucinations": hallucinations,
        "hallucination_rate": hallucination_rate,
        "description_missing": missing_descriptions,
        "description_missing_rate": description_missing_rate,
        "category_matches": category_matches,
        "category_match_rate": category_match_rate,
        "day_distribution": day_distrib,
        "show_pdf_button": bool(show_pdf),
        "weather_snippet": weather[:200] if weather else "",
        "raw_itinerary": sanitized_itinerary,
        "raw_response": sanitized_response
    }
    return summary

# --- tool sampling (for response time) ---
async def sample_tool_times(db, samples=8):
    results = []
    if not db:
        return results
    try:
        docs = db.similarity_search("맛집", k=50)
    except Exception:
        docs = []
    names = []
    for d in docs:
        nm = d.metadata.get("장소명") or d.metadata.get("name")
        if nm:
            names.append(nm)
        if len(names) >= samples:
            break

    for name in names:
        t0 = time.time()
        try:
            # ✨ Use async invocation of the structured tool
            output_str = await find_and_select_best_place.ainvoke({"query": name, "destination": "서울"})
            # tool returns JSON string in many cases; try to parse
            try:
                output = json.loads(output_str)
            except Exception:
                output = output_str
        except Exception as e:
            output = f"ERROR: {e}"
        elapsed = time.time() - t0
        results.append({"query": name, "elapsed_sec": elapsed, "output": output})
    return {"tool_times": results}

# --- plotting helpers ---
def plot_response_times(response_times: List[float], out_path: Path):
    if not response_times:
        print(f"[PLOT] No response times to plot - skipping {out_path.name}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(response_times, bins=10, color="#4c72b0", alpha=0.8)
    plt.title("Response Time Distribution (seconds)")
    plt.xlabel("Seconds")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    try:
        plt.savefig(str(out_path))
        print(f"[PLOT] Saved response time histogram to {out_path}")
    except Exception as e:
        print(f"[PLOT] Failed to save response time histogram: {e}")
    finally:
        plt.close()


def plot_hallucination_rates(results: List[Dict[str, Any]], out_path: Path):
    if not results:
        print(f"[PLOT] No scenario results to plot - skipping {out_path.name}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [r["scenario_name"] for r in results]
    rates = [r["hallucination_rate"] for r in results]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(names)), rates, color="#dd8452")
    plt.ylim(0, 1)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.title("Hallucination Rate per Scenario")
    plt.ylabel("Hallucination Rate")
    for b, v in zip(bars, rates):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha='center')
    plt.tight_layout()
    try:
        plt.savefig(str(out_path))
        print(f"[PLOT] Saved hallucination rates chart to {out_path}")
    except Exception as e:
        print(f"[PLOT] Failed to save hallucination rates chart: {e}")
    finally:
        plt.close()


def plot_plan_validity(results: List[Dict[str, Any]], out_path: Path):
    if not results:
        print(f"[PLOT] No plan validity results to plot - skipping {out_path.name}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [r["scenario_name"] for r in results]
    day_cov = [r["plan_validity"]["day_coverage_rate"] for r in results]
    activity_cov = [r["plan_validity"]["activity_coverage_rate"] for r in results]
    time_rate = [r["plan_validity"]["time_info_rate"] for r in results]

    x = range(len(names))
    width = 0.25
    plt.figure(figsize=(11, 6))
    plt.bar([p - width for p in x], day_cov, width=width, label="Day Coverage", color="#4c72b0")
    plt.bar(x, activity_cov, width=width, label="Activity Coverage", color="#55a868")
    plt.bar([p + width for p in x], time_rate, width=width, label="Time Info Rate", color="#c44e52")
    plt.ylim(0, 1)
    plt.xticks(x, names, rotation=45, ha='right')
    plt.legend()
    plt.title("Plan Validity Metrics per Scenario")
    plt.tight_layout()
    try:
        plt.savefig(str(out_path))
        print(f"[PLOT] Saved plan validity chart to {out_path}")
    except Exception as e:
        print(f"[PLOT] Failed to save plan validity chart: {e}")
    finally:
        plt.close()

# --- [추가] 검색 기반 환각율/응답시간 평가 ---
async def evaluate_search_queries(APP, db, scenarios: List[Dict[str, Any]], keywords: List[str], queries_per_scenario: int = 3):
    """
    For each scenario, form several queries combining destination and user-like keywords,
    call find_and_select_best_place for each query, log the result (returned place),
    verify existence in DB, and record latency. Print logs + summary.
    """
    all_attempts = []
    all_times = []
    total_hallucinations = 0
    total_attempts = 0

    print("\n=== Search-based Hallucination & Latency Tests ===")
    for scenario in scenarios:
        dest = scenario.get("destination", "")
        scenario_name = scenario.get("name", "unnamed")
        print(f"\n--- Scenario: {scenario_name} ({dest}) ---")
        # Build queries: combine keyword & destination in different phrase patterns
        queries = []
        for kw in keywords[:queries_per_scenario]:
            queries.append(f"{dest} {kw} 추천")
            # also reversed
            queries.append(f"{kw}와 가기 좋은 {dest} 장소")
        # ensure unique and limit
        queries = list(dict.fromkeys(queries))[:queries_per_scenario]
        for q in queries:
            total_attempts += 1
            t0 = time.time()
            try:
                result_json_str = await find_and_select_best_place(q, dest)
            except Exception as e:
                elapsed = time.time() - t0
                print(f"[SEARCH] Query: '{q}' -> Tool call error: {e} (t={elapsed:.3f}s)")
                all_times.append(elapsed)
                continue

            elapsed = time.time() - t0
            all_times.append(elapsed)

            # Parse JSON returned by the tool (it's a json string)
            try:
                result_data = json.loads(result_json_str)
            except Exception:
                # sometimes the tool may return plain text
                result_data = {"name": result_json_str}

            returned_name = (result_data.get("name") or "").strip()
            # Check DB existence
            exists = place_exists_in_db(db, returned_name) if returned_name else False
            is_hallucination = False
            # count as hallucination if tool returns a name but it doesn't exist in DB
            if returned_name and returned_name.lower() not in ["추천 장소 없음", "이름미상", "정보없음"] and not exists:
                is_hallucination = True
                total_hallucinations += 1

            all_attempts.append({
                "scenario": scenario_name,
                "query": q,
                "returned_name": returned_name,
                "exists_in_db": exists,
                "is_hallucination": is_hallucination,
                "elapsed_sec": elapsed,
                "raw_result": result_data
            })

            # Immediate log
            print(f"[SEARCH] Query: '{q}' -> Returned: '{returned_name}' | Exists: {exists} | Hallucination: {is_hallucination} | t={elapsed:.3f}s")

    # Summary
    avg_time = statistics.mean(all_times) if all_times else 0.0
    hallucination_rate = (total_hallucinations / total_attempts) if total_attempts else 0.0
    print("\n=== Search Test Summary ===")
    print(f"Total attempts: {total_attempts}")
    print(f"Total hallucinations: {total_hallucinations}")
    print(f"Hallucination rate: {hallucination_rate:.3f}")
    print(f"Average search latency: {avg_time:.3f}s")

    # Return details for further inspection if needed
    return {
        "attempts": all_attempts,
        "total_attempts": total_attempts,
        "total_hallucinations": total_hallucinations,
        "hallucination_rate": hallucination_rate,
        "avg_latency": avg_time
    }

# --- 간소화된 검색 평가 실행 ---
async def evaluate_search_queries_simple(db, keywords: List[str], destinations: List[Dict], queries_per_dest: int = 1, max_attempts: int | None = None):
	"""
	Simplified search test runner: build queries from (dest, keyword, pattern),
	call find_and_select_best_place directly using async tool invocation, measure latency and DB existence.
	'max_attempts' param controls the global cap on total attempts; if None uses MAX_SEARCH_ATTEMPTS.
	"""
	from src.tools import find_and_select_best_place

	# use provided max_attempts or fallback to module-level default
	if max_attempts is None:
		max_attempts = MAX_SEARCH_ATTEMPTS

	attempts = []
	total_attempts = 0
	total_hallucinations = 0
	all_times = []
	stop = False  # 중단 플래그

	print("\n=== Simple Search-Based Tests ===")
	print(f"[INFO] max attempts set to {max_attempts}")
	for dest_entry in destinations:
		if stop:
			break
		dest = dest_entry.get("destination", "")
		name = dest_entry.get("name", dest)
		print(f"\n--- Destination: {name} ({dest}) ---")

		# --- 변경: 키워드별로 개별 쿼리 생성 및 실행 ---
		for kw in keywords:
			# 키워드 단위 쿼리 빌드 (패턴 개수 제한)
			per_kw_queries = []
			for p in QUERY_PATTERNS:
				per_kw_queries.append(p.format(kw=kw, dest=dest))
			per_kw_queries = per_kw_queries[:queries_per_dest]

			for q in per_kw_queries:
				# 전체 시도 제한 체크 (max_attempts 사용)
				if total_attempts >= max_attempts:
					stop = True
					print(f"[INFO] Reached max search attempts ({max_attempts}). Stopping further queries.")
					break

				total_attempts += 1
				t0 = time.time()
				try:
					# ✨ use async invocation
					result_str = await find_and_select_best_place.ainvoke({"query": q, "destination": dest})
				except Exception as e:
					elapsed = time.time() - t0
					print(f"[SEARCH ERROR][KEYWORD:{kw}] Query: '{q}' -> error: {e} (t={elapsed:.3f}s)")
					attempts.append({"destination": dest, "keyword": kw, "query": q, "error": str(e), "elapsed_sec": elapsed})
					all_times.append(elapsed)
					continue

				elapsed = time.time() - t0
				all_times.append(elapsed)

				# Parse result safely
				try:
					result_data = json.loads(result_str)
				except Exception:
					result_data = {"name": str(result_str)}

				returned_name = (result_data.get("name") or "").strip()

				# 🚨 비동기 place_exists_in_db 호출 및 반환값 분해
				if returned_name:
					try:
						exists, best_match_name, sim_score, debug_top_candidates = await place_exists_in_db(db, returned_name, threshold=0.75)
					except Exception as e:
						print(f"DEBUG: place_exists_in_db 호출 실패: {e}")
						exists, best_match_name, sim_score, debug_top_candidates = False, "", 0.0, []
				else:
					exists, best_match_name, sim_score, debug_top_candidates = False, "", 0.0, []

				is_hallucination = False
				if returned_name and returned_name.lower() not in ["추천 장소 없음", "이름미상", "정보없음"] and not exists:
					is_hallucination = True
					total_hallucinations += 1

				attempts.append({
					"destination": dest,
					"keyword": kw,  # ✨ [추가] 어떤 키워드로 검색했는지 기록
					"query": q,
					"returned_name": returned_name,
					"exists_in_db": exists,
					"best_match_name": best_match_name,
					"similarity_score": round(sim_score, 3),
					"top_candidates": debug_top_candidates,
					"is_hallucination": is_hallucination,
					"elapsed_sec": elapsed,
					"raw_result": result_data
				})

				print(f"[SEARCH][KEYWORD:{kw}] Query: '{q}' -> Returned: '{returned_name}' | Exists: {exists} (best='{best_match_name}' score={sim_score:.2f}) | Hallucination: {is_hallucination} | t={elapsed:.3f}s")

			# 키워드 루프 내부에서 최대 시도 도달했는지 확인
			if stop:
				break

	# End destinations
	avg_latency = statistics.mean(all_times) if all_times else 0.0
	hallucination_rate = (total_hallucinations / total_attempts) if total_attempts else 0.0

	summary = {
		"total_attempts": total_attempts,
		"total_hallucinations": total_hallucinations,
		"hallucination_rate": hallucination_rate,
		"avg_latency_sec": avg_latency,
	}

	print("\n=== Search Test Summary ===")
	print(f"Attempts: {total_attempts}, Hallucinations: {total_hallucinations}, Hallucination rate: {hallucination_rate:.3f}, Avg latency: {avg_latency:.3f}s")

	return {"attempts": attempts, "summary": summary}

# --- RUNNER: run all evaluations and produce graphs ---
async def run_all_evaluations(output_json_file: Path = OUTPUT_JSON):
    db = load_faiss_index()

    # Directly run simplified search tests (no E2E APP.ainvoke runs)
    print("[EVAL] Running simplified search-only tests...")
    search_results = await evaluate_search_queries_simple(db, SEARCH_KEYWORDS, TEST_SCENARIOS, queries_per_dest=1, max_attempts=MAX_SEARCH_ATTEMPTS)

    # Save results to JSON
    overall = {
        "search_results": search_results,
        "timestamp": time.time()
    }

    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print(f"[EVAL] Search-only results saved to {output_json_file}")

    return overall

# --- CLI ---
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    res = loop.run_until_complete(run_all_evaluations())
    print("Evaluation Complete. Results saved to", OUTPUT_JSON)
    print("Graphs saved to", GRAPHS_DIR)