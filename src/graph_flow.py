from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from src.config import LLM
from src.tools import AVAILABLE_TOOLS, TOOLS 
import json, re, difflib
import asyncio

# --- 1. 상태 정의 ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    destination: str
    dates: str
    group_type: str
    total_days: int
    style: str
    preference: str
    
    current_weather: str
    itinerary: List[Dict]
    show_pdf_button: bool 
    current_anchor: str
    
    dialog_stage: str 
    ban_list: List[str]
    last_deleted_spot: Dict

planner_prompt = """당신은 '엄격한 여행 스케줄러(Backend Logic)'입니다.
사용자와 대화하지 말고, 오직 주어진 기간({total_days}일)에 맞춰 **빈 스케줄을 기계적으로 채우는 작업**만 수행하세요.

🏗️ **[일차별 시퀀스 정의 (절대 규칙)]**
데이터를 채울 때 아래 순서를 반드시 지키세요.

🔴 **Day 1 (12:00 점심 시작)**
   1. 점심 (식당) -> 2. 카페 -> 3. 관광지 -> 4. 저녁 (식당)
   👉 (총 4곳 / 오전 일정 없음)

🟠 **Day 2 ~ Day {total_days}-1 (중간 날, 10:00 시작)**
   1. 관광지 -> 2. 점심 (식당) -> 3. 카페 -> 4. 관광지 -> 5. 저녁 (식당)
   👉 (총 5곳)

🟢 **Day {total_days} (마지막 날, 10:00 시작)**
   1. 관광지
   👉 (총 1곳만 찾고 즉시 종료!)

[행동 지침]
1. 현재 `itinerary` 상태를 확인하고, 위 시퀀스에서 **빠진 장소 하나**를 `find_and_select_best_place`로 찾으세요.
2. 장소를 찾을 때는 **동선(거리)**과 **사용자 선호**를 최우선으로 고려하세요.
3. 모든 슬롯이 채워지면, 즉시 `plan_itinerary_timeline` 도구를 호출하여 전체 경로를 계산하세요.
4. **중요:** `plan_itinerary_timeline` 도구를 호출한 후에는 아무 말도 하지 말고 종료하세요. (결과 출력은 에디터가 담당합니다.)
"""

editor_prompt = """당신은 '여행 일정 편집자'이자 '전문 여행 가이드'입니다.
플래너(Planner)가 넘겨준 **여행 데이터(JSON)**를 바탕으로, 사용자에게 **가장 매력적이고 상세한 여행 계획표**를 브리핑하세요.

🎯 **[핵심 역할: 데이터의 스토리텔링화]**
단순히 장소만 나열하지 말고, **"왜 이 장소가 사용자에게 딱 맞는지"**와 **"어떻게 가는지"**를 상세히 설명해야 합니다.

1.  **추천 사유 (Reasoning):**
    * 사용자의 선호 정보(`{user_info}`)와 장소의 특징(`description`)을 연결하여 자연스럽게 설명하세요.
    * 예: "사용자님이 **가족 여행**이고 **소고기**를 선호하시므로, 룸이 완비된 이곳을 추천드려요."
2.  **이동 정보 (Transport):**
    * 데이터의 `transport_detail`(예: '1003번 버스', '도보')을 읽어 안내하되, **"도보->버스->도보" 처럼 기계적인 표현을 쓰지 마세요.**
    - **경고:** 절대로 없는 버스 번호나 경로를 지어내지 마세요. 데이터에 `1003번`이 없으면 `1003번`이라고 말하지 마세요.
    - 데이터가 `도보`라면 "걸어서 이동합니다", 버스라면 "N번 버스를 타고..."라고 말하세요.

🎨 **[최종 출력 양식 (Markdown) - 필수 엄수]**
일정이 확정되었거나 초안을 보여줄 때는 **반드시** 아래 포맷을 따르세요.

## ✈️ [여행지] 맞춤 여행 계획표 ({total_days}일)


🌤️ **[날씨 브리핑 지침]**
- 제공된 `{current_weather}` 데이터를 그대로 나열하지 마세요. (00:00 4도, 03:00 8도... -> **금지**)
- **반드시 한 문장으로 요약해서 조언해주세요.**
- 예시: "일교차가 크니 겉옷을 챙기시고, 오후에는 비 소식이 있으니 우산을 준비하세요."

🚌 **[이동 정보 안내 지침]**
- JSON 데이터의 `transport_detail`을 정확히 읽어서 안내하세요.
- 만약 `도보`라면 "걸어서 이동합니다"라고 하고, 버스나 지하철이면 구체적인 번호를 언급하세요.
- **절대 프롬프트의 예시 값을 그대로 사용하지 마세요.** 실제 데이터에 있는 값만 말하세요.

### 🗓️ Day N (YYYY-MM-DD)
---
*(JSON 데이터의 순서와 시간을 정확히 반영하세요)*

1️⃣ **[HH:MM] 장소명** (카테고리)
> 💡 *"[사용자 선호]를 고려하여 추천했어요. [장소 특징]을 즐겨보세요." (1~2문장)*
   
   ⬇️ *이동: [교통편 핵심 정보] (약 N분 소요)*

2️⃣ **[HH:MM] 다음 장소명** (카테고리)
... (반복)

---
[다운로드 안내]
"이 일정으로 확정하시겠습니까? 아래 버튼을 눌러 PDF를 받아보세요."

🚨 **[주의사항]**
- **Day 1**은 12:00 점심부터, **Day 2 이후**는 10:00 관광부터 시작하는 규칙을 준수하여 표시하세요.
- 데이터에 있는 **날씨 정보**를 꼭 상단에 표시하세요.
- 사용자가 수정을 요청하면(`수정`, `삭제`, `추가`) 주저 없이 도구를 사용하여 반영하고, 다시 이 양식으로 보여주세요.
"""


# --- 3. 에이전트 생성 ---
def create_agent(system_prompt):
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("placeholder", "{messages}")])
    llm_with_tools = LLM.bind_tools(TOOLS)
    chain = prompt | llm_with_tools
    
    async def agent_node(state: AgentState):
        user_info_str = f"동행:{state.get('group_type')}, 여행스타일:{state.get('style')}, 상세선호:{state.get('preference')}"
        
        input_vars = {**state, "user_info": user_info_str}
        filled_prompt = await prompt.ainvoke(input_vars)
        response = await llm_with_tools.ainvoke(filled_prompt)
        return {"messages": [response]}
    return agent_node

PlannerAgent = create_agent(planner_prompt)
EditorAgent = create_agent(editor_prompt)

# --- 4. 라우터 ---
def entry_router(state: AgentState):
    current_stage = state.get("dialog_stage", "planning")
    last_message = state['messages'][-1]
    
    if current_stage == "editing":
        return "EditorAgent"

    if isinstance(last_message, HumanMessage):
        content = last_message.content
        if any(k in content for k in ["수정", "바꿔", "추가", "삭제", "빼줘", "더 갈래", "변경"]):
            return "EditorAgent"
            
    return "PlannerAgent"

def agent_router(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        if len(messages) >= 3:
            prev_tool_msg = messages[-2]
            prev_ai_msg = messages[-3]
            
            if isinstance(prev_tool_msg, ToolMessage) and isinstance(prev_ai_msg, AIMessage):
                current_tools = [t['name'] for t in last_message.tool_calls]
                prev_tools = [t['name'] for t in prev_ai_msg.tool_calls] if prev_ai_msg.tool_calls else []
                
                target_tools = ["plan_itinerary_timeline", "optimize_and_get_routes"]
                
                for tool in current_tools:
                    if tool in target_tools and tool in prev_tools:
                        print(f"DEBUG: 🛑 재귀 루프 감지! ({tool} 연속 호출) -> 강제 종료")
                        return "EditorAgent"

        return "call_tools"
        
    # 2. PDF 버튼 활성화 시 종료
    if state.get('show_pdf_button'):
        return END

    return END

def normalize_name(raw_name):
    """이름 비교를 위해 괄호, 공백, 특수문자 제거 (Fuzzy Matching용)"""
    if not raw_name: return ""
    name = re.sub(r'\(.*?\)|\[.*?\]', '', str(raw_name))
    name = re.sub(r'[^a-zA-Z0-9가-힣]', '', name)
    return name

def get_category_group(type_str):
    """카테고리 단순화 (Planning 모드 정렬용)"""
    t = str(type_str).replace("맛집", "식당").replace("음식점", "식당")
    if any(x in t for x in ["식당", "요리", "레스토랑", "반점", "회관", "고기", "뷔페"]): return "식당"
    if any(x in t for x in ["카페", "커피", "베이커리", "디저트", "찻집"]): return "카페"
    return "관광지"



# src/graph_flow.py 내부

# --- [1] 도구 실행 함수 (Executor) ---
async def execute_tools(state: AgentState, current_itinerary: List[Dict]):
    """
    도구 실행 전 '삭제 대상'의 '직전 장소(Previous Place)'를 찾아 앵커로 설정합니다.
    (이동 흐름 끊김 방지)
    """    
    last_message = state['messages'][-1]
    tool_calls = last_message.tool_calls
    user_info_str = f"모임:{state.get('group_type')}, 스타일:{state.get('style')}, 선호:{state.get('preference')}"
    
    results = [] 
    current_ban_list = state.get("ban_list", [])

    is_schedule_full = False
    current_stage = state.get("dialog_stage", "planning")
    
    if current_stage == "planning":
        places_only = [x for x in current_itinerary if x.get('type') != 'move']
        total_days = state.get('total_days', 1)
        
        if places_only:
            last_day = places_only[-1].get('day', 1)
            count = len([p for p in places_only if p.get('day') == last_day])
            
            # 규칙: 마지막 날은 1곳, 그 외는 4~5곳
            if last_day == 1: max_places = 4
            elif last_day == total_days: max_places = 1
            else: max_places = 5
            
            # 마지막 날이고, 개수가 꽉 찼다면 -> 풀방 선언
            if (last_day >= total_days) and (count >= max_places):
                is_schedule_full = True
                print(f"🛑 [DEBUG_PRECHECK] 일정 가득 참 감지 (Day {last_day}, {count}/{max_places}). 검색 차단 예정.")

    # 🔍 [Step 1] Pre-scan: 삭제 대상의 '이전 장소' 찾기
    dynamic_anchor = None
    pending_deletions = []
    
    for tool_call in tool_calls:
        if tool_call.get("name") in ["delete_place", "replace_place"]:
            args = tool_call.get("args", {})
            tgt = args.get('place_name') or args.get('old')
            
            if tgt:
                tgt_norm = normalize_name(tgt)
                best_match_idx = -1
                highest_score = 0.0
                
                # 리스트에서 삭제 대상의 인덱스 찾기
                for i, place in enumerate(current_itinerary):
                    p_name = place.get('name', '')
                    p_norm = normalize_name(p_name)
                    
                    # 포함 관계 or 유사도
                    score = difflib.SequenceMatcher(None, tgt_norm, p_norm).ratio()
                    if tgt_norm in p_norm: score = 1.0
                    
                    if score > highest_score:
                        highest_score = score
                        best_match_idx = i
                
                # 유사도가 높고 인덱스를 찾았다면
                if best_match_idx != -1 and highest_score > 0.5:
                    if best_match_idx > 0:
                        # 🚨 [핵심 수정] 삭제 대상(Index)의 '직전 장소(Index-1)'를 앵커로 설정
                        prev_place = current_itinerary[best_match_idx - 1]
                        dynamic_anchor = prev_place.get('name')
                        print(f"DEBUG: ⚓️ 앵커 변경: '{tgt}' 삭제 -> 직전 장소 '{dynamic_anchor}' 기준 검색")
                    else:
                        # 만약 첫 번째 장소(Index 0)를 지운다면? -> 출발지/숙소를 앵커로
                        dynamic_anchor = state.get('current_anchor') or state.get('destination')
                        print(f"DEBUG: ⚓️ 앵커 변경: 첫 장소 삭제 -> 출발지 '{dynamic_anchor}' 기준 검색")
                    break

    # 🔍 [Step 2] 도구 순차 실행
    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        args = tool_call.get("args", {})
        
        # 1. PDF 도구 건너뜀
        if tool_name == "confirm_and_download_pdf":
            results.append((None, tool_name, "SKIP_FOR_LATER"))
            continue

        # 2. 검색 도구: 동적 앵커 적용
        if tool_name == "find_and_select_best_place":
            if is_schedule_full:
                print("🛑 [DEBUG] 일정 초과로 인한 검색 도구 실행 차단 (Block)")
                stop_msg = (
                    "\n\n[SYSTEM ALERT] 🛑 일정이 가득 찼습니다.\n"
                    "더 이상 장소를 검색하지 마세요. (검색 도구 실행 차단됨)\n"
                    "즉시 `plan_itinerary_timeline` 도구를 호출하여 일정을 확정하세요."
                )
                # 도구를 실행한 척 하면서 경고 메시지만 리턴
                results.append((ToolMessage(tool_call_id=tool_call['id'], content=stop_msg), tool_name, stop_msg))
                continue
            existing_names = [item['name'] for item in current_itinerary if 'name' in item]
            final_exclude_list = list(set(existing_names + pending_deletions + current_ban_list))
            args['exclude_places'] = final_exclude_list
            
            # 🚨 찾아낸 '직전 장소'를 앵커로 주입
            if dynamic_anchor:
                args['anchor'] = dynamic_anchor
            else:
                args['anchor'] = state.get('current_anchor') or state.get('destination')
            
            args['user_info'] = user_info_str
        
        # 3. 타임라인 도구
        elif tool_name == "plan_itinerary_timeline":
            args['itinerary'] = current_itinerary
            
        # 4. 실행
        if tool_name in AVAILABLE_TOOLS:
            try:
                res = await AVAILABLE_TOOLS[tool_name].ainvoke(args)
                results.append((ToolMessage(tool_call_id=tool_call['id'], content=str(res)), tool_name, str(res)))
            except Exception as e:
                results.append((ToolMessage(tool_call_id=tool_call['id'], content=f"Error: {e}"), tool_name, None))
        else:
            results.append((None, None, None))

    return results

# --- [2] 삭제/교체 처리 함수 (Deleter) ---
def process_deletions(tool_results, itinerary):
    places_only = [x for x in itinerary if x.get('type') != 'move']
    empty_slot_info = None
    modification_happened = False
    explicit_reschedule = False
    tool_outputs = []

    print("\n[DEBUG] --- process_deletions 시작 ---")

    for tool_message, tool_name, raw_json_output in tool_results:
        if raw_json_output == "SKIP_FOR_LATER": continue
        
        if tool_message: tool_outputs.append(tool_message)
        if not raw_json_output: continue

        if tool_name == "plan_itinerary_timeline":
            explicit_reschedule = True

        if tool_name in ["delete_place", "replace_place"]:
            try:
                print(f"[DEBUG] 도구 호출 확인: {tool_name}")
                data = json.loads(raw_json_output)
                tgt = data.get('place_name') or data.get('old')
                
                if tgt:
                    tgt_norm = normalize_name(tgt)
                    print(f"[DEBUG] 🎯 삭제 대상(원본): '{tgt}'")
                    print(f"[DEBUG] 🎯 삭제 대상(정규화): '{tgt_norm}'")
                    
                    best_match_idx = -1
                    highest_ratio = 0.0
                    
                    print(f"[DEBUG] >> 매칭 탐색 시작 (총 {len(places_only)}개 장소)")

                    for i, place in enumerate(places_only):
                        place_name = place.get('name', '')
                        place_norm = normalize_name(place_name)
                        
                        ratio = difflib.SequenceMatcher(None, tgt_norm, place_norm).ratio()
                        
                        is_included = (tgt_norm in place_norm) and (len(tgt_norm) > 1)
                        
                        if is_included:
                            ratio = max(ratio, 0.9)
                        
                        print(f"  [{i}] '{place_norm}' vs '{tgt_norm}' | 유사도: {ratio:.4f} | 포함여부: {is_included}")

                        if ratio > highest_ratio:
                            highest_ratio = ratio
                            best_match_idx = i
                            print(f"     👉 현재 1등 갱신! (Index: {i}, Score: {highest_ratio:.4f})")
                    
                    print(f"[DEBUG] >> 탐색 종료. 최고 점수: {highest_ratio:.4f}, 인덱스: {best_match_idx}")

                    if best_match_idx != -1 and highest_ratio > 0.5:
                        target_place = places_only[best_match_idx]
                        deleted_name=target_place['name']
                        empty_slot_info = {'index': best_match_idx, 'day': target_place.get('day', 1)}
                        
                        print(f"[DEBUG] ✅ 삭제 확정! Index {best_match_idx}: '{target_place['name']}'")
                        places_only.pop(best_match_idx)
                        is_still_there = any(p.get('name') == deleted_name for p in places_only)
                        if not is_still_there:
                            print(f"DEBUG: ✅ 확인 사살 완료. '{deleted_name}'은(는) 리스트에서 완전히 사라졌습니다.")
                        else:
                            print(f"DEBUG: ⚠️ 경고! '{deleted_name}'이(가) 아직 리스트에 남아있습니다. (동명이인 주의)")
                        modification_happened = True
                    else:
                        print(f"[DEBUG] ❌ 삭제 실패: 매칭되는 장소가 없거나 점수가 너무 낮음.")

            except Exception as e:
                print(f"[DEBUG] 🚨 에러 발생: {e}")
                import traceback
                traceback.print_exc()

    print("[DEBUG] --- process_deletions 종료 ---\n")
    return places_only, empty_slot_info, modification_happened, explicit_reschedule, tool_outputs

# --- [3] 추가/삽입 처리 함수 (Adder) ---
def process_additions(tool_results, itinerary: List[Dict], empty_slot_info, state: AgentState):
    print(f"🔍 [DEBUG_ADD] 받은 empty_slot_info: {empty_slot_info}")
    updated_itinerary = [x for x in itinerary if x.get('type') != 'move']
    print(f"🔍 [DEBUG_ADD] 순수 장소 리스트(Before): {[x.get('name') for x in updated_itinerary]}")
    current_stage = state.get("dialog_stage", "planning")
    new_anchor = state.get('current_anchor')
    modification_happened = False
    show_pdf = False
    is_full_stop = False
    
    # PDF 요청이 있었는지 확인
    for _, tool_name, raw_output in tool_results:
        if tool_name == "confirm_and_download_pdf":
            print(f"👉 [DEBUG_ADD] 도구 처리 중: {tool_name}")
            show_pdf = True

    for tool_message, tool_name, raw_json_output in tool_results:
        if raw_json_output == "SKIP_FOR_LATER": continue # PDF는 여기서 처리 안 함
        if not raw_json_output: continue
        
        if tool_name == "find_and_select_best_place":
            try:
                item_json = json.loads(raw_json_output)
                place_name = item_json.get('name')
                if not place_name or place_name == "추천 장소 없음": 
                    
                    print("⚠️ [DEBUG_ADD] 유효하지 않은 장소 데이터 -> Skip")
                    continue
                
                modification_happened = True

                if empty_slot_info:
                    print(f"⚡ [DEBUG_ADD] 빈자리 정보 감지됨! 로직 진입")
                    target_idx = empty_slot_info['index']
                    target_day = empty_slot_info['day']

                    print(f"   - 목표 인덱스: {target_idx}")
                    print(f"   - 현재 리스트 길이: {len(updated_itinerary)}")
                    
                    # 인덱스 안전장치 (리스트 범위 초과 방지)
                    if target_idx > len(updated_itinerary):
                        print(f"⚠️ [DEBUG_ADD] 인덱스 초과! ({target_idx} > {len(updated_itinerary)}) -> 맨 뒤로 보정")
                        target_idx = len(updated_itinerary)
                    
                    item_json['day'] = target_day
                    
                    # 삽입 (Insert)
                    updated_itinerary.insert(target_idx, item_json)
                    print(f"✅ [DEBUG_ADD] Insert 수행 완료! 이름: {place_name} -> Index: {target_idx}")
                    print(f"🔍 [DEBUG_ADD] 삽입 직후 리스트: {[x.get('name') for x in updated_itinerary]}")
                    print(f"DEBUG: ♻️ 빈자리(Place Index {target_idx})에 '{place_name}' 삽입 성공!")
                    
                    empty_slot_info = None 
                    new_anchor = place_name
                    continue
                
                # [CASE 1] Planning Mode
                if current_stage == "planning":
                    replaced = False
                    if updated_itinerary:
                        last = updated_itinerary[-1]
                        if get_category_group(item_json.get('type')) == get_category_group(last.get('type')):
                            if place_name != last.get('name'):
                                item_json['day'] = last.get('day', 1)
                                updated_itinerary.pop()
                                updated_itinerary.append(item_json)
                                replaced = True
                    if not replaced:
                        current_places = updated_itinerary 
                        last_day = current_places[-1].get('day', 1) if current_places else 1
                        count = len([p for p in current_places if p.get('day') == last_day])
                        total_days = state.get('total_days', 1)
                        
                        if last_day == 1:
                            max_places = 4
                        elif last_day == total_days:
                            max_places = 1 
                        else:
                            max_places = 5
                        
                        if count >= max_places:
                            if last_day >= total_days:
                                print(f"🛑 [DEBUG_ADD] 일정 가득 참 (Day {last_day}, {count}곳). '{place_name}' 추가 거부.")
                                is_full_stop = True
                                continue 
                            else:
                                day_to_add = last_day + 1
                        else:
                            day_to_add = last_day
                        
                        item_json['day'] = day_to_add
                        updated_itinerary.append(item_json)
                        print(f" [DEBUG_ADD] Planning Append: {place_name} (Day {day_to_add})")
                else:
                    if empty_slot_info:
                        item_json['day'] = empty_slot_info['day']
                        insert_idx = empty_slot_info['index']
                        if insert_idx > len(updated_itinerary): insert_idx = len(updated_itinerary)
                        updated_itinerary.insert(insert_idx, item_json)
                        print(f"DEBUG: ♻️ 빈자리(Index {insert_idx})에 '{place_name}' 삽입")
                        empty_slot_info = None 
                    else:
                        target_day = 1
                        insert_idx = len(updated_itinerary)
                        if new_anchor:
                            for idx, p in enumerate(updated_itinerary):
                                if p.get('name') == new_anchor:
                                    target_day = p.get('day', 1)
                                    insert_idx = idx + 1
                                    break
                            if insert_idx == len(updated_itinerary) and updated_itinerary:
                                target_day = updated_itinerary[-1].get('day', 1)
                        else:
                            if updated_itinerary: target_day = updated_itinerary[-1].get('day', 1)
                        item_json['day'] = target_day
                        updated_itinerary.insert(insert_idx, item_json)
                
                new_anchor = place_name
            except Exception as e: pass

    print(f"🔥🔥 [DEBUG_ADD] 종료. 최종 리스트: {[x.get('name') for x in updated_itinerary]} 🔥🔥\n")

    return updated_itinerary, new_anchor, modification_happened, show_pdf, is_full_stop

# --- [4] 타임라인 재계산 함수 (Scheduler) ---
async def update_timeline(itinerary, old_itinerary_json, modification_happened, explicit_reschedule, current_stage):
    # (기존 코드와 동일)
    new_itinerary_json = json.dumps(itinerary, sort_keys=True)
    is_changed = old_itinerary_json != new_itinerary_json
    should_calculate = False
    if current_stage == "planning":
        if explicit_reschedule and is_changed: should_calculate = True
    else:
        if explicit_reschedule or modification_happened: should_calculate = True

    if should_calculate:
        try:
            print("DEBUG: 🔄 타임라인 재계산 수행...")
            timeline_tool = AVAILABLE_TOOLS["plan_itinerary_timeline"]
            res = await timeline_tool.ainvoke({"itinerary": itinerary})
            return json.loads(res)
        except Exception: return itinerary
    return itinerary

def reorganize_itinerary_planning(items):
    """
    Planning 단계에서 Day 1의 '점심 -> 카페 -> 관광 -> 저녁' 순서를 강제로 맞춥니다.
    """
    if not items: return []
    
    # 날짜별로 그룹화
    days = sorted(list(set(item.get('day', 1) for item in items)))
    final_list = []
    
    for day in days:
        day_items = [x for x in items if x.get('day', 1) == day]
        
        # 카테고리별 분류
        rests = [x for x in day_items if get_category_group(x.get('type')) == "식당"]
        cafes = [x for x in day_items if get_category_group(x.get('type')) == "카페"]
        tours = [x for x in day_items if get_category_group(x.get('type')) == "관광지"]
        
        sorted_day = []
        
        if day == 1:
            if rests: sorted_day.append(rests.pop(0)) # 1. 점심
            sorted_day.extend(cafes)                  # 2. 카페
            sorted_day.extend(tours)                  # 3. 관광지
            sorted_day.extend(rests)                  # 4. 저녁 (남은 식당)
        else:
            sorted_day.extend(tours)
            if rests: sorted_day.append(rests.pop(0)) # 점심
            sorted_day.extend(cafes)
            sorted_day.extend(rests) # 저녁
            
        final_list.extend(sorted_day)
        
    return final_list

# --- [메인] 최종 통합 노드 ---
async def call_tools_node(state: AgentState):
    old_itinerary_str = json.dumps(state.get('itinerary', []), sort_keys=True)
    current_itinerary = [dict(item) for item in state.get('itinerary', [])] if state.get('itinerary') else []
    
    saved_slot_info = state.get("last_deleted_spot")
    current_ban_list = state.get("ban_list", [])

    # 1. 도구 실행 (PDF는 여기서 실행 안 함!)
    tool_results = await execute_tools(state, current_itinerary)
    
    # 2. 삭제 처리
    current_itinerary, empty_slot_info, mod_deleted, explicit_reschedule, tool_outputs = \
        process_deletions(tool_results, current_itinerary)
    
    if mod_deleted and empty_slot_info:
       
        for _, name, raw in tool_results:
            if name in ["delete_place", "replace_place"]:
                try:
                    data = json.loads(raw)
                    tgt = data.get('place_name') or data.get('old')
                    if tgt and tgt not in current_ban_list:
                        current_ban_list.append(tgt)
                        print(f"DEBUG: 🚫 블랙리스트에 추가: {tgt}")
                except: pass
    final_slot_info = empty_slot_info if empty_slot_info else saved_slot_info
    
    # 3. 추가 처리
    current_itinerary, new_anchor, mod_added, show_pdf, is_full_stop = \
        process_additions(tool_results, current_itinerary, final_slot_info, state)
    
    if is_full_stop and tool_outputs:
        print("🛑 [DEBUG] Full Stop 신호 감지 -> LLM 메시지 강제 변경")
        last_msg = tool_outputs[-1]
        if isinstance(last_msg, ToolMessage):
            # LLM에게 보여줄 시스템 경고 메시지
            stop_warning = (
                "\n\n[SYSTEM ALERT] 🛑 일정이 가득 찼습니다 (종료 조건 달성).\n"
                "더 이상 장소를 검색하거나 추가하지 마세요.\n"
                "규칙에 따라 즉시 `plan_itinerary_timeline` 도구를 호출하여 일정을 확정하고 종료하세요."
            )
       
            new_msg = ToolMessage(
                tool_call_id=last_msg.tool_call_id,
                content=stop_warning
            )
            tool_outputs[-1] = new_msg
    
    next_remembered_spot = None if mod_added else final_slot_info
    
    # 4. 재계산 처리
    modification_happened = mod_deleted or mod_added
    current_stage = state.get("dialog_stage", "planning")
    
    final_itinerary = await update_timeline(
        current_itinerary, 
        old_itinerary_str, 
        modification_happened, 
        explicit_reschedule, 
        current_stage
    )

    if explicit_reschedule:
        current_stage = "editing"

    # 5. 정렬
    if current_stage == "planning":
        final_itinerary = reorganize_itinerary_planning(final_itinerary)    
    else:
        final_itinerary = sorted(final_itinerary, key=lambda x: x.get('day', 1))

    if show_pdf:
        print("DEBUG: 📄 최종 PDF 생성 시작...")
        try:
            pdf_tool = AVAILABLE_TOOLS["confirm_and_download_pdf"]
            pdf_result = await pdf_tool.ainvoke({"itinerary": final_itinerary})
            

            for call in state['messages'][-1].tool_calls:
                if call['name'] == "confirm_and_download_pdf":
                    tool_outputs.append(ToolMessage(tool_call_id=call['id'], content=str(pdf_result)))
                    break
        except Exception as e:
            print(f"DEBUG: PDF 생성 실패 {e}")

    return {
        "messages": tool_outputs, 
        "itinerary": final_itinerary,
        "show_pdf_button": show_pdf,
        "dialog_stage": current_stage, 
        "current_anchor": new_anchor,
        "ban_list":current_ban_list,
        "last_deleted_spot": next_remembered_spot
    }
    
   
def route_after_tools(state: AgentState):
    """도구 실행 후 경로 결정"""
    if state.get("show_pdf_button"):
        return END
    
    # 2. [핵심] 사용자에게 보여줄 메시지(요약본)가 생성되었다면 즉시 종료
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage):
        return END

    # 3. 메시지가 없다면(중간 연산), 원래 에이전트로 복귀
    if state.get("dialog_stage") == "editing":
        return "EditorAgent"
    
    return "PlannerAgent"

# --- 6. 그래프 빌드 ---
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("PlannerAgent", PlannerAgent)
    workflow.add_node("EditorAgent", EditorAgent)
    workflow.add_node("call_tools", call_tools_node)
    
    workflow.set_conditional_entry_point(
        entry_router,
        {"PlannerAgent": "PlannerAgent", "EditorAgent": "EditorAgent"}
    )
    
    workflow.add_conditional_edges(
        "PlannerAgent", agent_router, {
            "call_tools": "call_tools",
            "EditorAgent": "EditorAgent",
            END: END}
    )
    workflow.add_conditional_edges(
        "EditorAgent", agent_router, {
            "call_tools": "call_tools",
            "PlannerAgent": "PlannerAgent", 
            END: END}
    )
    
    workflow.add_conditional_edges(
        "call_tools", route_after_tools,
        {"PlannerAgent": "PlannerAgent", "EditorAgent": "EditorAgent", END: END}
    )
    
    return workflow.compile()