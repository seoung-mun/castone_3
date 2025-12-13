from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from src.config import LLM
from src.tools import AVAILABLE_TOOLS, TOOLS 
import json
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
    
    dialog_stage: str # 'planning' | 'editing'

planner_prompt = """당신은 '엄격한 여행 스케줄러'입니다.
주어진 여행 기간({total_days}일) 동안 아래 [고정 스케줄]을 기계적으로 따르세요.

🚨 **[최우선 종료 규칙]**
1. **여행 기간이 2일 이상인 경우**:
   - **마지막 날(Day {total_days})**에는 **'관광지' 딱 1곳**만 찾고 종료합니다.
   - 이때는 식당, 카페, 저녁 일정을 절대 추가하지 말고 즉시 `plan_itinerary_timeline`을 호출하세요.
2. **여행 기간이 1일(당일치기)인 경우**:
   - 아래 **[Day 1]** 스케줄(총 4곳)을 모두 채워야 끝납니다.
   - 절대 '마지막 날 관광지 1곳' 규칙을 적용하지 마세요.

✅ **[최종 결과물 필수 요구사항]**
일정을 확정할 때(`plan_itinerary_timeline` 결과) 다음 3가지 요소에 집중하세요:
1. **각 일정의 대략적인 시간** (예: 10:00 ~ 11:30)
2. **장소 간 이동 시간** (예: 약 30분 소요)
3. **상세 교통편 정보** (예: 1003번 버스 ➡️ 도보)
4. **장소에 대한 정보** (예 : 맛골 : 뼈해장국이 맛있고 고기를 좋아하는 사용자님께 고기 양도 많아서 한끼 식사로는 손색없어요.)
*위 '시간'과 '이동', *장소에 대한 간단한 소개* 정보 위주로 구성하세요.*

**[시간 관리 규칙]**
- 만약 {total_days}가 하루라면 Day 1 일정만 적용하세요. 
    - 마지막날로 생각하지 마세요.
- Day 2 ~ Day {total_days} 일정은 무조건 **'오전 10시 시작'**으로 설정하세요.
- 모든 일정은 시간 순서대로 정렬되어야 합니다.

[일차별 시퀀스 정의]
🔴 **Day 1 (첫날)**
   1. 점심 (식당)
   2. 카페
   3. 관광지
   4. 저녁 (식당)
   👉 (총 4곳)

🟠 **Day 2 ~ Day {total_days}-1 (중간 날)**
   1. 관광지 (오전 10시 시작)
   2. 점심 (식당)
   3. 카페
   4. 관광지
   5. 저녁 (식당)
   👉 (총 5곳)

🟢 **Day {total_days} (마지막 날)**
   1. 관광지 (오전 10시 시작)
   👉 (총 1곳 -> 종료!)

[행동 지침]
- 현재 `itinerary`를 확인하고 위 순서에서 **빠진 다음 장소** 하나만 `find_and_select_best_place`로 찾으세요.
"""

editor_prompt = """당신은 '여행 일정 편집자'입니다.
사용자의 수정 요청을 처리하고, 최종 결과를 **가독성 좋게** 보여주세요.

[수정 원칙]
1. **장소 교체:** 사용자가 "A를 B로 바꿔줘"라고 하면:
   - 먼저 `delete_place(place_name="A")`를 호출하여 A를 지우세요.
   - 그 다음 `find_and_select_best_place(query="B")`를 호출하여 B를 추가하세요.
   - 마지막으로 `plan_itinerary_timeline`으로 전체 시간을 재계산하세요.
2. **단순 삭제:** `delete_place` 후 `plan_itinerary_timeline` 호출.

[최종 응답 형식 (Markdown)]
일정이 확정되면 아래 포맷으로 깔끔하게 브리핑하세요.

## 📅 [여행지] 여행 계획표
**Day N**
- 🕙 **10:00 장소명** (카테고리)
  - 💡 *추천 이유 한 줄 요약*
  - 🚌 *다음 장소로 이동: 1003번 버스 (약 30분)*

... (반복)

[다운로드 안내]
"이대로 확정하시겠습니까? 아래 버튼을 눌러 PDF를 받아보세요."
"""

# --- 3. 에이전트 생성 ---
def create_agent(system_prompt):
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("placeholder", "{messages}")])
    llm_with_tools = LLM.bind_tools(TOOLS)
    chain = prompt | llm_with_tools
    
    async def agent_node(state: AgentState):
        filled_prompt = await prompt.ainvoke(state)
        response = await llm_with_tools.ainvoke(filled_prompt)
        return {"messages": [response]}
    return agent_node

PlannerAgent = create_agent(planner_prompt)
EditorAgent = create_agent(editor_prompt)

# --- 4. 라우터 ---
def entry_router(state: AgentState):
    current_stage = state.get("dialog_stage", "planning")
    last_message = state['messages'][-1]
    
    # [수정] 사용자가 '수정', '대신', '바꿔', '삭제' 등을 말하면 편집 모드로 강제 전환
    if isinstance(last_message, HumanMessage):
        content = last_message.content
        # 단순 키워드 매칭 (필요시 더 정교하게 수정 가능)
        edit_keywords = ["대신", "바꿔", "삭제", "변경", "다른", "취소", "빼줘"]
        if any(k in content for k in edit_keywords):
            print(f"DEBUG: 🔄 수정 요청 감지 -> EditorAgent로 전환")
            return "EditorAgent"

    if current_stage == "editing":
        return "EditorAgent"
    
    return "PlannerAgent"

def agent_router(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    
    # 1. 도구 호출 확인
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # 🚨 [Loop Guard v2] 동일한 도구가 '연속'으로 호출될 때만 차단
        # 조건: 메시지 기록이 최소 3개 (AI_1 -> Tool -> AI_2(현재)) 이상이어야 비교 가능
        if len(messages) >= 3:
            prev_tool_msg = messages[-2]
            prev_ai_msg = messages[-3]
            
            # 직전 메시지가 ToolMessage이고, 그 전이 AIMessage인 경우 (전형적인 도구 실행 후 상황)
            if isinstance(prev_tool_msg, ToolMessage) and isinstance(prev_ai_msg, AIMessage):
                current_tools = [t['name'] for t in last_message.tool_calls]
                # 직전 AI가 호출했던 도구 이름들 추출
                prev_tools = [t['name'] for t in prev_ai_msg.tool_calls] if prev_ai_msg.tool_calls else []
                
                target_tools = ["plan_itinerary_timeline", "optimize_and_get_routes"]
                
                for tool in current_tools:
                    # [핵심 수정] 타겟 도구이면서 && '이전에도 똑같이 호출했던 도구'일 때만 차단
                    if tool in target_tools and tool in prev_tools:
                        print(f"DEBUG: 🛑 재귀 루프 감지! ({tool} 연속 호출) -> 강제 종료")
                        return END

        return "call_tools"
        
    # 2. PDF 버튼 활성화 시 종료
    if state.get('show_pdf_button'):
        return END

    # 3. 그 외(일반 대화)는 사용자에게 보여주고 종료
    return END

# src/graph_flow.py 내부함수 교체

async def call_tools_node(state: AgentState):
    last_message = state['messages'][-1]
    new_itinerary = state.get('itinerary', []).copy()
    new_anchor = state.get('current_anchor')
    
    user_info_str = f"모임:{state.get('group_type')}, 스타일:{state.get('style')}, 선호:{state.get('preference')}"
    current_stage = state.get("dialog_stage", "planning")
    show_pdf = state.get("show_pdf_button", False)
    
    tool_calls = last_message.tool_calls
    tool_outputs = []

    # [헬퍼 1] 카테고리 정규화 및 비교
    def get_category_group(type_str):
        t = str(type_str).replace("맛집", "식당").replace("음식점", "식당")
        if any(x in t for x in ["식당", "요리", "레스토랑", "반점", "회관", "고기", "뷔페"]): return "식당"
        if any(x in t for x in ["카페", "커피", "베이커리", "디저트", "찻집"]): return "카페"
        if any(x in t for x in ["관광", "명소", "여행", "공원", "박물관", "미술관", "산책", "전시"]): return "관광지"
        return "기타"

    def is_same_category(type1, type2):
        return get_category_group(type1) == get_category_group(type2)

    # [헬퍼 2] 지능형 일정 정렬 (핵심 로직)
    def reorganize_itinerary(items):
        if not items: return []
        
        # 1. 날짜별로 그룹화
        days = sorted(list(set(item.get('day', 1) for item in items)))
        final_list = []

        for day in days:
            day_items = [x for x in items if x.get('day', 1) == day]
            
            # 카테고리별 분리
            restaurants = [x for x in day_items if get_category_group(x.get('type')) == "식당"]
            cafes = [x for x in day_items if get_category_group(x.get('type')) == "카페"]
            tourists = [x for x in day_items if get_category_group(x.get('type')) == "관광지"]
            others = [x for x in day_items if get_category_group(x.get('type')) == "기타"]

            # 2. 표준 시퀀스대로 재배치: [점심(식당) -> 카페 -> 관광지 -> 저녁(식당)]
            # 식당이 2개 이상이면: 첫 번째를 점심, 나머지를 저녁으로 배치
            # (만약 식당이 1개라면 점심으로 배치)
            
            sorted_day = []
            
            # (1) 점심 (식당 첫 번째)
            if restaurants:
                sorted_day.append(restaurants.pop(0))
            
            # (2) 카페
            sorted_day.extend(cafes)
            
            # (3) 관광지
            sorted_day.extend(tourists)
            
            # (4) 기타 (중간에 끼워넣음)
            sorted_day.extend(others)
            
            # (5) 저녁 (남은 식당들)
            sorted_day.extend(restaurants)

            final_list.extend(sorted_day)
            
        return final_list

    # --- 내부 실행 함수 ---
    async def call_tool_executor(tool_call):
        tool_name = tool_call.get("name")
        args = tool_call.get("args", {})
        
        if tool_name == "find_and_select_best_place":
            args['exclude_places'] = [item['name'] for item in new_itinerary if 'name' in item]
            if not args.get('anchor'): args['anchor'] = new_anchor or state.get('destination')
            args['user_info'] = user_info_str
        elif tool_name == "plan_itinerary_timeline":
            args['itinerary'] = new_itinerary
            
        if tool_name in AVAILABLE_TOOLS:
            try:
                res = await AVAILABLE_TOOLS[tool_name].ainvoke(args)
                raw_output = str(res)
                
                llm_content = raw_output
                if tool_name == "plan_itinerary_timeline":
                    llm_content += "\n\n[SYSTEM INSTRUCTION: 일정 계획 완료. 재호출 금지. 결과 브리핑 요망.]"
                elif tool_name == "optimize_and_get_routes":
                    llm_content += "\n\n[SYSTEM INSTRUCTION: 경로 최적화 완료. 재호출 금지.]"
                
                return ToolMessage(tool_call_id=tool_call['id'], content=llm_content), tool_name, raw_output
            except Exception as e:
                return ToolMessage(tool_call_id=tool_call['id'], content=f"Error: {e}"), tool_name, None
        return None, None, None

    results = await asyncio.gather(*(call_tool_executor(t) for t in tool_calls))

    for tool_message, tool_name, raw_json_output in results:
        if tool_message:
            tool_outputs.append(tool_message)
            
            if raw_json_output:
                if tool_name == "find_and_select_best_place":
                    try:
                        item_json = json.loads(raw_json_output)
                        
                        if item_json.get('name') == "추천 장소 없음":
                            print("DEBUG: ⚠️ 검색 실패 - 일정 추가 안 함")
                        else:
                            # [덮어쓰기 로직]
                            if new_itinerary:
                                last_item = new_itinerary[-1]
                                # 날짜와 카테고리가 같으면 교체 시도
                                if (item_json.get('day', 1) == last_item.get('day', 1) and 
                                    is_same_category(item_json.get('type'), last_item.get('type'))):
                                    
                                    if item_json.get('name') == last_item.get('name'):
                                        print(f"DEBUG: ⏭️ 중복 장소 무시")
                                    else:
                                        print(f"DEBUG: 🔄 '{last_item['name']}' -> '{item_json['name']}' 교체")
                                        new_itinerary.pop()
                                        new_itinerary.append(item_json)
                                        new_anchor = item_json.get('name')
                                        continue 

                            # 일반 추가
                            if not any(x.get('name') == item_json.get('name') for x in new_itinerary):
                                current_places = [i for i in new_itinerary if i.get('type') != 'move']
                                day_to_add = 1
                                if current_places:
                                    day_to_add = current_places[-1].get('day', 1)
                                item_json['day'] = day_to_add
                                new_itinerary.append(item_json)
                                new_anchor = item_json.get('name')
                                
                    except Exception as e: pass

                elif tool_name in ["delete_place", "replace_place"]:
                    try:
                        action_data = json.loads(raw_json_output)
                        target = action_data.get('place_name') or action_data.get('old')
                        if target:
                            new_itinerary = [i for i in new_itinerary if target not in i.get('name', '')]
                    except: pass

                elif tool_name == "plan_itinerary_timeline":
                    try:
                        new_itinerary = json.loads(raw_json_output)
                    except: pass
                
                elif tool_name == "confirm_and_download_pdf":
                    show_pdf = True

    # [핵심] 일정이 뒤죽박죽 섞이지 않도록 마지막에 강제 정렬
    if current_stage == "planning":
        # 초기 생성 시에는 표준 순서(식당->카페->관광지)를 잡아줌
        new_itinerary = reorganize_itinerary(new_itinerary)
    else:
        # 수정 단계에서는 사용자가 추가한 순서를 존중하되, 날짜가 섞이지 않게 'Day' 기준으로만 정렬
        # (이렇게 하면 카페를 저녁 먹고 난 뒤로 보낼 수도 있습니다)
        new_itinerary = sorted(new_itinerary, key=lambda x: x.get('day', 1))

    return {
        "messages": tool_outputs, 
        "itinerary": new_itinerary,
        "show_pdf_button": show_pdf,
        "dialog_stage": current_stage
    }


def route_after_tools(state: AgentState):
    """도구 실행 후 경로 결정"""
    # 1. PDF 완료 시 종료
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
        "PlannerAgent", agent_router, {"call_tools": "call_tools", END: END}
    )
    workflow.add_conditional_edges(
        "EditorAgent", agent_router, {"call_tools": "call_tools", END: END}
    )
    
    workflow.add_conditional_edges(
        "call_tools", route_after_tools,
        {"PlannerAgent": "PlannerAgent", "EditorAgent": "EditorAgent", END: END}
    )
    
    return workflow.compile()