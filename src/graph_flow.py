# src/graph_flow.py

from typing import TypedDict, Annotated, List, Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from src.config import LLM
from src.tools import AVAILABLE_TOOLS, TOOLS
import re 
import json

# --- 헬퍼 함수 ---
def normalize_content_to_str(content: Any) -> str:
    if content is None: return ""
    if isinstance(content, list):
        return "\n".join([str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content])
    return str(content)

def clean_json_text(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(json)?\s*(.*)\s*```", text, re.DOTALL)
    if match: return match.group(2).strip()
    return text

# --- 1. 상태 정의 ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    destination: str
    dates: str
    total_days: int
    activity_level: int
    preference: str
    current_weather: str
    itinerary: List[Dict]
    current_planning_day: int
    show_pdf_button: bool 
    current_anchor: str 

# --- 2. 에이전트 생성 팩토리 ---
def create_specialist_agent(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("placeholder", "{messages}")])
    llm_with_tools = LLM.bind_tools(TOOLS)
    chain = prompt | llm_with_tools
    
    def agent_node(state: AgentState):
        # 1. 진행 상황 계산
        total_days = state.get('total_days', 1)
        activity_level = state.get('activity_level', 3)
        itinerary = state.get('itinerary', [])
        
        day_counts = {d: 0 for d in range(1, total_days + 1)}
        for item in itinerary:
            if item.get('type') != 'move':
                day = item.get('day')
                if day and isinstance(day, int) and day in day_counts:
                    day_counts[day] += 1

        target_day = 1
        all_finished = True
        progress_report = []
        for d in range(1, total_days + 1):
            count = day_counts[d]
            status = "완료" if count >= activity_level else f"진행 중 ({count}/{activity_level})"
            progress_report.append(f"- {d}일차: {count}/{activity_level}곳 ({status})")
            if count < activity_level and all_finished:
                all_finished = False
                target_day = d
        
        # 2. 상태 메시지 생성
        weather_info = state.get('current_weather')
        if not weather_info:
            goal_msg = "【긴급】 날씨 정보가 없습니다! 가장 먼저 `get_weather_forecast` 도구를 사용하세요."
        elif all_finished:
            goal_msg = "【모든 일정 완료!】 `plan_itinerary_timeline`을 호출하여 최종 일정을 정리하세요."
        else:
            goal_msg = f"【현재 목표: {target_day}일차 계획 수립】 다음 장소를 추천해주세요."

        itinerary_summary = [f"- {item.get('type', '장소')}: {item.get('name', '이름모름')}" for item in itinerary]

        state_summary = f"""
--- [시스템 현황판 (최신)] ---
1. 여행지: {state.get('destination')} ({state.get('dates')})
2. 날씨 정보: {weather_info if weather_info else "❌ 없음 (즉시 조회 필요)"}
3. 진행 상황:
{chr(10).join(progress_report)}
4. 현재까지의 일정:
{chr(10).join(itinerary_summary)}
5. 현재 상태: {goal_msg}
6. 현재 앵커: {state.get('current_anchor', '출발지')}
-----------------------------
"""
        current_messages = state['messages'] + [HumanMessage(content=state_summary)]
        
        response = chain.invoke({"messages": current_messages})
        
        # 🚨 [수정] agent_node는 더 이상 상태를 직접 수정하지 않음. 도구 호출에만 집중.
        return {"messages": [response]}
    
    return agent_node

# --- 3. 프롬프트 (최종 수정) ---
supervisor_prompt = """당신은 주어진 현황판을 분석하여 다음 행동을 결정하는 '지능형 여행 계획 슈퍼바이저'입니다.

### 🚀 실행 절차

**1. [우선순위 1] 날씨 확인:**
- '현황판'에 **날씨 정보가 없다면**, 다른 어떤 작업보다 먼저 `get_weather_forecast`를 호출하여 날씨 정보를 가져오세요.

**2. [우선순위 2] 계획 수립:**
- '현황판'의 '진행 상황'을 보고, 아직 **목표치를 채우지 못한 날**이 있는지 확인하세요.
- **만약 그런 날이 있다면, 해당 날짜(N일차)와 현재 채워진 일정 수에 따라 다음 규칙으로 `find_and_select_best_place`를 단 한 번 호출하세요:**

    **A. 1일차인 경우 (시작 시간 12:00 가정):**
    - **첫 번째 장소 (점심):** 무조건 **'맛집'**을 검색하세요.
    - **두 번째 장소:** 사용자의 선호가 '맛집 탐방'이라면 **'카페'**를, 아니라면 **'관광지'**를 검색하세요.
    - **세 번째 장소:**
        - 만약 이 날의 목표 일정 수가 4개 이상이라면: **'관광지'**를 검색하세요.
        - 그 외의 경우: 바로 저녁 식사를 위해 **'맛집'**을 검색하세요.
    - **네 번째 장소 (저녁):** 세 번째 장소에서 '관광지'를 갔다면, 이번에는 **'맛집'**을 검색하세요.
    - **다섯 번째 이후 (남은 일정):** 아직 목표치를 못 채웠다면 **'관광지'**를 검색하세요.

    **B. 중간 날짜 (1일차 아님 & 마지막 날 아님):**
    - **첫 번째 장소 (오전):** 무조건 **'관광지'**를 먼저 하나 검색하세요.
    - **두 번째 장소부터:** 1일차의 로직(점심 맛집 -> 선호에 따른 2번째 장소 -> ...)을 동일하게 따르세요.

    **C. 마지막 날인 경우:**
    - 활동량이나 다른 조건에 상관없이, 앵커(숙소 또는 거점) 근처의 **'맛집'**을 하나 검색하여 일정을 마무리하세요.

**3. [우선순위 3] 검색 실패 시 대처:**
- 만약 `find_and_select_best_place` 도구 호출 결과가 "더 이상 추천할 새로운 장소가 없습니다"와 같은 실패 메시지라면, **같은 종류의 장소를 다시 검색하지 마세요.**
- 대신, **다른 종류의 장소를 검색**하세요. (예: '관광지' 검색 실패 시 '카페' 또는 '공원' 검색)
- 여러 종류를 시도해도 계속 장소를 찾지 못하면, 그 날의 계획을 중단하고 `plan_itinerary_timeline` 도구를 호출하여 현재까지의 일정으로 최종 정리를 시작하세요.

**4. [우선순위 4] 최종 정리:**
- '진행 상황'의 **모든 날짜가 목표를 달성했다면**, `plan_itinerary_timeline` 도구를 호출하여 전체 일정을 시간순으로 정리하고 최종 결과를 만드세요.
"""
SupervisorAgent = create_specialist_agent(supervisor_prompt)

# --- 4. 라우터 ---
def supervisor_router(state: AgentState):
    return "SupervisorAgent"

def supervisor_loop_router(state: AgentState):
    """
    [수정] 에이전트의 루프를 제어하는 핵심 라우터.
    - 도구 호출이 필요하면 -> call_tools
    - 모든 일정이 완료되었고, PDF 버튼이 활성화(최종 정리 완료)되었다면 -> END
    - 그 외의 경우, 계속 계획을 수립하기 위해 -> SupervisorAgent
    """
    # 1. 도구 호출이 있으면 -> 도구 실행
    if hasattr(state['messages'][-1], 'tool_calls') and state['messages'][-1].tool_calls:
        return "call_tools"
    
    # 2. 모든 계획이 완료되었는지 확인
    total_days = state.get('total_days', 1)
    activity_level = state.get('activity_level', 3)
    itinerary = state.get('itinerary', [])
    
    # 'move'를 제외한 실제 활동 수 계산
    day_counts = {d: sum(1 for item in itinerary if item.get('day') == d and item.get('type') != 'move') for d in range(1, total_days + 1)}
    
    is_plan_complete = all(count >= activity_level for count in day_counts.values())

    # 3. 종료 조건: 계획이 완료되었고, 최종 정리(PDF 버튼 활성화)까지 끝났다면 종료
    if is_plan_complete and state.get('show_pdf_button'):
        print("DEBUG: 모든 계획 및 최종 정리가 완료되어 워크플로우를 종료합니다.")
        return END

    # 4. 그 외의 경우: 계속해서 Supervisor에게 제어권 전달
    return "SupervisorAgent"


# --- 5. 도구 노드 & 그래프 ---
def call_tools_node(state: AgentState):
    last_message = state['messages'][-1]
    results = []
    
    # 수정 가능한 상태 복사본
    new_itinerary = state.get('itinerary', []).copy()
    new_anchor = state.get('current_anchor')
    weather_update = state.get('current_weather')
    show_pdf = state.get('show_pdf_button', False)

    # 현재 계획 중인 날짜를 가져옴
    total_days = state.get('total_days', 1)
    activity_level = state.get('activity_level', 3)
    target_day = 1
    for d in range(1, total_days + 1):
        count = sum(1 for item in new_itinerary if item.get('day') == d and item.get('type') != 'move')
        if count < activity_level:
            target_day = d
            break

    for t in last_message.tool_calls:
        tool_name = t.get("name")
        if tool_name in AVAILABLE_TOOLS:
            try:
                args = t.get("args", {})
                
                if tool_name == "find_and_select_best_place":
                    args['exclude_places'] = [item['name'] for item in new_itinerary if 'name' in item]
                    if not args.get('anchor'):
                        args['anchor'] = new_anchor or state.get('destination')
                
                elif tool_name == "plan_itinerary_timeline":
                    args['itinerary'] = new_itinerary

                res = AVAILABLE_TOOLS[tool_name].invoke(args)
                output = str(res)
                
                # [수정] 도구 결과에 따른 상태 업데이트 로직
                if tool_name == "find_and_select_best_place":
                    try:
                        item_json = json.loads(output)
                        if not any(x.get('name') == item_json.get('name') for x in new_itinerary):
                            # 타입 추론 추가
                            if 'type' not in item_json:
                                if any(kw in item_json.get('description', '') for kw in ['맛집', '식당', '카페']):
                                     item_json['type'] = '맛집'
                                else:
                                     item_json['type'] = '관광지'
                            
                            item_json['day'] = target_day # 올바른 목표일차 설정
                            new_itinerary.append(item_json)
                            new_anchor = item_json.get('name')
                            print(f"DEBUG: [ADD BY TOOL] {new_anchor} to Day {target_day}")
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"DEBUG: [TOOL ERROR] find_and_select_best_place 결과 처리 실패: {e}, 원본: {output}")

                elif tool_name == "plan_itinerary_timeline":
                    try:
                        new_itinerary = json.loads(output)
                        show_pdf = True 
                        print("DEBUG: [FINAL] 최종 타임라인 생성 완료")
                    except json.JSONDecodeError:
                        print(f"DEBUG: [TOOL ERROR] plan_itinerary_timeline 결과가 JSON이 아님: {output}")

                elif tool_name == 'get_weather_forecast':
                    weather_update = output
                    
                results.append(ToolMessage(tool_call_id=t['id'], content=output))
            except Exception as e:
                print(f"ERROR in tool {tool_name}: {e}")
                results.append(ToolMessage(tool_call_id=t['id'], content=f"Error: {e}"))
    
    return {
        "messages": results, 
        "itinerary": new_itinerary,
        "current_anchor": new_anchor,
        "current_weather": weather_update,
        "show_pdf_button": show_pdf,
    }

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("SupervisorAgent", SupervisorAgent)
    workflow.add_node("call_tools", call_tools_node)
    workflow.set_entry_point("SupervisorAgent")
    workflow.add_conditional_edges(
        "SupervisorAgent",
        supervisor_loop_router,
        {"call_tools": "call_tools", END: END, "SupervisorAgent": "SupervisorAgent"}
    )
    workflow.add_edge("call_tools", "SupervisorAgent")
    return workflow.compile()