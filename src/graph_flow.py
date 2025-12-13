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
- **마지막 날(Day {total_days})**에는 **'관광지' 딱 1곳**만 찾으면 끝입니다.
- 식당, 카페, 저녁 일정을 절대 추가하지 마세요.
- **마지막 날 관광지 1곳이 확보되면**, 즉시 `plan_itinerary_timeline`을 호출하여 종료하세요.

✅ **[최종 결과물 필수 요구사항]**
일정을 확정할 때(`plan_itinerary_timeline` 결과) 다음 3가지 요소에 집중하세요:
1. **각 일정의 대략적인 시간** (예: 10:00 ~ 11:30)
2. **장소 간 이동 시간** (예: 약 30분 소요)
3. **상세 교통편 정보** (예: 1003번 버스 ➡️ 도보)
*장소에 대한 긴 설명이나 미사여구는 줄이고, 위 '시간'과 '이동' 정보 위주로 구성하세요.*

**[시간 관리 규칙]**
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
    if state.get("dialog_stage") == "editing":
        return "EditorAgent"
    return "PlannerAgent"

def agent_router(state: AgentState):
    last_message = state['messages'][-1]
    # 도구 호출 시 도구 노드로
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "call_tools"
    # PDF 버튼 활성화 시 종료
    if state.get('show_pdf_button'):
        return END
    # 그 외(일반 대화)는 사용자에게 보여주고 종료
    return END

# --- 5. 도구 실행 노드 ---
# src/graph_flow.py (수정된 call_tools_node 전체)

async def call_tools_node(state: AgentState):
    last_message = state['messages'][-1]
    new_itinerary = state.get('itinerary', []).copy()
    new_anchor = state.get('current_anchor')
    weather_update = state.get('current_weather')
    
    # [중요] 사용자 정보 스트링 생성
    user_info_str = f"모임:{state.get('group_type')}, 스타일:{state.get('style')}, 선호:{state.get('preference')}"

    # 상태 변수
    total_days = state.get('total_days', 1)
    current_stage = state.get("dialog_stage", "planning")
    show_pdf = state.get("show_pdf_button", False)
    
    # 타겟 데이 계산 (장소 할당 로직을 위한 준비)
    current_itinerary_places = [item for item in new_itinerary if item.get('type') != 'move']
    planned_days = set(item.get('day') for item in current_itinerary_places)
    
    tool_calls = last_message.tool_calls
    tool_outputs = []

    # ---------------------------------------------------------
    # [수정] 1. 도구 호출 함수 (결과만 반환)
    # ---------------------------------------------------------
    async def call_tool_executor(tool_call):
        tool_name = tool_call.get("name")
        
        # Args 주입은 여기서 한 번만 처리
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
                return ToolMessage(tool_call_id=tool_call['id'], content=str(res)), tool_name, str(res)
            except Exception as e:
                return ToolMessage(tool_call_id=tool_call['id'], content=f"Error: {e}"), tool_name, None
        return None, None, None

    # ---------------------------------------------------------
    # 2. 병렬 실행
    # ---------------------------------------------------------
    results = await asyncio.gather(*(call_tool_executor(t) for t in tool_calls))

    # ---------------------------------------------------------
    # 3. 결과 처리 루프 (여기서 로직 분기)
    # ---------------------------------------------------------
    for tool_message, tool_name, output in results:
        if tool_message:
            tool_outputs.append(tool_message)
            
            if output:
                # 1. 장소 추가 (find_and_select_best_place)
                if tool_name == "find_and_select_best_place":
                    try:
                        item_json = json.loads(output)
                        if not any(x.get('name') == item_json.get('name') for x in new_itinerary):
                            # 🚨 [수정] 고정 스케줄 기반 day 할당 로직
                            # Day 1: 4곳 (점심, 카페, 관광지, 저녁)
                            # Day 2~N-1: 5곳씩 (관광지, 점심, 카페, 관광지, 저녁)
                            # Day N: 1곳 (관광지)
                            current_places = [i for i in new_itinerary if i.get('type') != 'move']
                            place_count = len(current_places)
                            total_days = state.get('total_days', 1)

                            if place_count < 4:
                                # Day 1: 처음 4곳
                                item_json['day'] = 1
                            else:
                                # Day 2부터: (전체 - Day1의 4곳) ÷ 5 + 2
                                remaining = place_count - 4
                                calculated_day = 2 + (remaining // 5)
                                # total_days 초과 방지
                                item_json['day'] = min(calculated_day, total_days)

                            print(f"DEBUG: 장소 추가 - {item_json.get('name')} → Day {item_json['day']} (현재 {place_count+1}번째 장소)")

                            new_itinerary.append(item_json)
                            new_anchor = item_json.get('name')
                    except: pass

                # 2. [신규] 장소 삭제/교체 처리
                elif tool_name == "delete_place" or tool_name == "replace_place":
                    try:
                        action_data = json.loads(output)
                        target_name = action_data.get('place_name') or action_data.get('old')
                        if target_name:
                            # 이름이 포함된 장소를 찾아서 제거
                            initial_len = len(new_itinerary)
                            new_itinerary = [
                                item for item in new_itinerary 
                                if target_name not in item.get('name', '')
                            ]
                            if len(new_itinerary) < initial_len:
                                print(f"DEBUG: '{target_name}' 삭제 완료.")
                            
                    except Exception as e:
                        print(f"DEBUG: 삭제 처리 중 오류: {e}")

                # 3. 타임라인 재계산 (기존 로직 유지)
                elif tool_name == "plan_itinerary_timeline":
                    try:
                        new_itinerary = json.loads(output)
                        # 여기서 요약본을 생성하지 않고, EditorAgent가 직접 예쁘게 말하도록 유도
                        # tool_outputs에 데이터만 담아두면 됨
                    except: pass
                
                # 4. PDF 확정
                elif tool_name == "confirm_and_download_pdf":
                    show_pdf = True

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