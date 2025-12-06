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
    activity_level: int
    style: str
    preference: str
    
    current_weather: str
    itinerary: List[Dict]
    show_pdf_button: bool 
    current_anchor: str
    
    dialog_stage: str # 'planning' | 'editing'

# --- 2. 프롬프트 ---
planner_prompt = """당신은 '여행 계획 기획자(Planner)'입니다.
전체 여행 기간({total_days}일)의 일정을 채우세요.

[수칙]
1. `find_and_select_best_place`로 장소를 채우세요.
2. 한 날짜가 차면 `plan_itinerary_timeline`으로 시간을 계산하세요.
3. **[LOOP 방지]** 만약 직전 메시지가 **'TIMELINE_CALCULATED'**라면, 당신의 다음 행동은 **반드시** `find_and_select_best_place`를 호출하여 새로운 장소를 찾는 것이어야 합니다. 타임라인 도구를 연속으로 호출하지 마세요!
4. 모든 날짜가 채워지기 전까지는 멈추지 마세요.
"""

editor_prompt = """당신은 '여행 계획 편집자(Editor)'입니다.
사용자의 요청에 따라 일정을 수정합니다.

[수칙]
1. 사용자가 "OO를 XX로 바꿔줘"라고 하면 `find_and_select_best_place` 등을 사용해 해당 장소를 추가/교체하세요.
2. 장소 변경이 완료되면, **즉시 `plan_itinerary_timeline`을 호출하여 전체 일정을 갱신**하세요.
3. 다른 말은 하지 말고 도구 호출에만 집중하세요.
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
                            # 날짜 할당 로직: 가장 마지막 날짜 혹은 다음 날짜로 할당
                            current_places = [item for item in new_itinerary if item.get('type') != 'move']
                            last_day = max(item.get('day', 1) for item in current_places) if current_places else 1
                            count_on_last_day = sum(1 for x in current_places if x.get('day') == last_day)
                            
                            # 활동량(activity_level)을 초과하면 다음 날짜로 할당
                            if count_on_last_day >= state.get('activity_level', 3) and last_day < total_days:
                                item_json['day'] = last_day + 1
                            else:
                                item_json['day'] = last_day
                                
                            new_itinerary.append(item_json)
                            new_anchor = item_json.get('name')
                            print(f"DEBUG: 장소 추가됨: {new_anchor} (Day {item_json['day']})")
                    except: pass


                # 2. 타임라인 생성 (plan_itinerary_timeline)
                elif tool_name == "plan_itinerary_timeline":
                    try:
                        new_itinerary = json.loads(output) # 상세 정보(이동시간 등) 업데이트
                        
                        # [복원] 전체 N일차 계획이 모두 완성되었는지 확인
                        is_plan_complete = True
                        day_counts = {}
                        for item in new_itinerary:
                            if item.get('type') != 'move':
                                day = item.get('day')
                                if day:
                                    day_counts[day] = day_counts.get(day, 0) + 1
                        
                        for day_num in range(1, total_days + 1):
                            if day_counts.get(day_num, 0) < state.get('activity_level', 3):
                                is_plan_complete = False
                                break
                        
                        # [복원] 계획이 아직 미완성인 경우, Planner로 복귀
                        if not is_plan_complete:
                            print(f"DEBUG: 📅 Plan not yet complete. Returning to Planner agent.")
                            # [수정] Planner의 루프 방지 프롬프트를 위한 신호 메시지 추가
                            tool_outputs.append(HumanMessage(content="TIMELINE_CALCULATED"))
                        
                        # [복원] 계획이 완성된 경우, 요약본 생성 및 Editor 모드 전환
                        else:
                            print(f"DEBUG: 🎉 Plan complete! Switching to editing mode and showing summary.")
                            current_stage = "editing" # Switch stage
                            
                            # 사용자에게 보여줄 최종 요약본 생성
                            summary = "🚗 **여행 계획 초안이 완성되었습니다.**\n내용을 확인하시고, 수정이 필요하면 알려주세요.\n\n"
                            
                            current_day = 0
                            sorted_itinerary = sorted(new_itinerary, key=lambda x: (int(x.get('day', 1)), x.get('start', '00:00')))

                            for item in sorted_itinerary:
                                item_day = item.get('day', 0)
                                if item_day != current_day:
                                    summary += f"\n**🗓️ Day {item_day}**\n"
                                    current_day = item_day
                                
                                item_type = item.get('type', 'activity')
                                
                                if item_type == 'move':
                                    dur_text = item.get('duration_text', '이동')
                                    summary += f"   ⬇️ *{dur_text}*\n"
                                else:
                                    time_str = f"[{item.get('start')}] " if item.get('start') else ""
                                    name = item.get('name', '이름 없음')
                                    desc = item.get('description', '')
                                    summary += f"   📍 **{time_str}{name}**\n"
                                    if desc:
                                        summary += f"      └ 💡 {desc}\n"

                            summary += "\n\n**이대로 확정하고 PDF를 다운로드할까요? 아니면 수정할까요?**"
                            
                            # 요약 AIMessage를 추가하여 그래프가 종료되도록 함
                            tool_outputs.append(AIMessage(content=summary))
                            
                    except Exception as e: 
                        print(f"DEBUG: Timeline JSON 파싱 실패: {e}")
                        pass

                # 3. PDF 확정
                elif tool_name == "confirm_and_download_pdf":
                    show_pdf = True
                    tool_outputs.append(AIMessage(content="✅ **확정되었습니다!** 아래 버튼을 눌러주세요."))

    # ---------------------------------------------------------
    # 4. 최종 리턴
    # ---------------------------------------------------------
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