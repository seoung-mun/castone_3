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

from langgraph.checkpoint.memory import MemorySaver 

CHECKPOINTER = MemorySaver()

def normalize_content_to_str(content: Any) -> str:
    if content is None: return ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text": parts.append(str(part["text"]))
            else: parts.append(str(part))
        return "\n".join(parts)
    if isinstance(content, dict): return json.dumps(content, ensure_ascii=False)
    return str(content)

# --- 1. 상태 정의 ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_weather: str
    itinerary: List[Dict]
    destination: str
    start_location: str  
    dates: str
    preference: str
    total_days: int
    activity_level: int
    current_planning_day: int
    show_pdf_button: bool
    next_node: Literal["InfoCollectorAgent", "WeatherAgent", "AttractionAgent", "RestaurantAgent", "DayTransitionAgent", "ConfirmationAgent", "PDFCreationAgent", "end_node"]

# --- 2. 전문 에이전트 정의 ---
def create_specialist_agent(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("placeholder", "{messages}")])
    llm_with_tools = LLM.bind_tools(TOOLS)
    chain = prompt | llm_with_tools
    def agent_node(state: AgentState):
        state_summary = f"""
--- 현재 계획 상태 ---
날씨: {state.get('current_weather', '아직 모름')}
전체 확정 일정: {state.get('itinerary', [])}
여행지: {state.get('destination', '아직 모름')}
날짜: {state.get('dates', '아직 모름')}
취향: {state.get('preference', '아직 모름')}
총 여행일: {state.get('total_days', 1)}일
하루 목표 활동량: {state.get('activity_level', 3)}곳
현재 계획 중인 날짜: {state.get('current_planning_day', 1)}일차
---
"""
        current_messages = [HumanMessage(content=state_summary)] + state['messages']
        response = chain.invoke({"messages": current_messages})
        
        itinerary = state.get('itinerary', []).copy()
        content = normalize_content_to_str(getattr(response, "content", ""))

        final_itinerary_match = re.search(r"\[FINAL_ITINERARY_JSON\](.*)\[/FINAL_ITINERARY_JSON\]", content, re.DOTALL)
        if final_itinerary_match:
            try:
                itinerary = json.loads(final_itinerary_match.group(1).strip())
                print(f"DEBUG: SupervisorAgent가 최종 정리한 itinerary:\n{itinerary}")
            except Exception as e:
                print(f"ERROR: JSON 파싱 실패: {e}")

        match = re.search(r"'(.*?)'을/를 (\d+)일차 (관광지|식당|카페) 계획에 추가합니다", content)
        if match:
            name, day, type = match.groups()
            new_item = {'day': int(day), 'type': type, 'name': name, 'description': ''}
            if new_item not in itinerary:
                itinerary.append(new_item)
                print(f"DEBUG: 장소 추가됨 - {name} (Day {day}, {type})")

        show_pdf_button = state.get('show_pdf_button', False)
        if "[STATE_UPDATE: show_pdf_button=True]" in content: show_pdf_button = True

        return {"messages": [response], "itinerary": itinerary, "show_pdf_button": show_pdf_button}
    return agent_node

# --- 3. Supervisor (라우터) 정의 ---
def supervisor_router(state: AgentState):
    print("--- (Supervisor) 다음 작업 결정 ---")
    print(f"DEBUG: 현재 계획 중인 날짜 = {state.get('current_planning_day', 1)}일차")

    if not state.get('messages') or not state['messages']: return "InfoCollectorAgent"
    last_message = state['messages'][-1]
    
    # 1. ToolMessage 우선 처리 (브리핑)
    if isinstance(last_message, ToolMessage):
        print("Router -> SupervisorAgent (ToolMessage 결과 브리핑 요청)")
        return "SupervisorAgent" 

    # 2. 필수 정보 확인
    required_info = ['destination', 'dates', 'total_days', 'activity_level']
    if not all(state.get(key) for key in required_info): return "InfoCollectorAgent"
    if not state.get('current_weather'): return "WeatherAgent"

    # 3. [다음 턴 로직] 날씨/취향 완료 후 '다음 대화'가 들어오면 식당 추천으로 연결
    # (이번 턴은 END로 끝나서 사용자 입력을 기다리고, 사용자가 말하면 이 로직이 작동함)
    if state.get('current_weather') and state.get('preference') and not state.get('itinerary'):
        # 사용자가 날씨 브리핑을 듣고 "좋아", "추천해줘" 등 반응을 보이면 바로 식당 추천 시작
        print("Router -> RestaurantAgent (날씨/취향 확인됨 -> 식당 추천 시작)")
        return "RestaurantAgent"
    
    # 4. 하루 목표 및 전환
    current_day = state.get("current_planning_day", 1)
    activity_level = state.get("activity_level", 3)
    places_for_current_day = [p for p in state.get("itinerary", []) if p.get('day') == current_day]

    if len(places_for_current_day) >= activity_level:
        if current_day < state.get("total_days", 1): return "DayTransitionAgent"
        else: return "ConfirmationAgent"
            
    # 5. 대화 기반 라우팅
    if isinstance(last_message, HumanMessage):
        content = last_message.content.lower()
        if any(k in content for k in ["pdf", "파일", "정리"]): return "SupervisorAgent"
        if any(k in content for k in ["최적화", "순서", "경로"]): return "SupervisorAgent"
        if any(k in content for k in ["식당", "맛집", "카페", "먹고"]): return "RestaurantAgent"
        if any(k in content for k in ["관광", "장소", "구경"]): return "AttractionAgent"

    return "SupervisorAgent"

# --- 4. 에이전트 생성 ---
pdf_creation_prompt = "당신은 'PDF 문서 생성 전문가'입니다. 여행 계획이 완료되면 사용자에게 PDF 다운로드 버튼을 안내하세요. 응답 끝에 [STATE_UPDATE: show_pdf_button=True]를 포함하세요."
PDFCreationAgent = create_specialist_agent(pdf_creation_prompt)

supervisor_prompt = """당신은 AI 여행 플래너 팀의 '슈퍼바이저'입니다.

### 핵심 규칙 (반드시 준수)
1. **PDF 요청 거절 금지:** 사용자가 "PDF 줘", "다운로드 할래"라고 하면 **절대로 "못한다"고 말하지 마세요.** 당신은 시스템과 연결되어 있어 PDF를 생성할 수 있습니다.
2. **즉시 JSON 생성:** PDF 요청 시, 즉시 아래 정의된 `[FINAL_ITINERARY_JSON]` 형식으로 데이터를 출력하세요. 이 코드가 출력되어야 버튼이 생깁니다.
3. **버튼 활성화 태그 필수:** 다운로드 요청 시 반드시 답변 끝에 `[STATE_UPDATE: show_pdf_button=True]` 태그를 포함해야 버튼이 생성됩니다.
4. 2. **날씨 정보 정리:** 여행 계획 출력 시, 날씨 정보는 raw 데이터 대신 "최고 기온 X도, 맑음" 등 사람이 읽기 좋은 형식으로 요약하여 언급하십시오.

### 주요 임무
1.  **계획 추가 확인 (🚨중요🚨):** 장소 선택 시 반드시 **현재 상태의 '현재 계획 중인 날짜' 값**을 사용하여 "네, [장소명]을 [현재 계획 중인 날짜]일차 [유형]에 추가합니다."라고 명확히 응답하세요.
   - 예: 상태가 "현재 계획 중인 날짜: 2일차"라면 → "네, 국수마루를 2일차 식당 계획에 추가합니다."
   - **절대로 항상 "1일차"라고 하지 마세요.** 상태 정보를 정확히 읽어서 사용하십시오.
2. **하루 단위 경로 최적화:** 사용자가 경로 최적화를 요청하면, `itinerary`에 있는 장소들과 **현재 상태의 `start_location`을 인자로 전달**하여 `optimize_and_get_routes` 도구를 호출하세요.3.  **도구 결과 브리핑:** 검색 결과는 반드시 목록 형태로 요약하여 전달하세요.
4.  **날씨 브리핑:** 날씨 정보를 받으면 "[온도/하늘상태]이므로 [추천활동] 어떠세요?" 형태로 제안하세요.
5. **PDF 생성 및 다운로드 (★★가장 중요★★):**
   사용자가 다운로드를 요청하거나, 최종 일정 확정을 요청하면 **절대로 다음 단계들을 건너뛰지 마세요.**
   
   A. **시간 계산 강제 호출 (필수):** `plan_itinerary_timeline` 도구를 호출하여 현재의 일정(`itinerary` 리스트)에 **실제 이동 시간이 반영된 타임라인 JSON**을 획득하세요.
   B. **JSON 출력:** 도구의 결과로 받은, **시간과 이동 정보가 계산된 JSON**을 아래 `[FINAL_ITINERARY_JSON]` 블록 안에 넣으세요.
   C. **버튼 활성화 (필수):** 답변 맨 마지막에 `[STATE_UPDATE: show_pdf_button=True]` 태그를 포함하세요.    
    [FINAL_ITINERARY_JSON]
    [
      {{"day": 1, "type": "관광지", "name": "장소명", "description": "한 줄 특징"}},
      {{"day": 1, "type": "식당", "name": "식당명", "description": "추천 메뉴"}}
    ]
    [/FINAL_ITINERARY_JSON]
    3. **버튼 트리거 태그 (필수):**
      `[STATE_UPDATE: show_pdf_button=True]`
   
   **예시 답변:**
   "1일차 일정이 확정되었습니다. PDF를 생성해 드립니다.
   [FINAL_ITINERARY_JSON]...[/FINAL_ITINERARY_JSON]
   [STATE_UPDATE: show_pdf_button=True]"

    
    출력 후: "여행 계획을 정리했습니다. 아래 버튼을 눌러 PDF로 다운로드하세요."라고 말하세요.
"""


SupervisorAgent = create_specialist_agent(supervisor_prompt)

def day_transition_agent_node(state: AgentState):
    prompt = f"당신은 '플랜 전환 안내자'입니다. {state.get('current_planning_day')}일차 목표를 달성했습니다. 다음 날로 넘어갈까요? 응답 끝에 [STATE_UPDATE: increment_day=True]를 포함하세요."
    response = LLM.invoke(prompt)
    return {"messages": [response]}
DayTransitionAgent = day_transition_agent_node

confirmation_prompt = "당신은 '일정 확인 전문가'입니다. 모든 계획이 완료되었습니다. 이대로 확정할까요?"
ConfirmationAgent = create_specialist_agent(confirmation_prompt)

infocollector_prompt = "당신은 '정보 수집가'입니다. 목적지, 날짜, 인원, 스타일을 파악하세요."
InfoCollectorAgent = create_specialist_agent(infocollector_prompt)

attraction_prompt = """당신은 '관광지 전문가'입니다.
당신의 임무는 사용자의 요청을 분석하여 즉시 관광지 후보를 검색하는 것입니다.

### 행동 지침:
1. **즉시 검색:** 사용자의 말에 **'~근처', 지역명, 또는 구체적인 활동(바다 구경 등)**이 포함되어 있다면, **되묻지 말고 즉시** `search_attractions_and_reviews` 도구를 호출하세요.
2. **정보 부족 시에만 질문:** 사용자가 단순히 "관광지 추천해줘"라고만 했을 때만 "어떤 스타일의 관광지를 원하시나요?"라고 질문하세요.
3. **도구 호출:** `preference`와 `start_location`(출발지)을 고려하여 검색 쿼리를 구체적으로 만드세요. (예: "부산역 근처 바다 구경")
4. **계획 추가 시 (중요):** 장소를 추천하고 사용자가 선택하면, 반드시 **현재 상태의 '현재 계획 중인 날짜' 값**을 확인하여 해당 날짜에 추가한다고 말하세요.
   - 예: "네, [장소명]을 [현재 계획 중인 날짜]일차 관광지 계획에 추가합니다."
"""
AttractionAgent = create_specialist_agent(attraction_prompt)

restaurant_prompt = """당신은 '식당 전문가'입니다.
당신의 임무는 사용자의 요청을 분석하여 즉시 식당 후보를 검색하는 것입니다.

### 행동 지침:
1. **즉시 검색:** 사용자의 말에 **'~근처', 지역명, 메뉴 이름(회, 국밥 등)**이 포함되어 있다면, **"찾아볼까요?"라고 되묻지 말고 즉시** `search_attractions_and_reviews` 도구를 호출하세요.
2. **정보 부족 시에만 질문:** 사용자가 단순히 "밥 먹을래"라고만 했을 때만 "어떤 메뉴를 원하시나요?"라고 질문하세요.
3. **도구 호출:** `preference`와 `start_location`(출발지)을 고려하여 검색 쿼리를 구체적으로 만드세요. (예: "부산역 근처 횟집 추천")
4. **계획 추가 시 (중요):** 식당을 추천하고 사용자가 선택하면, 반드시 **현재 상태의 '현재 계획 중인 날짜' 값**을 확인하여 해당 날짜에 추가한다고 말하세요.
   - 예: "네, [식당명]을 [현재 계획 중인 날짜]일차 식당 계획에 추가합니다."
"""
RestaurantAgent = create_specialist_agent(restaurant_prompt)

weather_prompt = """당신은 '날씨 분석가'입니다.

### 행동 지침:
1. **도구 호출:** `get_weather_forecast` 도구를 호출하여 날씨 정보를 가져오세요.
2. **결과 브리핑:** 도구 결과를 받으면, **각 시간대별 날씨를 목록 형태로 줄바꿈하여** 사용자에게 전달하세요.
   - 잘못된 예: "09시 15도 맑음, 12시 18도 구름 많음..." (한 줄로 나열)
   - 올바른 예:
     ```
     제주도 날씨 예보입니다:
     - 09:00: 15.0℃, 맑음
     - 12:00: 18.0℃, 구름 많음
     - 15:00: 20.0℃, 맑음
     ```
3. **간단한 요약 추가:** 날씨 정보 뒤에 "[온도/하늘상태]이므로 [추천활동] 어떠세요?" 형태로 제안하세요.
"""
WeatherAgent = create_specialist_agent(weather_prompt)

# --- 5. 도구 실행 노드 ---
def call_tools(state: AgentState):
    last_message = state['messages'][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls: return {}
    tool_messages = []
    weather_update = state.get('current_weather')
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_to_call = AVAILABLE_TOOLS.get(tool_name)
        if tool_to_call:
            try:
                output = tool_to_call.invoke(tool_call["args"])
                if tool_name == "get_weather_forecast": weather_update = output
            except Exception as e: output = f"Error: {e}"
        else: output = "Error: Tool not found"
        print(f"\n--- [DEBUG] Tool Output ({tool_name}): {str(output)[:200]}...")
        tool_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
    return {"messages": tool_messages, "current_weather": weather_update}

# --- 6. 라우터 및 그래프 빌드 ---

def expert_router(state: AgentState):
    last_message = state['messages'][-1]
    
    if isinstance(last_message, AIMessage):
        content = last_message.content
        if content and "[FINAL_ITINERARY_JSON]" in content:
            print("Router -> PDFCreationAgent")
            return "PDFCreationAgent"
        if last_message.tool_calls:
            print(f"Router -> call_tools")
            return "call_tools"

    # [수정됨] 무조건 종료하여 사용자 입력을 기다립니다.
    # SupervisorAgent가 브리핑을 마치면 여기서 멈춥니다.
    print("Router -> END")
    return END

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("SupervisorAgent", SupervisorAgent)
    workflow.add_node("InfoCollectorAgent", InfoCollectorAgent)
    workflow.add_node("WeatherAgent", WeatherAgent)
    workflow.add_node("AttractionAgent", AttractionAgent)
    workflow.add_node("RestaurantAgent", RestaurantAgent)
    workflow.add_node("DayTransitionAgent", DayTransitionAgent)
    workflow.add_node("ConfirmationAgent", ConfirmationAgent)
    workflow.add_node("PDFCreationAgent", PDFCreationAgent)
    workflow.add_node("call_tools", call_tools)

    entry_points = {
        "InfoCollectorAgent": "InfoCollectorAgent",
        "WeatherAgent": "WeatherAgent",
        "AttractionAgent": "AttractionAgent",
        "RestaurantAgent": "RestaurantAgent",
        "SupervisorAgent": "SupervisorAgent",
        "DayTransitionAgent": "DayTransitionAgent",
        "ConfirmationAgent": "ConfirmationAgent",
        "PDFCreationAgent": "PDFCreationAgent",
        "end_node": END
    }
    workflow.set_conditional_entry_point(supervisor_router, entry_points)
    
    common_edge_mapping = {
        "call_tools": "call_tools", 
        END: END, 
        "PDFCreationAgent": "PDFCreationAgent"
        # "retry" 제거됨
    }

    for agent in ["InfoCollectorAgent", "WeatherAgent", "AttractionAgent", "RestaurantAgent", "SupervisorAgent"]:
        workflow.add_conditional_edges(agent, expert_router, common_edge_mapping)

    workflow.add_edge("DayTransitionAgent", END)
    workflow.add_edge("ConfirmationAgent", END)
    workflow.add_edge("PDFCreationAgent", END)
    workflow.add_conditional_edges("call_tools", supervisor_router, entry_points)

    return workflow.compile(checkpointer=CHECKPOINTER)