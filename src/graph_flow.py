# src/graph_flow.py

from typing import TypedDict, Annotated, List, Literal, Dict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from src.config import LLM
from src.tools import AVAILABLE_TOOLS, TOOLS
import re # 정규표현식 라이브러리 임포트
import json

# --- 1. LangGraph: 멀티 에이전트 상태 정의 ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_weather: str
    itinerary: List[Dict]
    destination: str
    dates: str
    preference: str
    total_days: int
    activity_level: int
    current_planning_day: int
    show_pdf_button: bool # [추가] PDF 다운로드 버튼 표시 여부를 제어하는 상태
    next_node: Literal[
        "InfoCollectorAgent", "WeatherAgent", "AttractionAgent", "RestaurantAgent",
        "DayTransitionAgent", "ConfirmationAgent", "PDFCreationAgent", "end_node" # [추가] 새 에이전트
    ]

# --- 2. 전문 에이전트(노드) 정의 ---
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
        content = response.content or ""

        # SupervisorAgent가 생성한 최종 itinerary JSON을 파싱하여 상태를 업데이트하는 로직
        final_itinerary_match = re.search(r"\[FINAL_ITINERARY_JSON\](.*)\[/FINAL_ITINERARY_JSON\]", content, re.DOTALL)
        if final_itinerary_match:
            try:
                # 정규표현식으로 추출한 JSON 문자열에서 불필요한 공백/줄바꿈 제거
                itinerary_json_str = final_itinerary_match.group(1).strip()
                # JSON 문자열을 파이썬 리스트 객체로 변환
                parsed_itinerary = json.loads(itinerary_json_str)
                
                # 디버깅을 위해 터미널에 출력
                print(f"DEBUG: SupervisorAgent가 최종 정리한 itinerary:\n{parsed_itinerary}")
                
                # 현재 itinerary 상태를 새로 파싱한 데이터로 완전히 교체
                itinerary = parsed_itinerary
            except json.JSONDecodeError as e:
                # JSON 변환 중 오류가 발생하면 터미널에 에러 메시지 출력
                print(f"ERROR: 최종 itinerary JSON 파싱에 실패했습니다. 오류: {e}")
                print(f"파싱 시도 원본 문자열: {itinerary_json_str}")

        # 기존의 간단한 일정 추가 로직 (대화 중에 장소를 하나씩 추가할 때 사용)
        match = re.search(r"'(.*?)'을/를 (\d+)일차 (관광지|식당|카페) 계획에 추가합니다", content)
        if match:
            name, day, type = match.groups()
            # [수정 3] 간단한 추가 시에는 'description' 키가 없으므로 기본값을 넣어줍니다.
            new_item = {'day': int(day), 'type': type, 'name': name, 'description': ''}
            if new_item not in itinerary:
                itinerary.append(new_item)

        # PDF 다운로드 버튼 표시 여부를 제어하는 로직
        show_pdf_button = state.get('show_pdf_button', False)
        if "[STATE_UPDATE: show_pdf_button=True]" in content:
            show_pdf_button = True

        return {"messages": [response], "itinerary": itinerary, "show_pdf_button": show_pdf_button}
    return agent_node

# --- 3. Supervisor (라우터) 정의 ---
def supervisor_router(state: AgentState):
    print("--- (Supervisor) 다음 작업 결정 ---")
    if not all(state.get(key) for key in ['destination', 'dates', 'total_days', 'activity_level']): return "InfoCollectorAgent"
    if not state.get('current_weather'): return "WeatherAgent"
    if not state.get('preference'): return "SupervisorAgent"

    # [수정] 라우터 로직에 PDF 요청 처리 추가
    last_message = state['messages'][-1]
    last_ai_message = state['messages'][-2] if len(state['messages']) > 1 and isinstance(state['messages'][-1], ToolMessage) else last_message
    
    # 1. 슈퍼바이저가 PDF 준비를 마쳤다는 신호를 보내면, 그때 PDF 에이전트로 보냅니다.
    if isinstance(last_ai_message, AIMessage) and "PDF 생성을 준비합니다" in last_ai_message.content:
        print("Supervisor -> PDFCreationAgent (슈퍼바이저가 준비 완료 신호를 보냄)")
        return "PDFCreationAgent"

    # 2. 사용자가 처음 PDF를 요청하면, '정리'를 위해 슈퍼바이저에게 먼저 보냅니다.
    if isinstance(last_message, HumanMessage):
        content = last_message.content.lower()
        if any(k in content for k in ["pdf", "파일", "정리", "다운로드"]):
            print("Supervisor -> SupervisorAgent (PDF 생성을 위한 데이터 정리 요청)")
            return "SupervisorAgent" # <--- 목적지를 PDFCreationAgent에서 SupervisorAgent로 변경!
            
        if any(k in content for k in ["최적화", "순서", "경로"]): return "SupervisorAgent"
        if any(k in content for k in ["식당", "맛집", "카페", "먹고"]): return "RestaurantAgent"
        if any(k in content for k in ["관광", "장소", "다른 곳"]): return "AttractionAgent"

    total_days, current_day, activity_level = state.get("total_days"), state.get("current_planning_day"), state.get("activity_level")
    places_for_current_day = [p for p in state.get("itinerary", []) if p.get('day') == current_day]

    if len(places_for_current_day) >= activity_level:
        if current_day < total_days: return "DayTransitionAgent"
        else: return "ConfirmationAgent"
            
    if isinstance(state['messages'][-1], AIMessage) and "계획에 추가합니다" in state['messages'][-1].content:
        print(f"Supervisor -> AttractionAgent ({current_day}일차 연속 추천, {len(places_for_current_day)}/{activity_level}곳)")
        return "AttractionAgent"
    
    if isinstance(last_message, ToolMessage):
        return "SupervisorAgent"

    return "SupervisorAgent"

# --- 4. 에이전트(노드) 생성 (프롬프트 전체 복원) ---

# [추가] PDF 생성을 전담하는 새 에이전트 프롬프트
pdf_creation_prompt = """당신은 'PDF 문서 생성 전문가'입니다.
사용자가 "pdf로 정리해줘", "파일로 만들어줘" 등 여행 계획을 파일로 만들고 싶어하는 요청을 받았습니다.

당신의 유일한 임무는 사용자에게 PDF 다운로드 버튼이 곧 표시될 것임을 알리는 것입니다.
아래와 같이 정확한 문장으로 응답하고, **응답 마지막에 상태 업데이트를 위한 명령어를 반드시 포함**해야 합니다.

"네, 전체 여행 계획을 PDF 파일로 정리해 드릴게요. 잠시 후 표시되는 버튼을 눌러 다운로드하세요."

명령어:
[STATE_UPDATE: show_pdf_button=True]
"""
PDFCreationAgent = create_specialist_agent(pdf_creation_prompt)

supervisor_prompt = """당신은 AI 여행 플래너 팀의 '슈퍼바이저'입니다.
당신은 전문가 팀(날씨, 관광, 식당)을 관리하고, 사용자와의 상호작용을 총괄하며, 계획의 전체적인 흐름을 책임집니다.

### 주요 임무
1.  **계획 추가 확인:** 사용자가 장소를 선택하면(예: "1번 경희궁으로 할래"), **현재 계획 중인 날짜(`current_planning_day`)** 와 장소 유형('관광지' 또는 '식당')을 명시하여 확인 메시지를 반환하세요. 이 메시지는 시스템이 다음 행동을 결정하는 중요한 신호입니다.
    *   (관광지 예시): "네, '경희궁'을 **1일차 관광지** 계획에 추가합니다."
    *   (식당 예시): "좋습니다. '금화왕돈까스'를 **1일차 식당** 계획에 추가합니다."
2.  **하루 단위 경로 최적화:** 사용자가 "경로 최적화", "1일차 순서 짜줘" 등을 요청하면, 요청받은 날짜(`current_planning_day`)에 해당하는 장소 목록만 `itinerary`에서 추출하여 `optimize_and_get_routes` 도구를 호출하세요. 식당과 관광지를 모두 포함해야 합니다.
3.  **도구 결과 브리핑:** 모든 도구의 결과를 사용자에게 친절하게 요약하여 제시하고, 다음 행동(장소 선택, 취향 질문 등)을 유도하세요.
4.  **취향 질문:** `get_weather_forecast` 도구 결과를 받으면, 날씨 정보를 브리핑하고 날씨에 맞는 활동 '취향'을 질문하세요.
5.  **일반 대화:** 그 외 사용자의 일반적인 질문이나 애매한 요청에 응답합니다.
### 추가 임무 (★★매우 중요★★)
6.  **PDF용 콘텐츠 생성 및 최종 데이터 정리:** 사용자가 "pdf 만들어줘", "파일로 정리해줘" 등의 요청을 하면, 당신의 가장 중요한 임무가 시작됩니다.
    a. **전체 대화 기록(`messages`)과 현재 일정(`itinerary`)을 종합적으로 분석**하세요.
    b. 각 장소가 **왜 추천되었는지, 어떤 특징이 언급되었는지 대화의 맥락에서 파악**하세요. (예: "친구와 가기 좋은", "인테리어가 멋진", "비빔밥이 유명한")
    c. 이 맥락을 바탕으로, 각 장소에 대한 **1~2줄의 매력적인 설명(`description`)을 작성**하세요.
    d. 작성이 끝나면, **아래와 같은 JSON 형식을 사용하여, 설명이 추가된 '최종 itinerary' 전체를 반드시 출력**해야 합니다. 이 형식은 시스템이 PDF에 내용을 쓰는 유일한 방법입니다.

    --- 최종 데이터 출력 형식 (이 안에 최종 itinerary를 넣으세요) ---
    [FINAL_ITINERARY_JSON]
    [
      {{"day": 1, "type": "식당", "name": "경리단길", "description": "통이 터지도록 듬뿍 담아주는 비빔밥이 유명한 한식당입니다."}},
      {{"day": 1, "type": "카페", "name": "8IGHTY4OUR 카페", "description": "친구와 함께 방문하기 좋으며, 멋진 인테리어와 편안한 공간이 특징입니다."}},
      {{"day": 2, "type": "관광지", "name": "경복궁", "description": "한국의 역사를 느낄 수 있는 아름다운 궁궐입니다."}}
    ]
    [/FINAL_ITINERARY_JSON]
    ---

    e. 위 데이터 출력이 끝난 후, 사용자에게 "네, 대화 내용을 바탕으로 여행 계획을 상세하게 정리했습니다. 잠시 후 PDF 다운로드 버튼이 나타납니다." 라고 간단히 응답하세요.
"""
SupervisorAgent = create_specialist_agent(supervisor_prompt)

def day_transition_agent_node(state: AgentState):
    """한 날의 계획이 끝나고 다음 날 계획을 시작할 것을 알리는 에이전트입니다."""
    
    # 상태에서 필요한 변수들을 직접 가져옵니다.
    current_day = state.get("current_planning_day", 1)
    activity_level = state.get("activity_level", 3)
    next_day = current_day + 1

    # f-string을 사용하여 프롬프트 텍스트를 완성합니다.
    prompt_text = f"""당신은 '플랜 전환 안내자'입니다.
당신의 유일한 임무는 한 날의 계획이 끝나고 다음 날 계획을 시작할 것을 알리는 것입니다.
현재 상태를 참고하여, 아래와 같이 정확한 문장으로 응답하고 사용자의 동의를 구하세요.

"이제 {current_day}일차 목표 활동량({activity_level}곳)이 모두 채워졌습니다. 다음 날인 {next_day}일차 계획을 시작할까요?"

응답 마지막에 다음 상태 업데이트를 위한 명령어를 반드시 포함해야 합니다:
[STATE_UPDATE: increment_day=True]
"""

    # 이 에이전트는 도구를 사용하지 않으므로 LLM만 호출합니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("placeholder", "{messages}")
    ])
    chain = prompt | LLM
    
    # state_summary 없이 마지막 메시지만 전달하여 단순 응답을 유도합니다.
    response = chain.invoke({"messages": state['messages'][-1:]})

    return {"messages": [response]}

# DayTransitionAgent를 위에서 정의한 새 함수로 지정합니다.
DayTransitionAgent = day_transition_agent_node

confirmation_prompt = """당신은 '일정 확인 전문가'입니다.
당신의 유일한 임무는 모든 날의 계획이 완료되었음을 알리고 최종 확인 질문을 하는 것입니다.
"모든 날의 계획이 완료되었습니다. 이대로 전체 일정을 확정하고 경로 최적화를 진행할까요?"
"""
ConfirmationAgent = create_specialist_agent(confirmation_prompt)

infocollector_prompt = """당신은 '정보 수집가'입니다. 
당신의 임무는 사용자와의 대화에서 (1)여행 목적지, (2)여행 날짜, (3)하루 활동량 정보를 파악하는 것입니다.

1.  **현재 상태**와 **대화 기록**을 분석하세요.
2.  **부족한 정보 파악:** `destination`, `dates`, `activity_level`, `total_days` 중 비어있는 정보를 확인합니다.
3.  **대화 분석:** 사용자의 마지막 메시지에 필요한 정보가 있는지 확인합니다.
4.  **응답 생성:**
    * **(파싱 성공 시):** 파악한 모든 정보를 **상태 업데이트 태그**에 반드시 포함하여 응답하세요.
        * (예시): "네, '서울' 1박 2일, 활동량 보통(3)으로 설정하겠습니다. [STATE_UPDATE: destination="서울", dates="1박 2일", total_days="2", activity_level="3", current_planning_day="1"]"
    * **(정보 부족 시):** 파싱할 정보가 없다면, 비어있는 정보를 사용자에게 정중하게 **질문**하세요. ('취향'은 묻지 않습니다)
"""
InfoCollectorAgent = create_specialist_agent(infocollector_prompt)

attraction_prompt = """당신은 '관광지 전문가'입니다.
`search_attractions_and_reviews` 도구를 호출하여 관광지 후보를 검색해야 합니다.
`destination`과 `preference`를 조합하여 RAG 쿼리를 생성하세요. (예: "서울 실내 활동")"""
AttractionAgent = create_specialist_agent(attraction_prompt)

restaurant_prompt = """당신은 '식당 전문가'입니다.
당신의 임무는 사용자의 요청을 분석하여 식당 후보를 검색하는 것입니다.
1.  사용자의 요청(예: "식당 고를래", "파스타 먹고 싶어")을 확인합니다.
2.  정보가 모호하면(예: "식당 고를래"), 도구를 호출하지 말고 **반드시 "어떤 종류의 식당(메뉴/분위기)을 찾으시나요?"라고 질문**하세요.
3.  정보가 충분하면, `destination`과 사용자 요청(예: "파스타")을 조합하여 RAG 쿼리를 생성하고 `search_attractions_and_reviews` 도구를 호출하세요.
"""
RestaurantAgent = create_specialist_agent(restaurant_prompt)

weather_prompt = """당신은 '날씨 분석가'입니다.
`get_weather_forecast` 도구를 호출하여 `destination`과 `dates`의 날씨를 확인하세요."""
WeatherAgent = create_specialist_agent(weather_prompt)

# --- 5. 도구 실행 노드 ---
def call_tools(state: AgentState):
    last_message = state['messages'][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
        
    tool_messages = []
    weather_update = state.get('current_weather')
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_to_call = AVAILABLE_TOOLS.get(tool_name)
        output = ""
        if not tool_to_call:
            output = f"오류: '{tool_name}'라는 이름의 도구를 찾을 수 없습니다."
        else:
            try:
                output = tool_to_call.invoke(tool_args)
                if tool_name == "get_weather_forecast":
                    weather_update = output
            except Exception as e:
                output = f"도구 '{tool_name}' 실행 중 오류 발생: {e}"
        tool_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
        
    return {"messages": tool_messages, "current_weather": weather_update}

# --- 6. 그래프 빌더 함수 (명시적 버전) ---
def build_graph():
    workflow = StateGraph(AgentState)

    # 모든 노드를 명시적으로 추가
    workflow.add_node("SupervisorAgent", SupervisorAgent)
    workflow.add_node("InfoCollectorAgent", InfoCollectorAgent)
    workflow.add_node("WeatherAgent", WeatherAgent)
    workflow.add_node("AttractionAgent", AttractionAgent)
    workflow.add_node("RestaurantAgent", RestaurantAgent)
    workflow.add_node("DayTransitionAgent", DayTransitionAgent)
    workflow.add_node("ConfirmationAgent", ConfirmationAgent)
    workflow.add_node("PDFCreationAgent", PDFCreationAgent) # [추가] 새 노드 추가
    workflow.add_node("call_tools", call_tools)

    # 진입점 설정
    entry_points = {
        "InfoCollectorAgent": "InfoCollectorAgent",
        "WeatherAgent": "WeatherAgent",
        "AttractionAgent": "AttractionAgent",
        "RestaurantAgent": "RestaurantAgent",
        "SupervisorAgent": "SupervisorAgent",
        "DayTransitionAgent": "DayTransitionAgent",
        "ConfirmationAgent": "ConfirmationAgent",
        "PDFCreationAgent": "PDFCreationAgent", # [추가] 새 진입점 추가
        "end_node": END
    }
    workflow.set_conditional_entry_point(supervisor_router, entry_points)
    
    # 전문가 노드들의 다음 경로 설정 (도구 호출 또는 종료)
    def expert_router(state: AgentState):
        last_message = state['messages'][-1]
        if isinstance(last_message, AIMessage):
            # 1순위: Supervisor가 PDF용 JSON을 생성했는지 확인
            if "[FINAL_ITINERARY_JSON]" in last_message.content:
                print("Router -> PDFCreationAgent (JSON 데이터 확인)")
                return "PDFCreationAgent"
            # 2순위: 도구 호출이 있는지 확인
            if last_message.tool_calls:
                print(f"Router -> call_tools (도구: {[tc['name'] for tc in last_message.tool_calls]})")
                return "call_tools"
        # 3순위: 위 조건에 해당하지 않으면 종료
        print("Router -> END")
        return END

    # [변경점] 모든 전문가 노드를 수정된 expert_router에 연결합니다.
    workflow.add_conditional_edges("InfoCollectorAgent", expert_router, {"call_tools": "call_tools", END: END, "PDFCreationAgent": "PDFCreationAgent"})
    workflow.add_conditional_edges("WeatherAgent", expert_router, {"call_tools": "call_tools", END: END, "PDFCreationAgent": "PDFCreationAgent"})
    workflow.add_conditional_edges("AttractionAgent", expert_router, {"call_tools": "call_tools", END: END, "PDFCreationAgent": "PDFCreationAgent"})
    workflow.add_conditional_edges("RestaurantAgent", expert_router, {"call_tools": "call_tools", END: END, "PDFCreationAgent": "PDFCreationAgent"})
    workflow.add_conditional_edges("SupervisorAgent", expert_router, {"call_tools": "call_tools", END: END, "PDFCreationAgent": "PDFCreationAgent"})

    # 전환/확인/PDF 노드는 항상 종료 (사용자 입력 대기)
    workflow.add_edge("DayTransitionAgent", END)
    workflow.add_edge("ConfirmationAgent", END)
    workflow.add_edge("PDFCreationAgent", END) # [추가] 새 노드의 엣지 추가

    # 도구 실행 후에는 항상 SupervisorRouter로 돌아가 다음 작업 결정
    workflow.add_conditional_edges("call_tools", supervisor_router, entry_points)

    return workflow.compile()