# src/time_planner.py

from typing import List, Union, Dict
import json
import re
from datetime import datetime, timedelta
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.config import LLM

# --- 1. 출력 스키마 정의 ---
class TimedItineraryItem(BaseModel):
    day: int = Field(description="여행 일차")
    type: str = Field(description="장소 유형")
    name: str = Field(description="장소 이름")
    description: str = Field(description="장소 설명")
    estimated_start_time: str = Field(description="시작 시간 (예: 10:00)")
    estimated_end_time: str = Field(description="종료 시간 (예: 12:00)")
    estimated_duration_minutes: int = Field(description="소요 시간(분)")

class TimedItinerary(BaseModel):
    timed_itinerary: List[TimedItineraryItem] = Field(description="시간 정보가 할당된 전체 일정 리스트")

# --- 2. 시간 검증 및 수정 함수 ---
def validate_and_fix_time(time_str: str, default_time: str = "10:00") -> str:
    """시간 형식 검증 및 수정 (HH:MM 형식으로 변환)"""
    try:
        # "HH:MM" 형식 확인
        if re.match(r'^\d{1,2}:\d{2}$', time_str):
            hour, minute = map(int, time_str.split(':'))
            # 시간이 0-23, 분이 0-59 범위인지 확인
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"
    except:
        pass

    # 잘못된 형식이면 기본값 반환
    print(f"DEBUG: 잘못된 시간 형식 '{time_str}' → '{default_time}'로 수정")
    return default_time

def fix_time_sequence(items: List[Dict]) -> List[Dict]:
    """시간 순서 검증 및 수정 (종료 시간이 시작 시간보다 빠른 경우 수정)"""
    for item in items:
        start_str = item.get('estimated_start_time', '10:00')
        end_str = item.get('estimated_end_time', '11:00')

        # 시간 검증 및 수정
        start_str = validate_and_fix_time(start_str, '10:00')
        end_str = validate_and_fix_time(end_str, '11:00')

        try:
            start_time = datetime.strptime(start_str, '%H:%M')
            end_time = datetime.strptime(end_str, '%H:%M')

            # 종료 시간이 시작 시간보다 빠르면 수정
            if end_time <= start_time:
                duration = item.get('estimated_duration_minutes', 60)
                end_time = start_time + timedelta(minutes=duration)
                end_str = end_time.strftime('%H:%M')
                print(f"DEBUG: 시간 순서 오류 수정 - {item.get('name', '?')}: {start_str}~{item.get('estimated_end_time')} → {start_str}~{end_str}")

            item['estimated_start_time'] = start_str
            item['estimated_end_time'] = end_str

        except Exception as e:
            print(f"DEBUG: 시간 파싱 오류: {e}")
            # 기본값 설정
            item['estimated_start_time'] = start_str
            item['estimated_end_time'] = end_str

    return items

# --- 3. 프롬프트 ---
TIMELINE_SYSTEM_PROMPT = """당신은 '여행 일정 시간 계산 전문가'입니다.
주어진 여행 일정 목록을 분석하여, 각 항목에 합리적인 활동 시간(시작, 종료, 소요 시간)을 할당하세요.

**중요 규칙:**
1. 시작 시간은 10:00부터 시작합니다
2. 점심 식사는 12:00~13:00 사이에 배치합니다
3. 저녁 식사는 18:00~19:30 사이에 배치합니다
4. 모든 시간은 HH:MM 형식 (24시간제)으로 작성하세요
5. 종료 시간은 반드시 시작 시간보다 늦어야 합니다
6. 이동 시간과 활동 시간을 고려하여 현실적인 일정을 작성하세요
"""

def create_time_planner_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", TIMELINE_SYSTEM_PROMPT),
        ("human", "아래 여행 일정에 대해 시간 계획을 할당하세요:\n{itinerary_json_str}")
    ])
    # 최신 LangChain에서는 표준 Pydantic 모델을 지원합니다.
    chain = prompt | LLM.with_structured_output(TimedItinerary)
    return chain

# --- 3. 구현 함수 (이름: plan) ---
def plan(itinerary_input: Union[str, List[Dict]]) -> str:
    print(f"\n--- [DEBUG TIME PLANNER] 시간 계획 시작 ---")
    
    # 입력값 전처리 (리스트/문자열 모두 처리)
    try:
        if isinstance(itinerary_input, str):
            itinerary_data = json.loads(itinerary_input)
        else:
            itinerary_data = itinerary_input
            
    except json.JSONDecodeError:
        return "오류: 여행 일정 데이터 형식이 올바르지 않습니다."

    # 날짜순 정렬
    try:
        sorted_itinerary = sorted(itinerary_data, key=lambda x: x.get('day', 1))
    except:
        sorted_itinerary = itinerary_data

    chain = create_time_planner_chain()
    
    try:
        # LLM 호출
        result_obj = chain.invoke({"itinerary_json_str": json.dumps(sorted_itinerary, ensure_ascii=False)})

        # Pydantic v2의 경우 model_dump(), v1의 경우 dict()를 사용
        # 호환성을 위해 try-except로 처리하거나 dict() 사용
        try:
            final_list = [item.model_dump() for item in result_obj.timed_itinerary]
        except AttributeError:
            final_list = [item.dict() for item in result_obj.timed_itinerary]

        # 🚨 [핵심 추가] 시간 검증 및 수정
        final_list = fix_time_sequence(final_list)

        final_json_str = json.dumps(final_list, ensure_ascii=False, indent=2)

        print(f"DEBUG: 시간 검증 완료 후 최종 계획 JSON:\n{final_json_str}")
        return final_json_str
        
    except Exception as e:
        print(f"DEBUG: Error details: {e}")
        return f"오류: 여행 시간 계획 실패 ({e})"