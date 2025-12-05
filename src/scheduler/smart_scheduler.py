# src/smart_scheduler.py

import datetime
from typing import List, Dict
import re

# 기존 도구 활용
from src.tools import get_detailed_route, GMAPS_CLIENT

# --- 설정: 장소 유형별 기본 체류 시간 (분 단위) ---
DEFAULT_DURATIONS = {
    "식당": 90,      # 1시간 30분
    "카페": 60,      # 1시간
    "관광지": 120,   # 2시간
    "산책로": 60,
    "테마파크": 180, # 3시간
    "숙소": 0
}

# --- 장소 이름 정제 함수 (API 호출용) ---
def extract_place_name_for_api(raw_name: str) -> str:
    """
    Google Maps API 호출을 위해 장소명에서 불필요한 부분을 제거합니다.

    예:
    - "점심: 국수마루에서 맛있는 국수 식사" → "국수마루"
    - "저녁: 제주공항 근처 맛집에서 저녁 식사" → "제주공항 근처 맛집"
    """
    if not raw_name or not isinstance(raw_name, str):
        return raw_name

    # 1. "시간대: " 접두어 제거
    cleaned = re.sub(r'^(점심|저녁|아침|오전|오후|숙소|출발|도착)\s*:\s*', '', raw_name)

    # 2. "~에서", "~및" 이후 설명 제거
    cleaned = re.sub(r'\s+(에서|및)\s+.*', '', cleaned)

    # 3. 공백 정리
    cleaned = cleaned.strip()

    return cleaned if cleaned else raw_name

class SmartScheduler:
    def __init__(self, start_time_str: str = "10:00", start_date=None):
        """
        초기화: 여행 시작 시간을 설정합니다. (기본값: 오전 10시)
        start_date: 여행 시작 날짜 (datetime 객체, 없으면 오늘)
        """
        now = datetime.datetime.now()
        base_date = start_date if start_date else now

        try:
            # HH:MM 형식 파싱
            h, m = map(int, start_time_str.split(":"))
            self.current_time = base_date.replace(hour=h, minute=m, second=0, microsecond=0)
            self.start_date = self.current_time.date()  # 시작 날짜 저장

        except ValueError:
            self.current_time = now
            self.start_date = now.date()

    def _estimate_duration(self, place_info: Dict) -> int:
        """장소 유형이나 이름을 분석하여 체류 시간을 추정합니다."""
        place_type = place_info.get('type', '관광지')
        place_name = place_info.get('name', '')
        
        for key, duration in DEFAULT_DURATIONS.items():
            if key in place_type:
                return duration
        
        if "카페" in place_name or "커피" in place_name: return 60
        if "식당" in place_name or "맛집" in place_name: return 90
        
        return 90 # 기본값

    def plan_day(self, places: List[Dict]) -> List[Dict]:
        """
        [핵심 로직] 장소 목록을 받아서 타임라인을 생성합니다.
        (이동 시간 API 조회 + 체류 시간 계산)
        """
        timeline = []
        ordered_places = places 
        cursor_time = self.current_time 

        for i in range(len(ordered_places)):
            current_place = ordered_places[i]
            
            # --- A. 이동 (이전 장소 -> 현재 장소) ---
            if i > 0:
                prev_place = ordered_places[i-1]

                # 🚨 [수정] API 호출용 장소명 정제
                prev_place_api = extract_place_name_for_api(prev_place['name'])
                current_place_api = extract_place_name_for_api(current_place['name'])

                print(f"DEBUG: API 호출 - '{prev_place['name']}' → '{prev_place_api}'")
                print(f"DEBUG: API 호출 - '{current_place['name']}' → '{current_place_api}'")

                # 구글 맵 API로 실제 이동 시간 조회
                route_result = get_detailed_route(
                    prev_place_api,  # 정제된 이름 사용
                    current_place_api,  # 정제된 이름 사용
                    mode="transit",
                    departure_time=cursor_time
                )
                
                if route_result:
                    # 실제 소요 시간(초)을 가져와서 계산
                    travel_seconds = route_result.get('duration_value', 1800) # 없으면 30분 가정
                    travel_text = route_result.get('duration', '30분')
                    
                    start_move_time = cursor_time
                    cursor_time += datetime.timedelta(seconds=travel_seconds)

                    # 🚨 [수정] 날짜가 바뀌면 표시 (자정 넘김 감지)
                    start_date_suffix = ""
                    end_date_suffix = ""
                    if start_move_time.date() != self.start_date:
                        days_diff = (start_move_time.date() - self.start_date).days
                        start_date_suffix = f" (+{days_diff}일)"
                    if cursor_time.date() != self.start_date:
                        days_diff = (cursor_time.date() - self.start_date).days
                        end_date_suffix = f" (+{days_diff}일)"

                    travel_info = {
                        "type": "move",
                        "from": prev_place['name'],  # 원본 이름 유지 (PDF 표시용)
                        "to": current_place['name'],  # 원본 이름 유지 (PDF 표시용)
                        "start": start_move_time.strftime("%H:%M") + start_date_suffix,
                        "end": cursor_time.strftime("%H:%M") + end_date_suffix,
                        "duration_text": travel_text,
                        "transport": route_result['steps'][0] if route_result['steps'] else "이동"
                    }
                    timeline.append(travel_info)
                else:
                    # 경로 못 찾음 (도보 10분 가정)
                    cursor_time += datetime.timedelta(minutes=10)

            # --- B. 활동 (현재 장소 체류) ---
            stay_minutes = self._estimate_duration(current_place)

            activity_start = cursor_time
            cursor_time += datetime.timedelta(minutes=stay_minutes)
            activity_end = cursor_time

            # 🚨 [수정] 날짜가 바뀌면 표시
            start_date_suffix = ""
            end_date_suffix = ""
            if activity_start.date() != self.start_date:
                days_diff = (activity_start.date() - self.start_date).days
                start_date_suffix = f" (+{days_diff}일)"
            if activity_end.date() != self.start_date:
                days_diff = (activity_end.date() - self.start_date).days
                end_date_suffix = f" (+{days_diff}일)"

            activity_info = {
                "type": "activity",
                "name": current_place['name'],  # 원본 이름 유지 (PDF 표시용)
                "category": current_place.get('type', '장소'),
                "start": activity_start.strftime("%H:%M") + start_date_suffix,
                "end": activity_end.strftime("%H:%M") + end_date_suffix,
                "duration_minutes": stay_minutes,
                "description": current_place.get('description', '')
            }
            timeline.append(activity_info)

        return timeline