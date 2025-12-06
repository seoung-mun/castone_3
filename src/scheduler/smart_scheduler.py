# src/scheduler/smart_scheduler.py

import datetime
from typing import List, Dict
import re
import asyncio

# tools에서 복구한 get_detailed_route 사용
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
    예: "점심: 국수마루" → "국수마루"
    """
    if not raw_name or not isinstance(raw_name, str):
        return raw_name

    cleaned = re.sub(r'^(점심|저녁|아침|오전|오후|숙소|출발|도착)\s*:\s*', '', raw_name)
    cleaned = re.sub(r'\s+(에서|및)\s+.*', '', cleaned)
    cleaned = cleaned.strip()

    return cleaned if cleaned else raw_name

class SmartScheduler:
    def __init__(self, start_time_str: str = "10:00", start_date=None):
        """
        초기화: 여행 시작 시간 및 날짜 설정
        """
        now = datetime.datetime.now()
        base_date = start_date if start_date else now

        try:
            h, m = map(int, start_time_str.split(":"))
            self.current_time = base_date.replace(hour=h, minute=m, second=0, microsecond=0)
            self.start_date = self.current_time.date() 

        except ValueError:
            self.current_time = now
            self.start_date = now.date()

    def _estimate_duration(self, place_info: Dict) -> int:
        """체류 시간 추정"""
        place_type = place_info.get('type', '관광지')
        place_name = place_info.get('name', '')
        
        for key, duration in DEFAULT_DURATIONS.items():
            if key in place_type: return duration
        
        if "카페" in place_name or "커피" in place_name: return 60
        if "식당" in place_name or "맛집" in place_name: return 90
        
        return 90 # 기본값

    async def plan_day(self, places: List[Dict]) -> List[Dict]:
        """
        [핵심 로직] 장소 목록을 받아서 타임라인 생성
        """
        timeline = []
        ordered_places = places 
        cursor_time = self.current_time 

        for i in range(len(ordered_places)):
            current_place = ordered_places[i]
            
            # --- A. 이동 (이전 장소 -> 현재 장소) ---
            if i > 0:
                prev_place = ordered_places[i-1]

                # API 호출용 이름 정제
                prev_api_name = extract_place_name_for_api(prev_place['name'])
                curr_api_name = extract_place_name_for_api(current_place['name'])

                print(f"DEBUG: 🚗 경로 계산: '{prev_api_name}' -> '{curr_api_name}'")

                # API 호출
                route_result = await get_detailed_route(
                    prev_api_name, 
                    curr_api_name, 
                    mode="transit",
                    departure_time=cursor_time
                )
                
                # 기본값 설정
                travel_seconds = 1800 # 30분
                travel_text = "약 30분"
                transport_mode = "transit"
                transport_detail = "" # 상세 정보 (버스 번호 등)

                if route_result:
                    travel_seconds = route_result.get('duration_value', 1800)
                    travel_text = route_result.get('duration', '30분')
                    transport_mode = route_result.get('mode', 'transit')
                    
                    # [핵심 수정] steps에서 버스/지하철 정보 추출
                    steps = route_result.get('steps', [])
                    
                    # "[1003번 버스] 부산역" 같은 형식 찾기
                    details = [s for s in steps if '[' in s or '버스' in s or '지하철' in s]
                    
                    if details:
                        transport_detail = details[0] # 가장 첫 번째 주요 수단 사용
                    elif steps:
                        transport_detail = steps[0]   # 없으면 첫 번째 단계 (예: 도보)
                    else:
                        transport_detail = "이동"

                # 시간 업데이트
                start_move_time = cursor_time
                cursor_time += datetime.timedelta(seconds=travel_seconds)

                # 날짜 변경 체크
                s_suffix = f" (+{(start_move_time.date() - self.start_date).days}일)" if start_move_time.date() != self.start_date else ""
                e_suffix = f" (+{(cursor_time.date() - self.start_date).days}일)" if cursor_time.date() != self.start_date else ""

                travel_info = {
                    "type": "move",
                    "from": prev_place['name'],
                    "to": current_place['name'],
                    "start": start_move_time.strftime("%H:%M") + s_suffix,
                    "end": cursor_time.strftime("%H:%M") + e_suffix,
                    "duration_min": travel_seconds // 60,
                    
                    # [중요] tools.py의 파싱 로직을 위해 필요한 필드들
                    "transport_mode": transport_mode,
                    "transport_detail": transport_detail, 
                    "duration_text_raw": travel_text
                }
                timeline.append(travel_info)


            # --- B. 활동 (현재 장소 체류) ---
            stay_minutes = self._estimate_duration(current_place)

            activity_start = cursor_time
            cursor_time += datetime.timedelta(minutes=stay_minutes)
            activity_end = cursor_time

            # 날짜 변경 체크
            s_suffix = f" (+{(activity_start.date() - self.start_date).days}일)" if activity_start.date() != self.start_date else ""
            e_suffix = f" (+{(activity_end.date() - self.start_date).days}일)" if activity_end.date() != self.start_date else ""

            activity_info = {
                "type": "activity",
                "name": current_place['name'],
                "category": current_place.get('type', '장소'),
                "start": activity_start.strftime("%H:%M") + s_suffix,
                "end": activity_end.strftime("%H:%M") + e_suffix,
                "duration_minutes": stay_minutes,
                "description": current_place.get('description', '')
            }
            timeline.append(activity_info)

        return timeline