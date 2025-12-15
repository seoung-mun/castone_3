import datetime
from typing import List, Dict
import re
import asyncio
from src.tools import get_detailed_route, GMAPS_CLIENT

# --- 설정: 장소 유형별 기본 체류 시간 (분 단위) ---
DEFAULT_DURATIONS = {
    "식당": 90, "카페": 60, "관광지": 120, "산책로": 60, "테마파크": 180, "숙소": 0
}

# [헬퍼 함수] API 호출용 이름 정제
def extract_place_name_for_api(raw_name: str) -> str:
    if not raw_name or not isinstance(raw_name, str): return raw_name
    cleaned = re.sub(r'^(점심|저녁|아침|오전|오후|숙소|출발|도착)\s*:\s*', '', raw_name)
    cleaned = re.sub(r'\s+(에서|및)\s+.*', '', cleaned)
    return cleaned.strip()

class SmartScheduler:
    def __init__(self, start_time_str: str = "10:00", start_date=None):
        now = datetime.datetime.now()
        self.base_date = start_date if start_date else now
        # 기본 초기화 (실제 시간은 plan_day에서 재설정)
        self.current_time = datetime.datetime.combine(self.base_date.date(), datetime.time(10, 0))

    def _estimate_duration(self, place_info: Dict) -> int:
        # DB에서 가져온 type 혹은 이름 기반으로 체류 시간 추정
        place_type = place_info.get('type', '관광지')
        place_name = place_info.get('name', '')
        
        for key, duration in DEFAULT_DURATIONS.items():
            if key in place_type: return duration
        if "카페" in place_name: return 60
        if "식당" in place_name: return 90
        return 90

    async def plan_day(self, places: List[Dict]) -> List[Dict]:
        """
        [로직 수정]
        1. Day 1 -> 12:00 시작 (점심부터)
        2. Day 2~ -> 10:00 시작 (관광부터)
        3. PDF 출력용 type을 'activity'가 아닌 실제 카테고리(식당/카페 등)로 반환
        """
        if not places: return []
        
        timeline = []
        ordered_places = places 
        
        current_day_num = ordered_places[0].get('day', 1)
        
        # 기준 날짜 계산 (여행 시작일 + (N-1)일)
        target_date = self.base_date.date() + datetime.timedelta(days=current_day_num - 1)
        
        if current_day_num == 1:
            # Day 1: 12:00 PM 시작
            self.current_time = datetime.datetime.combine(target_date, datetime.time(12, 0))
            print(f"DEBUG: 📅 Day 1 스케줄링 시작 -> 12:00 PM (점심 기준)")
        else:
            # Day 2+: 10:00 AM 시작
            self.current_time = datetime.datetime.combine(target_date, datetime.time(10, 0))
            print(f"DEBUG: 📅 Day {current_day_num} 스케줄링 시작 -> 10:00 AM")

        cursor_time = self.current_time 

        for i in range(len(ordered_places)):
            current_place = ordered_places[i]
            
            # --- A. 이동 경로 계산 (이전 장소 -> 현재 장소) ---
            if i > 0:
                prev_place = ordered_places[i-1]
                prev_name_api = extract_place_name_for_api(prev_place['name'])
                curr_name_api = extract_place_name_for_api(current_place['name'])

                # Google Maps API 호출
                route_result = await get_detailed_route(
                    prev_name_api, curr_name_api, mode="transit", departure_time=cursor_time
                )
                
                # 기본값 (API 실패 시)
                travel_seconds = 1800 
                travel_text = "약 30분"
                transport_mode = "transit"
                transport_detail = "이동"

                if route_result:
                    travel_seconds = route_result.get('duration_value', 1800)
                    travel_text = route_result.get('duration', '30분')
                    transport_mode = route_result.get('mode', 'transit')
                    steps = route_result.get('steps', [])
                    if steps:
                        transport_detail = " ➡️ ".join(steps)

                # 타임라인 커서 업데이트
                start_move = cursor_time
                cursor_time += datetime.timedelta(seconds=travel_seconds)
                
                # 이동 정보 추가
                timeline.append({
                    "type": "move",
                    "from": prev_place['name'],
                    "to": current_place['name'],
                    "start": start_move.strftime("%H:%M"),
                    "end": cursor_time.strftime("%H:%M"),
                    "duration_min": travel_seconds // 60,
                    "transport_mode": transport_mode,
                    "transport_detail": transport_detail, 
                    "duration_text_raw": travel_text,
                    "day": current_day_num # day 정보 유지
                })

            # --- B. 장소 체류 (Activity) ---
            stay_minutes = self._estimate_duration(current_place)
            activity_start = cursor_time
            cursor_time += datetime.timedelta(minutes=stay_minutes)
            activity_end = cursor_time

            real_category = current_place.get('type', '관광지')
            
            timeline.append({
                "type": real_category,   
                "name": current_place['name'],
                "category": real_category,
                "start": activity_start.strftime("%H:%M"),
                "end": activity_end.strftime("%H:%M"),
                "duration_minutes": stay_minutes,
                "description": current_place.get('description', ''),
                "address": current_place.get('address', ''), 
                "day": current_day_num 
            })

        return timeline