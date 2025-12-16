import datetime
from typing import List, Dict
import re
import asyncio
from src.tools import get_detailed_route, GMAPS_CLIENT

DEFAULT_DURATIONS = {
    "식당": 90, "카페": 60, "관광지": 120, "산책로": 60, "테마파크": 180, "숙소": 0
}

def extract_place_name_for_api(raw_name: str) -> str:
    if not raw_name or not isinstance(raw_name, str): return raw_name
    cleaned = re.sub(r'^(점심|저녁|아침|오전|오후|숙소|출발|도착)\s*:\s*', '', raw_name)
    cleaned = re.sub(r'\s+(에서|및)\s+.*', '', cleaned)
    return cleaned.strip()

class SmartScheduler:
    def __init__(self, start_time_str: str = "10:00", start_date=None):
        now = datetime.datetime.now()
        self.base_date = start_date if start_date else now
        self.current_time = datetime.datetime.combine(self.base_date.date(), datetime.time(10, 0))

    def _estimate_duration(self, place_info: Dict) -> int:
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
        
        target_date = self.base_date.date() + datetime.timedelta(days=current_day_num - 1)
        
        if current_day_num == 1:
            self.current_time = datetime.datetime.combine(target_date, datetime.time(12, 0))
            print(f" Day 1 스케줄링 시작 -> 12:00 PM (점심 기준)")
        else:
            self.current_time = datetime.datetime.combine(target_date, datetime.time(10, 0))
            print(f" Day {current_day_num} 스케줄링 시작 -> 10:00 AM")

        cursor_time = self.current_time 

        for i in range(len(ordered_places)):
            current_place = ordered_places[i]
            
            if i > 0:
                prev_place = ordered_places[i-1]
                prev_name_api = extract_place_name_for_api(prev_place['name'])
                curr_name_api = extract_place_name_for_api(current_place['name'])

                route_result = await get_detailed_route(
                    prev_name_api, curr_name_api, mode="transit", departure_time=cursor_time
                )
                
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

                start_move = cursor_time
                cursor_time += datetime.timedelta(seconds=travel_seconds)
                
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
                    "day": current_day_num 
                })

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