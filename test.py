import sys
import os

# 현재 경로를 시스템 경로에 추가 (모듈 인식용)
sys.path.append(os.getcwd())

print("🚀 [1단계] src.config 임포트 시도...")
try:
    from src import config
    print("✅ config 임포트 성공!")
except Exception as e:
    print(f"❌ config 임포트 실패: {e}")
    exit()

print("\n🚀 [2단계] src.time_planner 임포트 시도...")
try:
    from src import time_planner
    print("✅ time_planner 임포트 성공!")
except Exception as e:
    print(f"❌ time_planner 임포트 실패: {e}")
    exit()

print("\n🚀 [3단계] src.tools 임포트 시도...")
try:
    from src import tools
    print("✅ tools 임포트 성공!")
except Exception as e:
    print(f"❌ tools 임포트 실패: {e}")
    exit()

print("\n🚀 [4단계] src.graph_flow 임포트 시도...")
try:
    from src import graph_flow
    print("✅ graph_flow 임포트 성공!")
except Exception as e:
    print(f"❌ graph_flow 임포트 실패: {e}")
    exit()

print("\n🎉 모든 모듈이 정상입니다. 코드 문법에는 문제가 없습니다.")