from langchain_core.tools import tool
from datetime import datetime

@tool
def get_current_date() -> str:
    """현재 날짜를 YYYY-MM-DD 형식으로 반환"""
    return datetime.now().strftime("%Y-%m-%d")

@tool
def calculate_days_until(deadline: str) -> str:
    """
    마감일까지 남은 일수를 계산
    
    Args:
        deadline: 마감일 (YYYY-MM-DD 형식)
    
    Returns:
        남은 일수 정보 문자열
    """
    try:
        today = datetime.now()
        target = datetime.strptime(deadline, "%Y-%m-%d")
        days = (target - today).days
        
        if days > 0:
            return f"{days}일 남음"
        elif days == 0:
            return "오늘 마감"
        else:
            return f"마감됨 ({abs(days)}일 전)"
    except:
        return "날짜 형식 오류 (YYYY-MM-DD 형식 필요)"

# 툴 리스트
date_tools = [get_current_date, calculate_days_until]
