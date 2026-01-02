# ==========================================================================================================
# Base LLM Module
# ==========================================================================================================
# 용도: LLM 공통 인터페이스 정의
# 목적: Gemma, Qwen 등 여러 모델을 동일한 방식으로 사용
# ==========================================================================================================

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    LLM 추상 베이스 클래스
    
    모든 LLM 구현체는 이 클래스를 상속받아야 함
    """
    
    @abstractmethod
    def stream(self, messages):
        """
        스트리밍 생성 인터페이스
        
        Args:
            messages (List[Dict]): 메시지 리스트
                [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": "..."}
                ]
        
        Yields:
            str: 토큰 단위로 생성된 텍스트 (실시간 스트리밍)
        """
        pass
    
    @abstractmethod
    def generate(self, messages):
        """
        일반 생성 인터페이스 (스트리밍 아님)
        
        Args:
            messages (List[Dict]): 메시지 리스트
        
        Returns:
            str: 완성된 응답 텍스트
        """
        pass
    
    def _format_messages(self, messages):
        """
        메시지 리스트를 모델별 프롬프트 형식으로 변환
        (각 모델이 오버라이드 가능)
        
        Args:
            messages (List[Dict]): 표준 메시지 형식
        
        Returns:
            str: 모델별 프롬프트 문자열
        """
        # 기본 구현 (시스템 + 사용자)
        system_msg = ""
        user_msg = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                user_msg = msg["content"]
        
        # 간단한 결합
        if system_msg:
            return f"{system_msg}\n\n{user_msg}"
        return user_msg
