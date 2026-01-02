# ==========================================================================================================
# OpenAI LLM Module
# ==========================================================================================================
# 모델: gpt-4o-mini / gpt-4o
# 특징: OpenAI API 사용 (API Key 필요)
# ==========================================================================================================

import os
from langchain_openai import ChatOpenAI
from llm.base import BaseLLM
from config.settings import LLM_DEVICE, LLM_MAX_TOKENS, LLM_TEMPERATURE

class OpenAILLM(BaseLLM):
    """
    OpenAI Chat Model (GPT-4o-mini 등)
    """
    
    def __init__(self, model_name="gpt-4o-mini"):
        """
        Args:
            model_name (str): 사용할 모델 이름 (기본값: gpt-4o-mini)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            
        print(f"\nOpenAI LLM 로드 중... (Model: {model_name})")
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=api_key,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        print(f"✓ OpenAI LLM 로드 완료")
    
    
    def bind_tools(self, tools, **kwargs):
        """
        Tool Binding (LangChain 호환)
        """
        return self.llm.bind_tools(tools, **kwargs)

    def __getattr__(self, name):
        """
        그 외 메서드는 내부 ChatOpenAI 객체로 위임
        (예: with_structured_output 등)
        """
        return getattr(self.llm, name)

    def generate(self, messages):
        """
        일반 생성
        Args:
            messages: List[Dict] 또는 List[BaseMessage]
        """
        from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
        
        lc_messages = []
        
        # [Fix] 입력 타입 확인 (Dict vs BaseMessage)
        if messages and isinstance(messages[0], BaseMessage):
            # 이미 LangChain 메시지 객체라면 그대로 사용
            lc_messages = messages
        else:
            # Dict 형태라면 변환
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "user":
                    lc_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                
        response = self.llm.invoke(lc_messages)
        return response.content

    def stream(self, messages):
        """
        스트리밍 생성
        """
        from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
        
        lc_messages = []
        
        # [Fix] 입력 타입 확인 (Dict vs BaseMessage)
        if messages and isinstance(messages[0], BaseMessage):
            # 이미 LangChain 메시지 객체라면 그대로 사용
            lc_messages = messages
        else:
            # Dict 형태라면 변환
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "user":
                    lc_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
        
        for chunk in self.llm.stream(lc_messages):
            yield chunk
