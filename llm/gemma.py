# ==========================================================================================================
# Gemma 2 9B LLM Module
# ==========================================================================================================
# 모델: google/gemma-2-9b-it
# 특징: Google 모델, Instruction-tuned, 한국어 지원 양호
# 최적화: 4-bit quantization (메모리 절약)
# ==========================================================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread
from llm.base import BaseLLM
from config.settings import LLM_MODELS, LLM_DEVICE, LLM_MAX_TOKENS, LLM_TEMPERATURE


class GemmaLLM(BaseLLM):
    """
    Gemma 2 9B Instruction-Tuned 모델
    """
    
    def __init__(self, device=None):
        """
        Args:
            device (str): 'cuda' 또는 'cpu' (기본값: settings.LLM_DEVICE)
        """
        if device is None:
            device = LLM_DEVICE
        
        model_id = LLM_MODELS["Gemma 2 9B"]
        
        print(f"\nGemma 2 9B 로드 중... (device: {device})")
        print("  4-bit quantization 적용 (메모리 절약)")
        
        # ========================================
        # [1] 4-bit Quantization 설정
        # ========================================
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # ========================================
        # [2] Tokenizer 로드
        # ========================================
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # ========================================
        # [3] 모델 로드
        # ========================================
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        self.device = device
        print(f"✓ Gemma 2 9B 로드 완료")
    
    def _format_messages(self, messages):
        """
        Gemma 2 형식으로 프롬프트 변환
        
        Gemma 2 템플릿:
        <start_of_turn>user
        {user_message}<end_of_turn>
        <start_of_turn>model
        """
        formatted_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                # 시스템 메시지는 user turn 앞에 추가
                formatted_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "user":
                formatted_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        
        # 모델 turn 시작
        formatted_parts.append("<start_of_turn>model")
        
        return "\n".join(formatted_parts)
    
    def stream(self, messages):
        """
        스트리밍 생성
        
        Args:
            messages (List[Dict]): [{"role": "system|user", "content": "..."}]
        
        Yields:
            str: 토큰 단위 생성 텍스트
        """
        # 프롬프트 포맷팅
        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # ========================================
        # [스트리밍] TextIteratorStreamer 사용
        # ========================================
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # 프롬프트 제외
            skip_special_tokens=True
        )
        
        # 별도 스레드에서 생성 실행
        generation_kwargs = {
            **inputs,
            "max_new_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "do_sample": LLM_TEMPERATURE > 0,  # temperature=0이면 greedy
            "streamer": streamer
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 스트리밍 출력
        for text in streamer:
            yield text
    
    def generate(self, messages):
        """
        일반 생성 (스트리밍 아님)
        
        Args:
            messages (List[Dict]): 메시지 리스트
        
        Returns:
            str: 완성된 응답
        """
        # 스트리밍 결과를 모아서 반환
        return "".join(self.stream(messages))


if __name__ == "__main__":
    # 테스트 코드
    from llm.prompt_template import create_messages
    
    llm = GemmaLLM()
    
    test_context = """
[문서 1]
- 사업명: 테스트 학사정보시스템 구축
- 발주기관: 테스트대학교
- 예산: 1억원
"""
    test_query = "이 사업에 대해 간단히 설명해줘"
    
    messages = create_messages(test_context, test_query)
    
    print("\n스트리밍 시작:")
    print("-" * 60)
    for chunk in llm.stream(messages):
        print(chunk, end="", flush=True)
    print("\n" + "-" * 60)
