# ==========================================================================================================
# Qwen 2.5 7B LLM Module
# ==========================================================================================================
# 모델: Qwen/Qwen2.5-7B-Instruct
# 특징: Alibaba 모델, 한국어 및 코딩 능력 우수
# 최적화: 4-bit quantization (메모리 절약)
# ==========================================================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread
from llm.base import BaseLLM
from config.settings import LLM_MODELS, LLM_DEVICE, LLM_MAX_TOKENS, LLM_TEMPERATURE


class QwenLLM(BaseLLM):
    """
    Qwen 2.5 7B Instruction-Tuned 모델
    """
    
    def __init__(self, device=None):
        """
        Args:
            device (str): 'cuda' 또는 'cpu' (기본값: settings.LLM_DEVICE)
        """
        if device is None:
            device = LLM_DEVICE
        
        model_id = LLM_MODELS["Qwen 2.5 7B"]
        
        print(f"\nQwen 2.5 7B 로드 중... (device: {device})")
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
        print(f"✓ Qwen 2.5 7B 로드 완료")
    
    def _format_messages(self, messages):
        """
        Qwen 2.5 형식 (ChatML)으로 프롬프트 변환
        
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        """
        # Tokenizer의 apply_chat_template 사용이 가장 정확함
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            # Fallback (수동 포맷팅)
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            formatted += "<|im_start|>assistant\n"
            return formatted
    
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
            "do_sample": LLM_TEMPERATURE > 0,
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
        """
        return "".join(self.stream(messages))


if __name__ == "__main__":
    # 테스트 코드
    from llm.prompt_template import create_messages
    
    llm = QwenLLM()
    
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
