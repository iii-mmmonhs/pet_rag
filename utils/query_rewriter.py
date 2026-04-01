import logging
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import REWRITER_MODEL, DEVICE
import torch

logger = logging.getLogger(__name__)

class QueryRewriter:
    """
    Query Rewriting для переформулировки запросов пользователей:
    заменяет местоимения на конкретные сущности из истории чата, 
    что делает запрос самодостаточным для поиска.
    
    Использует локальную LLM, но легкую (0.6B).
    """
    def __init__(self):
        self.device = DEVICE
        self.model_name = REWRITER_MODEL
        
        logger.info("Загрузка рерайтера")
        
        self.tokenizer = AutoTokenizer.from_pretrained(REWRITER_MODEL, trust_remote_code=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            REWRITER_MODEL,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
                    
        self.model.eval()
        logger.info("Рерайтер готов")

    def rewrite(self, query: str, chat_history: List[str]) -> str:
        """
        Переформулирует запрос на основе контекста диалога.
        
        Args:
            query: Текущий вопрос пользователя.
            chat_history: История сообщений
            
        Returns:
            str: Переформулированный вопрос или исходный query.
        """
        if not chat_history:
            return query

        history_texts = []
        for msg in chat_history:
            if hasattr(msg, 'content'):
                history_texts.append(msg.content)
            else:
                history_texts.append(str(msg))

        history_str = "\n".join(history_texts)
        
        prompt_template = (
            "Дан диалог между пользователем и ассистентом и последний вопрос пользователя.\n"
            "Твоя задача: переформулировать последний вопрос так, чтобы он был понятен без контекста диалога.\n"
            "Замени местоимения (он, она, оно, этот, тот) на конкретные сущности из истории.\n"
            "Если вопрос уже самодостаточен, верни его без изменений.\n"
            "Выведи только переформулированный вопрос. Никаких пояснений.\n\n"
            "Пример:\n"
            "История: Пользователь: \"Какие условия по карте Альфа?\" Ассистент: \"Ставка 20%...\"\n"
            "Вопрос: \"А есть кэшбэк?\"\n"
            "Ответ: \"Есть ли кэшбэк по карте Альфа?\"\n\n"
            f"История диалога:\n{history_str}\n\n"
            f"Последний вопрос: {query}\n\n"
            f"Переформулированный вопрос:"
        )
        
        inputs = self.tokenizer(prompt_template, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
        if not generated_text:
            logger.warning("Модель вернула пустую строку, используем оригинал")
            return query
            
        logger.info(f"Оригинал: '{query}', переформулировано как '{generated_text}'")
        return generated_text