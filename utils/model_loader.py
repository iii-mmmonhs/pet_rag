import os
import logging
from langchain_openai import ChatOpenAI
from config import GENERATION_MODEL_NAME, BASE_URL, TEMPERATURE, MAX_TOKENS

logger = logging.getLogger(__name__)

API_TOKEN = os.getenv("API_TOKEN")

def load_llm():
    """
    Инициализирует и возвращает экземпляр LLM.
    """
    if not API_TOKEN:
        raise ValueError("API_TOKEN нет в секретах окружения")

    llm = ChatOpenAI(
        model=GENERATION_MODEL_NAME,
        openai_api_key=API_TOKEN,
        openai_api_base=BASE_URL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )
    return llm