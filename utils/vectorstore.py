import logging
import os
from langchain_community.vectorstores import FAISS
from config import LOAD_PATH

logger = logging.getLogger(__name__)


def build_vectorstore(chunks, embeddings):
    """
    Строит векторный индекс FAISS из списка документов и сохраняет
    """
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    vectorstore.save_local(LOAD_PATH)
    if vectorstore.index.ntotal == 0:
        raise ValueError("Индекс FAISS пуст")
    
    logger.info("Индекс готов")
    return vectorstore


def load_vectorstore(embeddings):
    """
    Загружает существующий векторный индекс FAISS
    """
    if not os.path.exists(LOAD_PATH):
        raise FileNotFoundError(
            f"Индекс FAISS не найден по пути {LOAD_PATH}."
        )
    
    vectorstore = FAISS.load_local(LOAD_PATH, embeddings)
    logger.info(f"Индекс загружен. Размер {vectorstore.index.ntotal}")
    
    return vectorstore