import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from typing import List

logger = logging.getLogger(__name__)

def load_and_split_pdf(file_path: str, embeddings):  
    """
    Загружает PDF-файл и разбивает на чанки с SemanticChunker.
    
    Args:
        file_path: Путь к PDF.
        embeddings: HuggingFaceEmbeddings
    
    Returns:
        List[Document]: Список Document, содержащих текст чанков 
                        и пару (источник, страница)
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    except Exception as e:
        logger.error(f"Ошибка при чтении PDF: {e}")
        raise
                
    logger.info(f"Загружен документ: {file_path}, {len(documents)} страниц")
    
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile" 
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Создано {len(chunks)} чанков")
    
    return chunks