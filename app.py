import gradio as gr
import logging
import os
import pickle
from utils.model_loader import load_llm
from utils.pdf_parser import load_and_split_pdf
from utils.vectorstore import build_vectorstore, load_vectorstore
from utils.rag_pipeline import create_rag_chain
from utils.query_rewriter import QueryRewriter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from config import (
    PDF_PATH, EMBEDDING_MODEL_NAME, LOAD_PATH, MEMORY_WINDOW, 
    SEMANTIC_WEIGHT, LEXICAL_WEIGHT, MAX_CONTEXT_SNIPPET_LENGTH, DEVICE, DOCS_CACHE_PATH
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s:%(funcName)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class RAGBot:
    """
    Управление пайплайном.
    
    Отвечает за создание эмбеддингов, векторного хранилища, 
    загрузку LLM и обработку пользовательских запросов с учетом истории диалога.
    Поддерживает кэширование документов и векторного индекса для ускорения запуска.
    """
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.chain = None
        self.documents = None
        self.rewriter = None

    def setup(self):
        """
        Инициализирует все компоненты:
        загружает кэш или обрабатывает PDF, строит индекс и сохраняет,
        инициализирует LLM, цепь и QueryRewriter.
        """
        logger.info("Инициализация RAG")
        
        model_kwargs = {'device': DEVICE}
        encode_kwargs = {'normalize_embeddings': True}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        if os.path.exists(LOAD_PATH) and os.path.exists(DOCS_CACHE_PATH):
            self.vectorstore = load_vectorstore(self.embeddings)

            with open(DOCS_CACHE_PATH, "rb") as f:
                self.documents = pickle.load(f)
        else:
            self.documents = load_and_split_pdf(file_path=PDF_PATH, embeddings=self.embeddings)
            self.vectorstore = build_vectorstore(self.documents, self.embeddings)
            
            os.makedirs(LOAD_PATH, exist_ok=True)
            with open(DOCS_CACHE_PATH, "wb") as f:
                pickle.dump(self.documents, f)
                
        llm = load_llm()
        self.chain = create_rag_chain(
            vectorstore=self.vectorstore,
            llm=llm,
            documents=self.documents,
            hybrid_weights=(SEMANTIC_WEIGHT, LEXICAL_WEIGHT)
        )

        try:
            self.rewriter = QueryRewriter()
        except Exception as e:
            logger.error(f"Ошибка при инициализации рерайтера {e}")
            self.rewriter = None

        logger.info("RAG готов")

    def answer_question(self, question: str, gradio_history: list) -> tuple:
        """
        Обрабатывает вопрос пользователя, обогащает его контекстом истории и источниками.
        
        Args:
            question: Текущий вопрос пользователя.
            gradio_history: История сообщений из интерфейса Gradio. 
        
        Returns:
            tuple: Кортеж (текст ответа, список источников).
        """
        try:
            chat_history_context = []
            for msg in gradio_history[-MEMORY_WINDOW * 2:]:
                role = msg.get("role")
                content = msg.get("content")
                    
                if not content: continue
                    
                if role == "user":
                    chat_history_context.append(HumanMessage(content=content))
                elif role == "assistant":
                    chat_history_context.append(AIMessage(content=content))

            search_query = question
            if self.rewriter and len(gradio_history) > 0:
                try:
                    history_texts = [msg.content for msg in chat_history_context]
                    search_query = self.rewriter.rewrite(question, chat_history_context)
                    logger.info(f"Исходный вопрос: '{question}', переформулирован: '{search_query}'")
                except Exception as e:
                    logger.warning(f"Ошибка рерайтера: {e}, используется оригинальный запрос")

            inputs = {
                "input": search_query,
                "chat_history": chat_history_context
            }
            
            response = self.chain.invoke(inputs)
            
            answer = response.get("answer", "Извините, я не могу найти ответ.")
            context_docs = response.get("context", [])
            
            return answer, context_docs

        except Exception as e:
            logger.error(f"Ошибка: {e}")
            return f"Извините, произошла ошибка при генерации ответа", []

bot = RAGBot()

def respond(message, chat_history):
    """
    Интеграция с интерфейсом.
        
    Args:
        message: Вопрос пользователя.
        chat_history: История диалога.
        
    Yields:
        str: Промежуточное сообщение о статусе, финальный ответ с источниками.
    """
    yield "Ищу информацию в источниках..."
    
    answer, context_docs = bot.answer_question(message, chat_history)
    
    formatted_context = ""
    if context_docs:
        formatted_context = "\n\n Источники: \n"
        sources = []
        for doc in context_docs:
            page = doc.metadata.get('page', 'не указана')
            snippet = doc.page_content[:300].replace('\n', ' ')
            sources.append(f"- Стр. {page}: \"{snippet}...\"")
        
        formatted_context += "\n".join(sources[:3])

    yield answer + formatted_context

if __name__ == "__main__":
    bot.setup()
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Служба поддержки банка")
        gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(height=600, label="Задайте Ваш вопрос"),
            examples=["Условия по карте 'Премиум'", "Мне не перезвонил курьер, что делать?"]
        )
        
    demo.launch(server_name="0.0.0.0", server_port=7860)