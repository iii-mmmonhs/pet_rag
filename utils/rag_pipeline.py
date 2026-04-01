import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
from config import K_RETRIEVE, TOP_N

logger = logging.getLogger(__name__)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Ты - оператор службы поддержки банка. Твоя задача - дать только финальный ответ пользователю.\n\n"
         "Твоя задача — ответить на этот вопрос полно и развернуто, как если бы пользователь задал его впервые.\n\n"
         "1. Никогда не выводи текст внутри тегов <think>.\n"
         "2. Никогда не пиши фразы типа 'В предыдущем сообщении...', 'Вы спрашивали...', 'Исходя из вашего вопроса...'. "
         "Просто давай факт.\n"
         "3. Отвечай сразу по делу. Начинай с сути (например, 'Ставка составляет...', 'Для оформления нужно...').\n"
         "4. Если ответа нет в контексте, скажи об этом честно и кратко.\n\n"
         "Контекст из документации:\n{context}"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
    ]
)

def create_rag_chain(vectorstore, llm, documents=None, hybrid_weights=(0.5, 0.5)):
    """
    Пайплайн с гибридным поиском и реранкингом.
    
    1. Retrieval:
       - Semantic Search
       - Lexical Search (BM25 по ключевым словам) если переданы документы
       - Ensemble
    2. Compression:
       - Cross-Encoder Reranker для переупорядочивания топ-N документов 
    3. Generation:
       - LLM генерирует ответ на основе контекста
    """
        
    semantic_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": K_RETRIEVE}
    )
    
    bm25_retriever = None
    if documents is not None:
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = K_RETRIEVE
    else:
        logger.warning("documents не переданы, BM25 отключен")
    
    if bm25_retriever:
        hybrid_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=list(hybrid_weights)
        )
        base_retriever = hybrid_retriever
    else:
        base_retriever = semantic_retriever
    
    try:
        compressor = FlashrankRerank(top_n=TOP_N)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    except Exception as e:
        logger.error(f"Ошибка инициализации реранкера: {e}. Используется базовый ретривер.")
        compression_retriever = base_retriever
            
    question_answer_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    
    chain = create_retrieval_chain(
        compression_retriever,
        question_answer_chain
    )
    
    logger.info("Цепь создана")
    return chain