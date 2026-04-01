import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

DEVICE = "cpu"

EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
GENERATION_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
REWRITER_MODEL = "Qwen/Qwen3-0.6B"
BASE_URL = "https://router.huggingface.co/v1"

PDF_PATH = os.path.join(PROJECT_DIR, "data", "bank_faq_data.pdf")
SAVE_PATH = os.path.join(PROJECT_DIR, "embeddings", "index.faiss") 
LOAD_PATH = os.path.join(PROJECT_DIR, "embeddings", "index.faiss")
DOCS_CACHE_PATH = os.path.join(LOAD_PATH, "documents_cache.pkl")

TEMPERATURE = 0.3
MAX_TOKENS = 512
MEMORY_WINDOW = 5
MAX_CONTEXT_SNIPPET_LENGTH = 300

SEMANTIC_WEIGHT = 0.5
LEXICAL_WEIGHT = 0.5
K_RETRIEVE = 6
TOP_N = 3