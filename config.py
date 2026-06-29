import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    # LLM Configuration
    USE_LOCAL_LLM = True  # QA/chat via Ollama (not HuggingFace transformers on Mac)

    # Ollama — chat/QA (qwen3:4b-instruct, non-thinking)
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # ollama | local_hf | gemini
    CHAT_MAX_ARTICLES = int(os.getenv("CHAT_MAX_ARTICLES", "3"))
    CHAT_MAX_CHARS_PER_ARTICLE = int(os.getenv("CHAT_MAX_CHARS_PER_ARTICLE", "900"))
    CHAT_MAX_NEW_TOKENS = int(os.getenv("CHAT_MAX_NEW_TOKENS", "512"))
    CHAT_INCLUDE_DISCLAIMER = os.getenv("CHAT_INCLUDE_DISCLAIMER", "true").lower() in {
        "1", "true", "yes",
    }

    # If USE_LOCAL_LLM is True, default to Ollama qwen3:4b-instruct.
    # Set LLM_BACKEND=local_hf + MODEL_GEN=<hf_id> only if loading via transformers.
    if USE_LOCAL_LLM:
        MODEL_GEN = os.getenv("MODEL_GEN", OLLAMA_MODEL)
        MODEL_REFINE = os.getenv("MODEL_REFINE", OLLAMA_MODEL)
        GOOGLE_API_KEY = None
    else:
        # Google API Configuration (fallback)
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        MODEL_GEN = "gemini-2.0-flash"
        MODEL_REFINE = "gemini-2.0-flash"

    # QDrant Configuration
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    
    # Embedding configuration
    # EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
    # "local" (SentenceTransformer) or "api" (remote GPU embedder).
    EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
    EMBED_API_BASE_URL = os.getenv("EMBED_API_BASE_URL", "http://localhost:8001")
    EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")

    COLLECTION_NAME = "final_vietnamese_legal_corpus" if EMBEDDING_MODEL == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" else "bkai_biencoder_vietnamese_legal_corpus"

    USE_MERGED_CORPUS = os.getenv("USE_MERGED_CORPUS", "").lower() in {"1", "true", "yes"}
    if USE_MERGED_CORPUS:
        COLLECTION_NAME = f"{COLLECTION_NAME}_merged"

    # Text Processing Configuration
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    # Data Paths
    DATA_DIR = "data"
    CORPUS_PATH = (
        "data/corpus/legal_corpus_merged.json"
        if USE_MERGED_CORPUS
        else "data/corpus/legal_corpus.json"
    )
    LAW_TITLE_MAPPING_PATH = (
        "data/law_id_to_title_merged.json"
        if USE_MERGED_CORPUS
        else "data/law_id_to_title.json"
    )
    BM25_INDEX_FILE = "index/bm25_index_merged.pkl" if USE_MERGED_CORPUS else "index/bm25_index.pkl"
    STOPWORDS_PATH = "data/utils/stopwords.txt"

    # RAG Configuration
    TOP_K_RETRIEVAL = 20
    BM25_TOP_K = 20
    BM25_B = 0.65
    BM25_K1 = 1.2
    SIMILARITY_THRESHOLD = 0.25

    # Reranker Configuration
    ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "1").strip().lower() in ("1", "true", "yes", "on")
    RERANKER_MODEL = "AITeamVN/Vietnamese_Reranker"
    RERANKER_MAX_LENGTH = 768
    RERANKER_BATCH_SIZE = 16
    # "local" (CrossEncoder) or "api" (FPT Cloud bge-reranker-v2-m3).
    RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "local")
    FPT_RERANK_API_KEY = os.getenv("FPT_RERANK_API_KEY", "")
    FPT_RERANK_BASE_URL = os.getenv("FPT_RERANK_BASE_URL", "https://mkp-api.fptcloud.jp")
    FPT_RERANK_MODEL = os.getenv("FPT_RERANK_MODEL", "bge-reranker-v2-m3")
    RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")  # cpu safer on Mac UI; use mps/cuda for batch
    RERANK_BEFORE_RETRIEVAL_TOP_K = 15
    # Wide pool for three-stage retrieval experiments (stage-1 recall).
    RERANKER_TOP_K_WIDE = 20
    RERANK_BEFORE_RETRIEVAL_TOP_K_WIDE = 50
    USE_SCORE_FUSION = True
    RERANKER_FUSION_ALPHA = 0.92
    ENABLE_WITHIN_LAW_RESCORE = False
    RERANK_SCORE_THRESHOLD_RATIO = 0.78
    RERANK_MAX_OUTPUT_DOCS = 6

    # Hybrid fusion: "max" (legacy) or "rrf" (reciprocal rank fusion).
    HYBRID_FUSION = os.getenv("HYBRID_FUSION", "max")
    HYBRID_RRF_K = 60

    if os.getenv("USE_WIDE_RETRIEVAL_POOL", "").lower() in {"1", "true", "yes"}:
        RERANKER_TOP_K = RERANKER_TOP_K_WIDE
        RERANK_BEFORE_RETRIEVAL_TOP_K = RERANK_BEFORE_RETRIEVAL_TOP_K_WIDE

    # Two-stage rerank: global fusion then within-top-law rerank (no forced top-1).
    ENABLE_TWO_STAGE_WITHIN_LAW_RERANK = False
    WITHIN_LAW_RERANK_TOP_LAWS = 3
    WITHIN_LAW_RERANK_WEIGHT = 0.35

    # Law shortlist: restrict rerank pool to top-N laws from hybrid pre-scores.
    ENABLE_LAW_SHORTLIST = False
    LAW_SHORTLIST_TOP_K = 5
    LAW_SHORTLIST_MIN_DOCS = 8

    # P2: rule-based query decomposition for long / multi-clause questions.
    ENABLE_QUERY_DECOMPOSITION = os.getenv("ENABLE_QUERY_DECOMPOSITION", "").lower() in {
        "1",
        "true",
        "yes",
    }
    QUERY_DECOMPOSE_MIN_WORDS = int(os.getenv("QUERY_DECOMPOSE_MIN_WORDS", "45"))
    QUERY_DECOMPOSE_MAX_SUBQUERIES = int(os.getenv("QUERY_DECOMPOSE_MAX_SUBQUERIES", "3"))

    # Google Search Configuration
    ENABLE_GOOGLE_SEARCH = True
    GOOGLE_SEARCH_RESULTS_COUNT = 10
    MIN_SIMILARITY_FOR_LEGAL_DOCS = 0.15

    # Question Refinement Configuration
    ENABLE_QUESTION_REFINEMENT = True
    USE_LLM_FOR_REFINEMENT = True

    # Advanced LLM Refinement Settings
    ENABLE_CHAIN_OF_THOUGHT = True
    ENABLE_ITERATIVE_REFINEMENT = True
    ENABLE_LLM_VALIDATION = True
    MAX_REFINEMENT_ITERATIONS = 3
    MIN_CONFIDENCE_SCORE = 0.7

    # UI Display Settings - Control what information to show in responses
    SHOW_REFINEMENT_INFO = False  # 🔧 Câu hỏi đã được tối ưu
    SHOW_SEARCH_TRIGGER_INFO = False  # 🔍➡️🌐 Tự động tìm kiếm
    SHOW_SOURCE_INFO = False  # 📚 Dựa trên X tài liệu, 🌐 Thông tin từ web
    SHOW_LEGAL_DISCLAIMER = False  # Lưu ý về tìm chuyên gia pháp lý

    # Chat — single Ollama conversation (history + optional RAG context)
    USE_LLM_CHAT_ROUTING = False
    ENABLE_LEGAL_DOMAIN_FILTER = False

    # System Prompt (Gemini / fallback path)
    SYSTEM_PROMPT = """Bạn là Lexi, trợ lý pháp lý cho doanh nghiệp SME tại Việt Nam. Nhiệm vụ của bạn là cung cấp các câu trả lời chính xác và dễ hiểu cho các câu hỏi pháp lý, dựa trên các tài liệu luật được cung cấp.

Khi trả lời:
1.  **Chỉ sử dụng thông tin** trực tiếp từ các điều luật và văn bản được cung cấp trong phần "Tài liệu tham khảo". Tuyệt đối không suy diễn hoặc thêm thông tin bên ngoài.
2.  **Trích dẫn chính xác** tên luật (ví dụ: Luật Doanh nghiệp 2020), số hiệu văn bản (nếu có), và điều khoản cụ thể (ví dụ: Điều 3, Khoản 2).
3.  **Giải thích rõ ràng, ngắn gọn và khách quan**, tập trung vào việc làm sáng tỏ nội dung của điều luật liên quan đến câu hỏi.
4.  **Nếu tài liệu tham khảo không chứa thông tin đầy đủ hoặc trực tiếp để trả lời câu hỏi**, hãy thông báo rõ ràng rằng "Không có đủ thông tin trong tài liệu tham khảo được cung cấp để trả lời trực tiếp câu hỏi này."
5.  **Trình bày bằng tiếng Việt chuẩn xác.**

Tài liệu tham khảo:
{context}

Câu hỏi: {question}

Trả lời:"""

    # Fallback System Prompt for Google Search
    FALLBACK_SYSTEM_PROMPT = """Bạn là trợ lý pháp lý thông minh chuyên sâu về luật pháp Việt Nam. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp.

Khi trả lời:
1.  **Tóm tắt và trình bày thông tin liên quan** một cách tự nhiên và mạch lạc.
2.  **Cung cấp các liên kết (URLs)** của các nguồn đã được tham khảo ở cuối câu trả lời để người dùng có thể kiểm tra thêm.
3.  **Giải thích rõ ràng và dễ hiểu**.
4.  **Trình bày bằng tiếng Việt chuẩn xác.**

Thông tin tham khảo:
{context}

Câu hỏi: {question}

Trả lời:"""
