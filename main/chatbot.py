from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain.prompts import PromptTemplate
except ImportError:
    from langchain_core.prompts import PromptTemplate
try:
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    from langchain_core.messages import HumanMessage, SystemMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from pathlib import Path

from main.vector_store import QdrantVectorStore
from main.bm25_retriever import BM25Retriever
from main.reranker import DocumentReranker
from main.api_reranker import ApiDocumentReranker
from utils.text_processor import VietnameseTextProcessor
from utils.google_search import GoogleSearchTool
from utils.question_refiner import VietnameseLegalQuestionRefiner
from utils.query_decomposition import decompose_query, should_decompose
from utils.retrieval_scoring import (
    apply_law_shortlist_filter,
    apply_two_stage_within_law_rerank,
    apply_within_law_rescore,
    fuse_hybrid_rrf,
    fuse_multi_query_rrf,
)
from utils.submission_formatter import (
    article_label,
    canonical_law_id,
    format_law_title,
    get_mapping_title,
    load_law_title_mapping,
)
from config import Config

class VietnameseLegalRAG:
    """Vietnamese Legal RAG System"""
    
    def __init__(self):
        self.vector_store = None
        self.bm25_retriever = None
        self.reranker = None
        self.llm = None
        self.text_processor = VietnameseTextProcessor()
        self.google_search = GoogleSearchTool()
        self.question_refiner = VietnameseLegalQuestionRefiner()
        self._corpus_lookup = None
        self._law_title_mapping = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG components"""
        try:
            # Initialize LLM if configured
            if Config.USE_LOCAL_LLM and Config.LLM_BACKEND == "ollama":
                print(f"✅ Chat LLM via Ollama ({Config.OLLAMA_MODEL}) — no local weight load")
                self.llm = None
            elif Config.USE_LOCAL_LLM:
                # BYPASS LLM LOADING: Do not load the LLM to save 10GB RAM during pipeline execution
                print("⚠️ LLM initialization skipped (Bypass mode to save RAM)")
                self.llm = None
            elif Config.GOOGLE_API_KEY:
                print("Loading Google Gemini API...")
                self.llm = ChatGoogleGenerativeAI(
                    model=Config.MODEL_GEN,
                    google_api_key=Config.GOOGLE_API_KEY,
                    temperature=0.1
                )
                print("✅ Google Gemini LLM initialized")
            else:
                print("⚠️ Warning: No LLM configured")
            
            # Initialize vector store
            self.vector_store = QdrantVectorStore()
            
            # Initialize BM25 retriever
            self.bm25_retriever = BM25Retriever()
            
            # Initialize reranker if enabled
            if Config.ENABLE_RERANKING:
                self.reranker = self._make_reranker()
            else:
                print("Reranking disabled in configuration")
            
        except Exception as e:
            print(f"Error initializing RAG components: {e}")

    def _make_reranker(self):
        if Config.RERANKER_BACKEND == "api":
            print(f"Using API reranker: {Config.FPT_RERANK_MODEL} @ {Config.FPT_RERANK_BASE_URL}")
            return ApiDocumentReranker()
        return DocumentReranker(model_name=Config.RERANKER_MODEL)

    def set_reranker_backend(self, backend: str) -> None:
        """Swap reranker implementation (local vs api) without reloading indices."""
        Config.RERANKER_BACKEND = backend
        if Config.ENABLE_RERANKING:
            self.reranker = self._make_reranker()
    
    def _initialize_qwen_model(self):
        """Initialize Qwen3-4B model for local inference"""
        try:
            model_name = Config.MODEL_GEN
            print(f"Loading tokenizer from {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"Loading model from {model_name}...")
            # Use float16 for MPS to save memory and improve speed
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16 if device == "mps" else torch.float32
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Wrap in a simple class to mimic LLM interface
            class QwenLLMWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.device = device
                
                def invoke(self, messages):
                    """Generate response from messages"""
                    # Extract text from HumanMessage or SystemMessage
                    if isinstance(messages, list):
                        prompt_text = ""
                        for msg in messages:
                            if isinstance(msg, HumanMessage):
                                prompt_text = msg.content
                            elif isinstance(msg, SystemMessage):
                                prompt_text = msg.content
                    else:
                        prompt_text = str(messages)
                    
                    # Apply chat template
                    messages_for_model = [{"role": "user", "content": prompt_text}]
                    text = self.tokenizer.apply_chat_template(
                        messages_for_model,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    # Generate response
                    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **model_inputs,
                            max_new_tokens=1024,
                            temperature=0.1,
                            top_p=0.9,
                            do_sample=False
                        )
                    
                    response_text = self.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )[0]
                    
                    # Create response object with .content attribute
                    class Response:
                        def __init__(self, content):
                            self.content = content
                    
                    return Response(response_text)
            
            return QwenLLMWrapper(model, tokenizer)
        
        except Exception as e:
            print(f"Error initializing Qwen model: {e}")
            raise
    
    def setup_indices(self, documents: List[Dict[str, Any]], force_rebuild: bool = False):
        """Setup both vector and BM25 indices"""
        print("Setting up RAG indices...")
        
        try:
            # Setup vector store
            if self.vector_store:
                # Check if we need to create collection
                try:
                    # First, do a simple existence check
                    collections = self.vector_store.client.get_collections().collections
                    collection_exists = any(col.name == self.vector_store.collection_name for col in collections)
                    print(f"Collection existence check: {collection_exists}")
                    
                    if collection_exists:
                        # Collection exists, try to get detailed info
                        try:
                            collection_info = self.vector_store.get_collection_info()
                            has_documents = collection_info.get('points_count', 0) > 0
                            
                            if force_rebuild:
                                print("Force rebuild requested - recreating vector store...")
                                if Config.EMBEDDING_MODEL == "bkai-foundation-models/vietnamese-bi-encoder":
                                    self.vector_store.create_collection(force_recreate=True, vector_size=768)
                                else:
                                    self.vector_store.create_collection(force_recreate=True)
                                self.vector_store.add_documents(documents)
                            elif not has_documents:
                                print("Collection exists but is empty - adding documents...")
                                self.vector_store.add_documents(documents)
                            else:
                                print(f"Vector store collection already exists with {collection_info.get('points_count', 0)} documents")
                        except Exception as info_e:
                            print(f"Could not get collection info: {info_e}")
                            if force_rebuild:
                                print("Force rebuild requested - recreating vector store...")
                                if Config.EMBEDDING_MODEL == "bkai-foundation-models/vietnamese-bi-encoder":
                                    self.vector_store.create_collection(force_recreate=True, vector_size=768)
                                else:
                                    self.vector_store.create_collection(force_recreate=True)
                                self.vector_store.add_documents(documents)
                            else:
                                print("Assuming collection has documents - skipping setup")
                    else:
                        # Collection doesn't exist, create it
                        print("Collection does not exist - creating new collection...")
                        if Config.EMBEDDING_MODEL == "bkai-foundation-models/vietnamese-bi-encoder":
                            self.vector_store.create_collection(force_recreate=True, vector_size=768)
                        else:
                            self.vector_store.create_collection(force_recreate=True)
                        self.vector_store.add_documents(documents)
                        
                except Exception as e:
                    print(f"Error during vector store setup: {e}")
                    print("Attempting to create collection...")
                    if Config.EMBEDDING_MODEL == "bkai-foundation-models/vietnamese-bi-encoder":
                        self.vector_store.create_collection(force_recreate=True, vector_size=768)
                    else:
                        self.vector_store.create_collection(force_recreate=True)
                    self.vector_store.add_documents(documents)
            
            # Setup BM25 index
            if self.bm25_retriever:
                # Try to load existing index
                if not self.bm25_retriever.load_index() or force_rebuild:
                    self.bm25_retriever.build_index(documents)
                    self.bm25_retriever.save_index()
                else:
                    print("BM25 index loaded from file")
            
            print("RAG indices setup completed")
            
        except Exception as e:
            print(f"Error setting up indices: {e}")
            raise
    
    def _hybrid_retrieve_pool(
        self,
        query: str,
        bm25_top_k: int,
        vector_top_k: int,
    ) -> List[Dict[str, Any]]:
        bm25_results = self.bm25_retriever.get_relevant_documents(query, top_k=bm25_top_k)
        vector_results = self.vector_store.search_similar_documents(query, top_k=vector_top_k)

        if Config.HYBRID_FUSION == "rrf":
            return fuse_hybrid_rrf(
                bm25_results,
                vector_results,
                rrf_k=Config.HYBRID_RRF_K,
            )

        all_docs: dict[str, Dict[str, Any]] = {}
        for doc in bm25_results:
            doc_id = doc.get("id", "")
            if doc_id:
                all_docs[doc_id] = {
                    **doc,
                    "retrieval_method": "bm25",
                    "bm25_score": doc.get("score", 0),
                }
        for doc in vector_results:
            doc_id = doc.get("id", "")
            if doc_id:
                if doc_id in all_docs:
                    all_docs[doc_id]["retrieval_method"] = "hybrid"
                    all_docs[doc_id]["vector_score"] = doc.get("score", 0)
                    all_docs[doc_id]["score"] = max(
                        all_docs[doc_id].get("bm25_score", 0),
                        doc.get("score", 0),
                    )
                else:
                    all_docs[doc_id] = {
                        **doc,
                        "retrieval_method": "vector",
                        "vector_score": doc.get("score", 0),
                    }
        retrieved_docs = list(all_docs.values())
        retrieved_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
        return retrieved_docs

    def retrieve_documents(self, query: str, use_hybrid: bool = True, use_reranking: bool = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using hybrid approach with optional reranking"""
        retrieved_docs = []
        
        # Use config default if not specified
        if use_reranking is None:
            use_reranking = Config.ENABLE_RERANKING
        
        # Adjust retrieval counts if reranking is enabled
        bm25_top_k = Config.RERANK_BEFORE_RETRIEVAL_TOP_K if use_reranking else Config.BM25_TOP_K
        vector_top_k = Config.RERANK_BEFORE_RETRIEVAL_TOP_K if use_reranking else Config.TOP_K_RETRIEVAL
        
        try:
            if use_hybrid and self.bm25_retriever and self.vector_store:
                sub_queries = decompose_query(query) if should_decompose(query) else [query]
                if len(sub_queries) > 1:
                    pools = [
                        self._hybrid_retrieve_pool(sub_q, bm25_top_k, vector_top_k)
                        for sub_q in sub_queries
                    ]
                    retrieved_docs = fuse_multi_query_rrf(pools, rrf_k=Config.HYBRID_RRF_K)
                else:
                    retrieved_docs = self._hybrid_retrieve_pool(query, bm25_top_k, vector_top_k)
                
            elif self.vector_store:
                # Vector search only
                retrieved_docs = self.vector_store.search_similar_documents(query, top_k=vector_top_k)
                
            elif self.bm25_retriever:
                # BM25 only
                retrieved_docs = self.bm25_retriever.get_relevant_documents(query, top_k=bm25_top_k)
            
            # Improved similarity filtering logic
            if retrieved_docs:
                if Config.ENABLE_LAW_SHORTLIST and use_reranking:
                    retrieved_docs = apply_law_shortlist_filter(
                        retrieved_docs,
                        top_laws=Config.LAW_SHORTLIST_TOP_K,
                        min_docs=Config.LAW_SHORTLIST_MIN_DOCS,
                    )

                # Apply reranking FIRST if enabled (before similarity filtering)
                if use_reranking and self.reranker and retrieved_docs:
                    print(f"Applying reranking to {len(retrieved_docs)} documents...")
                    
                    if Config.USE_SCORE_FUSION:
                        # Use score fusion for better results
                        retrieved_docs = self.reranker.rerank_with_fusion(
                            query, 
                            retrieved_docs, 
                            alpha=Config.RERANKER_FUSION_ALPHA,
                            top_k=Config.RERANKER_TOP_K
                        )
                    else:
                        # Use pure reranker scores
                        retrieved_docs = self.reranker.rerank_documents(
                            query, 
                            retrieved_docs, 
                            top_k=Config.RERANKER_TOP_K
                        )
                    if Config.ENABLE_WITHIN_LAW_RESCORE:
                        retrieved_docs = apply_within_law_rescore(query, retrieved_docs)
                    if Config.ENABLE_TWO_STAGE_WITHIN_LAW_RERANK and self.reranker:
                        retrieved_docs = apply_two_stage_within_law_rerank(
                            self.reranker,
                            query,
                            retrieved_docs,
                            top_laws=Config.WITHIN_LAW_RERANK_TOP_LAWS,
                            within_weight=Config.WITHIN_LAW_RERANK_WEIGHT,
                        )
                    # Dynamic Thresholding for F2-Macro optimization
                    if retrieved_docs:
                        max_score = retrieved_docs[0].get('score', 0)
                        threshold = max(
                            max_score * Config.RERANK_SCORE_THRESHOLD_RATIO,
                            0.5,
                        )
                        
                        filtered_docs = [doc for doc in retrieved_docs if doc.get('score', 0) >= threshold]
                        
                        # Fallback to Top-1 if threshold is too strict
                        if not filtered_docs:
                            filtered_docs = [retrieved_docs[0]]
                            
                        filtered_docs = filtered_docs[:Config.RERANK_MAX_OUTPUT_DOCS]
                            
                        retrieved_docs = filtered_docs

                    print(f"Reranking & Dynamic Thresholding completed, returning {len(retrieved_docs)} documents")
                    print([(retrieved_doc['id'], retrieved_doc['score']) for retrieved_doc in retrieved_docs])
                    return retrieved_docs
                
                # Check for high-quality documents first (if no reranking)
                high_quality_docs = []
                moderate_quality_docs = []
                
                for doc in retrieved_docs:
                    score = doc.get('score', 0)
                    if score >= Config.SIMILARITY_THRESHOLD:
                        high_quality_docs.append(doc)
                    elif score >= Config.MIN_SIMILARITY_FOR_LEGAL_DOCS:
                        moderate_quality_docs.append(doc)
                
                # Return high quality docs if available
                if high_quality_docs:
                    print(f"Retrieved {len(high_quality_docs)} high-quality documents")
                    print([(high_quality_doc['id'], high_quality_doc['score']) for high_quality_doc in high_quality_docs])
                    return high_quality_docs[:Config.TOP_K_RETRIEVAL]
                
                # Return moderate quality docs if no high quality ones
                elif moderate_quality_docs:
                    print(f"Retrieved {len(moderate_quality_docs)} moderate-quality documents")
                    print([(moderate_quality_doc['id'], moderate_quality_doc['score']) for moderate_quality_doc in moderate_quality_docs])
                    return moderate_quality_docs[:Config.TOP_K_RETRIEVAL]
                
                else:
                    print("No documents found with sufficient similarity scores")
                    # Fallback: return best available documents anyway (with lower threshold)
                    if retrieved_docs:
                        print(f"Fallback: returning top {min(5, len(retrieved_docs))} documents with best scores")
                        # Sort by score and return best ones
                        retrieved_docs.sort(key=lambda x: x.get('score', 0), reverse=True)
                        fallback_docs = retrieved_docs[:min(5, len(retrieved_docs))]
                        print([(fallback_doc['id'], fallback_doc['score']) for fallback_doc in fallback_docs])
                        return fallback_docs
                    return []
            else:
                # No documents retrieved
                return []
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as context for LLM"""
        if not documents:
            return "Không có tài liệu pháp luật liên quan được tìm thấy."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Không có tiêu đề')
            content = doc.get('content', '')
            doc_id = doc.get('id', '')
            metadata = doc.get('metadata', {})
            law_id = metadata.get('law_id', '')
            article_id = metadata.get('article_id', '')
            
            # Limit content length
            if len(content) > 500:
                content = content[:500] + "..."
            
            # Format law and article information
            law_info = f"Luật: {law_id}" if law_id else ""
            article_info = f"Điều {article_id}" if article_id else f"ID: {doc_id}"
            
            context_part = f"""
Tài liệu {i}:
{law_info}
{article_info}: {title}
Nội dung: {content}
"""
            context_parts.append(context_part.strip())
        
        return "\n\n".join(context_parts)

    def _get_law_title_mapping(self) -> dict[str, str]:
        if self._law_title_mapping is None:
            self._law_title_mapping = load_law_title_mapping(
                Path(Config.LAW_TITLE_MAPPING_PATH)
            )
        return self._law_title_mapping

    def _get_corpus_lookup(self) -> dict:
        if self._corpus_lookup is None:
            from utils.qa_answer_generator import build_corpus_lookup

            corpus_path = Path(Config.CORPUS_PATH)
            if not corpus_path.exists():
                raise FileNotFoundError(f"Corpus not found: {corpus_path}")
            print(f"Loading corpus lookup from {corpus_path}...")
            self._corpus_lookup = build_corpus_lookup(corpus_path)
        return self._corpus_lookup

    def _article_refs_from_documents(self, documents: List[Dict[str, Any]]) -> list[str]:
        mapping = self._get_law_title_mapping()
        refs: list[str] = []
        seen: set[str] = set()
        for doc in documents:
            metadata = doc.get("metadata", {})
            law_id = canonical_law_id(metadata.get("law_id", ""))
            if not law_id:
                continue
            raw_title = get_mapping_title(mapping, law_id)
            law_title = format_law_title(law_id, raw_title)
            label = article_label(metadata.get("title", ""), metadata.get("article_id", ""))
            if not label:
                continue
            ref = f"{law_id}|{law_title}|{label}"
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
        return refs

    def generate_answer(
        self,
        query: str,
        context: str,
        is_fallback: bool = False,
        retrieved_docs: List[Dict[str, Any]] | None = None,
        chat_history: List[Dict[str, str]] | None = None,
    ) -> str:
        """Generate answer via unified Ollama chat (history + optional RAG blocks)."""
        blocks: list[dict[str, str]] | None = None
        if retrieved_docs and not is_fallback:
            from utils.qa_answer_generator import build_context_blocks

            article_refs = self._article_refs_from_documents(retrieved_docs)
            if article_refs:
                blocks = build_context_blocks(
                    article_refs[: Config.CHAT_MAX_ARTICLES],
                    self._get_corpus_lookup(),
                    Config.CHAT_MAX_ARTICLES,
                    Config.CHAT_MAX_CHARS_PER_ARTICLE,
                    question=query,
                )

        if Config.USE_LOCAL_LLM and Config.LLM_BACKEND == "ollama":
            from utils.chat_llm import generate_unified_sme_reply

            try:
                return generate_unified_sme_reply(
                    query,
                    history=chat_history,
                    blocks=blocks,
                    extra_context=context if is_fallback or not blocks else "",
                    model=Config.OLLAMA_MODEL,
                    max_new_tokens=Config.CHAT_MAX_NEW_TOKENS,
                )
            except Exception as exc:
                print(f"Ollama unified reply failed: {exc}")

        if self.llm and not is_fallback:
            try:
                prompt = f"{Config.SYSTEM_PROMPT}\n\nCâu hỏi: {query}\n\nTài liệu tham khảo:\n{context}"
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return response.content if hasattr(response, "content") else str(response)
            except Exception as exc:
                print(f"LLM generation failed: {exc}")

        return "Theo các quy định pháp luật liên quan:\n\n" + context
    
    def _is_negative_response(self, response: str) -> bool:
        """Check if the response is a negative/unable to answer response"""
        negative_indicators = [
            "không thể trả lời",
            "không tìm thấy",
            "không có thông tin",
            "xin lỗi",
            "không thể tìm thấy",
            "không có dữ liệu",
            "không rõ",
            "không biết",
            "không đủ thông tin",
            "thiếu thông tin",
            "không có trong",
            "ngoài phạm vi",
            # Add the specific pattern mentioned by user
            "không có đủ thông tin trong tài liệu tham khảo được cung cấp để trả lời trực tiếp câu hỏi này",
            "cần tham khảo thêm các văn bản pháp luật khác",
            "tìm kiếm thông tin chuyên sâu hơn về",
            "tài liệu tham khảo không chứa thông tin đầy đủ"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in negative_indicators)

    def _ollama_chat_enabled(self) -> bool:
        return bool(Config.USE_LOCAL_LLM and Config.LLM_BACKEND == "ollama")

    def _chat_result(
        self,
        answer: str,
        original_query: str,
        *,
        chat_mode: bool = False,
        **extra: Any,
    ) -> Dict[str, Any]:
        base = {
            "answer": answer,
            "retrieved_documents": [],
            "fallback_used": False,
            "context": "",
            "search_results": [],
            "search_results_html": "",
            "original_question": original_query,
            "refined_question": original_query,
            "question_refinement": None,
            "chat_mode": chat_mode,
        }
        base.update(extra)
        return base

    def answer_question(
        self,
        query: str,
        use_fallback: bool = True,
        refine_question: bool = True,
        chat_history: List[Dict[str, str]] | None = None,
    ) -> Dict[str, Any]:
        """Single-LLM SME assistant: history + optional RAG context."""
        print(f"Processing question: {query}")
        original_query = query
        history = chat_history or []

        # Refine the question if enabled
        refinement_result = None

        if (
            refine_question
            and Config.ENABLE_QUESTION_REFINEMENT
            and self.question_refiner
            and len(query.strip()) >= 12
        ):
            print("🔧 Refining question for better search accuracy...")
            refinement_result = self.question_refiner.refine_question(query, use_llm=Config.USE_LLM_FOR_REFINEMENT)
            
            if refinement_result["refined_question"] != query:
                refined_query = refinement_result["refined_question"]
                print(f"📝 Original: {query}")
                print(f"✨ Refined: {refined_query}")
                query = refined_query

        # Step 2: Retrieve relevant documents using refined query
        retrieved_docs = self.retrieve_documents(query)
        
        # Check if we have relevant documents
        if not retrieved_docs and Config.ENABLE_GOOGLE_SEARCH and use_fallback:
            print("No relevant legal documents found, using Google search fallback")
            
            # Use Google search as fallback
            search_results = self.google_search.search_legal_info(query)
            
            if search_results:
                fallback_context = self.google_search.format_search_results(search_results)
                
                # Generate answer with fallback context
                fallback_answer = self.generate_answer(
                    query, fallback_context, True, chat_history=history
                )
                
                return {
                    'answer': fallback_answer,
                    'retrieved_documents': [],
                    'fallback_used': True,
                    'search_results': search_results,
                    'context': fallback_context,
                    'search_results_html': self.google_search.format_search_results_for_display(search_results),
                    'original_question': original_query,
                    'refined_question': query,
                    'question_refinement': refinement_result,
                    'chat_mode': False,
                }
            else:
                no_doc_answer = self.generate_answer(
                    original_query, "", False, chat_history=history
                )
                return {
                    'answer': no_doc_answer,
                    'retrieved_documents': [],
                    'fallback_used': True,
                    'search_results': [],
                    'context': "",
                    'search_results_html': "",
                    'original_question': original_query,
                    'refined_question': query,
                    'question_refinement': refinement_result,
                    'chat_mode': False,
                }
        elif not retrieved_docs:
            no_doc_answer = self.generate_answer(
                original_query, "", False, chat_history=history
            )
            return {
                'answer': no_doc_answer,
                'retrieved_documents': [],
                'fallback_used': False,
                'context': "",
                'search_results': [],
                'search_results_html': "",
                'original_question': original_query,
                'refined_question': query,
                'question_refinement': refinement_result,
                'chat_mode': False,
            }

        # Format context
        context = self.format_context(retrieved_docs)

        # Generate answer
        answer = self.generate_answer(
            query, context, False, retrieved_docs=retrieved_docs, chat_history=history
        )

        # Check if the generated answer is negative and retry with Google search
        if self._is_negative_response(answer) and use_fallback:
            print("🔍 Detected insufficient information response, activating search tools...")
            
            # Inform user that search is being performed
            search_notification = f"\n\n*🔍 Đang tìm kiếm thông tin bổ sung để trả lời câu hỏi của bạn...*"
            
            # Try Google search if enabled
            if Config.ENABLE_GOOGLE_SEARCH:
                print("📡 Trying web search...")
                search_results = self.google_search.search_legal_info(query)
                
                if search_results:
                    # Generate enhanced response with web information
                    web_context = self.google_search.format_search_results(search_results)
                    combined_context = context + "\n\nThông tin bổ sung từ web:\n" + web_context
                    enhanced_answer = self.generate_answer(
                        query, combined_context, True,
                        retrieved_docs=retrieved_docs, chat_history=history,
                    )
                    
                    return {
                        'answer': enhanced_answer,
                        'retrieved_documents': retrieved_docs,
                        'fallback_used': True,
                        'context': combined_context,
                        'search_results': search_results,
                        'search_results_html': self.google_search.format_search_results_for_display(search_results),
                        'search_triggered': True,
                        'original_question': original_query,
                        'refined_question': query,
                        'question_refinement': refinement_result,
                        'chat_mode': False,
                    }
                else:
                    # Google search found nothing useful
                    return {
                        'answer': answer + "\n\n*⚠️ Tôi đã cố gắng tìm kiếm thêm thông tin trên web nhưng không tìm thấy kết quả phù hợp. Để có câu trả lời chính xác hơn, bạn có thể tham khảo ý kiến chuyên gia pháp lý.*",
                        'retrieved_documents': retrieved_docs,
                        'fallback_used': True,
                        'context': context,
                        'search_results': [],
                        'search_results_html': "",
                        'search_triggered': True,
                        'original_question': original_query,
                        'refined_question': query,
                        'question_refinement': refinement_result,
                        'chat_mode': False,
                    }
            else:
                # Google search disabled
                return {
                    'answer': answer + "\n\n*⚠️ Để có câu trả lời chính xác hơn, bạn có thể tham khảo ý kiến chuyên gia pháp lý.*",
                    'retrieved_documents': retrieved_docs,
                    'fallback_used': False,
                    'context': context,
                    'search_results': [],
                    'search_results_html': "",
                    'original_question': original_query,
                    'refined_question': query,
                    'question_refinement': refinement_result,
                    'chat_mode': False,
                }

        # Return successful result
        return {
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'fallback_used': False,
            'context': context,
            'search_results': [],
            'search_results_html': "",
            'original_question': original_query,
            'refined_question': query,
            'question_refinement': refinement_result,
            'chat_mode': False,
        }
    
    def _build_context_from_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents"""
        return self.format_context(documents)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of RAG system components"""
        status = {
            'llm_available': self.llm is not None or (
                Config.USE_LOCAL_LLM and Config.LLM_BACKEND == "ollama"
            ) or bool(Config.GOOGLE_API_KEY),
            'vector_store_available': self.vector_store is not None,
            'bm25_available': self.bm25_retriever is not None,
            'reranker_available': self.reranker is not None and self.reranker.model is not None,
            'reranking_enabled': Config.ENABLE_RERANKING,
            'google_api_configured': bool(Config.GOOGLE_API_KEY),
            'qdrant_configured': bool(Config.QDRANT_URL and Config.QDRANT_API_KEY)
        }
        
        # Get collection info if available
        if self.vector_store:
            try:
                status['vector_store_info'] = self.vector_store.get_collection_info()
            except:
                status['vector_store_info'] = {}
        
        # Get BM25 stats if available
        if self.bm25_retriever:
            status['bm25_stats'] = self.bm25_retriever.get_index_stats()
        
        # Get reranker info if available
        if self.reranker:
            status['reranker_info'] = self.reranker.get_model_info()
        
        return status 