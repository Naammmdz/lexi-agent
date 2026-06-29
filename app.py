import gradio as gr
import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import asyncio
import json
import os
import sys

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_APP_DIR, "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(_APP_DIR, ".env"))
# Retrieval stack aligned with cache_only_top1 (RRF wide + merged corpus).
os.environ.setdefault("USE_MERGED_CORPUS", "1")
os.environ.setdefault("HYBRID_FUSION", "rrf")
os.environ.setdefault("USE_WIDE_RETRIEVAL_POOL", "1")

from main.chatbot import VietnameseLegalRAG
from utils.data_loader import LegalDataLoader
from config import Config

class VietnameseLegalChatbot:
    """
    Vietnamese Legal Chatbot implementing the architecture described in the documentation.
    
    Core Components:
    1. Natural Language Understanding (NLU) / NLP Module - handled by RAG system
    2. Dialogue Manager - manages conversation flow and context
    3. Knowledge Base / Data Store - vector store + BM25 + external search
    4. Response Generation Module - LLM with contextual information
    """
    
    def __init__(self):
        # Core chatbot backend components
        self.rag_system: Optional[VietnameseLegalRAG] = None
        self.initialization_status = {
            "status": "initializing", 
            "message": "🚀 Đang khởi tạo hệ thống trợ lý pháp lý...",
            "progress": 0,
            "details": "Chuẩn bị khởi động..."
        }
        
        # Dialogue manager - conversation state management
        self.conversation_sessions = {}
        self.current_session_id = "default"
        
        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "fallback_queries": 0,
            "average_response_time": 0,
            "start_time": time.time()
        }
        
        # Initialize system in background
        self._initialize_system_async()
    
    def _initialize_system_async(self):
        """Initialize the RAG system asynchronously with detailed progress"""
        def initialize():
            try:
                # Phase 1: Data Loading
                self.initialization_status.update({
                    "status": "loading_data", 
                    "message": "📚 Đang tải dữ liệu pháp luật Việt Nam...",
                    "progress": 10,
                    "details": "Đọc tệp dữ liệu từ thư mục..."
                })
                time.sleep(1)  # Visual feedback
                
                # Initialize data loader
                data_loader = LegalDataLoader()
                
                self.initialization_status.update({
                    "progress": 20,
                    "details": "Xử lý và chuẩn bị tài liệu..."
                })

                # Initialize RAG first — skip loading 237k docs if indices already exist
                self.initialization_status.update({
                    "status": "initializing_rag",
                    "message": "🤖 Đang khởi tạo hệ thống RAG...",
                    "progress": 30,
                    "details": "Kết nối vector store và BM25...",
                })
                self.rag_system = VietnameseLegalRAG()

                self.initialization_status.update({
                    "status": "building_indices",
                    "message": "🔍 Đang xây dựng chỉ mục tìm kiếm...",
                    "progress": 60,
                    "details": "Kiểm tra chỉ mục hiện có...",
                })

                documents = []
                try:
                    collection_info = self.rag_system.vector_store.get_collection_info()
                    bm25_loaded = self.rag_system.bm25_retriever.load_index()
                    if not collection_info or not bm25_loaded:
                        documents = data_loader.prepare_documents_for_indexing()
                except Exception:
                    documents = data_loader.prepare_documents_for_indexing()

                if not documents and not self.rag_system.bm25_retriever.bm25:
                    self.initialization_status.update({
                        "status": "error", 
                        "message": "❌ Không thể tải dữ liệu pháp luật",
                        "progress": 0,
                        "details": "Kiểm tra thư mục data và tệp dữ liệu"
                    })
                    return

                doc_count = len(documents) or self.rag_system.bm25_retriever.get_index_stats().get(
                    "total_documents", 0
                )
                self.initialization_status.update({
                    "progress": 70,
                    "details": f"{'Đã tải' if documents else 'Dùng'} {doc_count:,} tài liệu",
                })

                # Setup indices only when missing
                try:
                    collection_info = self.rag_system.vector_store.get_collection_info()
                    bm25_loaded = bool(self.rag_system.bm25_retriever.bm25)
                    if not collection_info or not bm25_loaded:
                        self.initialization_status.update({
                            "message": "🏗️ Đang xây dựng chỉ mục lần đầu...",
                            "progress": 75,
                            "details": "Quá trình này có thể mất vài phút...",
                        })
                        self.rag_system.setup_indices(documents, force_rebuild=False)
                    else:
                        self.initialization_status.update({
                            "progress": 90,
                            "details": "Sử dụng chỉ mục có sẵn...",
                        })
                except Exception as e:
                    self.initialization_status.update({
                        "message": "🏗️ Đang xây dựng chỉ mục mới...",
                        "progress": 75,
                        "details": f"Xây dựng lại do lỗi: {str(e)[:50]}...",
                    })
                    if documents:
                        self.rag_system.setup_indices(documents, force_rebuild=False)
                
                # Phase 4: Final Validation
                self.initialization_status.update({
                    "progress": 95,
                    "details": "Kiểm tra tính toàn vẹn hệ thống..."
                })
                time.sleep(1)
                
                # Validate system
                system_status = self.rag_system.get_system_status()
                if not all([
                    system_status.get('llm_available'),
                    system_status.get('vector_store_available'),
                    system_status.get('bm25_available')
                ]):
                    raise Exception("Một số thành phần hệ thống không khả dụng")
                
                # Success
                self.initialization_status.update({
                    "status": "ready", 
                    "message": "✅ Trợ lý pháp lý đã sẵn sàng phục vụ!",
                    "progress": 100,
                    "details": f"Hệ thống hoạt động với {doc_count:,} tài liệu"
                })
                
            except Exception as e:
                self.initialization_status.update({
                    "status": "error", 
                    "message": f"❌ Lỗi khởi tạo: {str(e)}",
                    "progress": 0,
                    "details": "Kiểm tra cấu hình và thử lại"
                })
        
        # Start initialization in background thread
        init_thread = threading.Thread(target=initialize)
        init_thread.daemon = True
        init_thread.start()
    
    def get_system_status(self):
        """Get comprehensive system status"""
        base_status = {
            "initialization": self.initialization_status.copy(),
            "metrics": self.metrics.copy(),
            "uptime": time.time() - self.metrics["start_time"]
        }
        
        if self.initialization_status["status"] == "ready" and self.rag_system:
            # Get detailed system status
            rag_status = self.rag_system.get_system_status()
            base_status.update(rag_status)
            
            # Calculate success rate
            total = self.metrics["total_queries"]
            if total > 0:
                base_status["success_rate"] = (self.metrics["successful_queries"] / total) * 100
                base_status["fallback_rate"] = (self.metrics["fallback_queries"] / total) * 100
            else:
                base_status["success_rate"] = 0
                base_status["fallback_rate"] = 0
        
        return base_status
    
    def get_formatted_status(self):
        """Get formatted status for display"""
        status = self.get_system_status()
        init_status = status["initialization"]
        
        if init_status["status"] == "ready" and self.rag_system:
            # System ready - show comprehensive status
            indicators = []
            if status.get('llm_available'):
                indicators.append("🤖 LLM")
            if status.get('vector_store_available'):
                indicators.append("🔍 Vector Store")
            if status.get('bm25_available'):
                indicators.append("📊 BM25")
            if status.get('reranking_enabled') and status.get('reranker_available'):
                indicators.append("🎯 Reranker")
            
            uptime_hours = status["uptime"] / 3600
            
            return f"""
            <div class="status-ready">
                <h3>✅ Hệ thống hoạt động</h3>
                <p><strong>Thành phần:</strong> {', '.join(indicators)}</p>
                <p><strong>Thời gian hoạt động:</strong> {uptime_hours:.1f} giờ</p>
                <p><strong>Truy vấn:</strong> {status['metrics']['total_queries']} 
                   (Thành công: {status.get('success_rate', 0):.1f}%)</p>
            </div>
            """
        elif init_status["status"] == "error":
            return f"""
            <div class="status-error">
                <h3>❌ Lỗi hệ thống</h3>
                <p>{init_status['message']}</p>
                <p class="status-details">{init_status['details']}</p>
            </div>
            """
        else:
            # Loading state with progress
            return f"""
            <div class="status-loading">
                <h3>{init_status['message']}</h3>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {init_status['progress']}%"></div>
                </div>
                <p class="progress-text">{init_status['progress']}% - {init_status['details']}</p>
            </div>
            """
    
    def create_new_session(self):
        """Create a new conversation session"""
        session_id = f"chat_{int(time.time() * 1000)}"
        self.conversation_sessions[session_id] = {
            "title": "Cuộc trò chuyện mới",
            "messages": [],
            "created_at": datetime.now(),
            "context": {},
            "metrics": {"queries": 0, "avg_response_time": 0}
        }
        self.current_session_id = session_id
        return session_id
    
    def update_session_title(self, session_id: str, first_message: str):
        """Update session title based on first user message"""
        if session_id in self.conversation_sessions:
            title = first_message[:50] + "..." if len(first_message) > 50 else first_message
            self.conversation_sessions[session_id]["title"] = title
    
    def process_message(self, message: str, history: List, session_id: str = None):
        """
        Core message processing with enhanced error handling and metrics
        """
        start_time = time.time()
        processing_status = "🤔 Đang suy nghĩ..."
        
        # Check if system is ready
        if self.initialization_status["status"] != "ready" or not self.rag_system:
            error_response = f"⚠️ {self.initialization_status['message']}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_response})
            return history, "", "❌ Hệ thống chưa sẵn sàng", [], {"show_steps": False}
        
        if not message.strip():
            return history, "", "💬 Nhập câu hỏi để bắt đầu", [], {"show_steps": False}
        
        try:
            # Update metrics
            self.metrics["total_queries"] += 1
            
            # Create session if it doesn't exist
            if session_id is None:
                session_id = self.create_new_session()
            elif session_id not in self.conversation_sessions:
                self.conversation_sessions[session_id] = {
                    "title": "Cuộc trò chuyện mới",
                    "messages": [],
                    "created_at": datetime.now(),
                    "context": {},
                    "metrics": {"queries": 0, "avg_response_time": 0}
                }
            
            # Update session title if this is the first message
            session = self.conversation_sessions[session_id]
            if len(session["messages"]) == 0:
                self.update_session_title(session_id, message.strip())
            
            # Store user message in session context
            session["messages"].append({"role": "user", "content": message, "timestamp": datetime.now()})
            session["metrics"]["queries"] += 1
            
            # Show processing status
            processing_status = "🔍 Đang tìm kiếm tài liệu..."
            
            # Process query through RAG system (pass session history for unified LLM)
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in session.get("messages", [])
                if m.get("role") in ("user", "assistant") and m.get("content")
            ][-16:]
            result = self.rag_system.answer_question(
                message.strip(),
                chat_history=chat_history,
            )
            
            processing_status = "✍️ Đang tạo câu trả lời..."
            
            # Format response with enhanced context
            response = result['answer']
            
            # Add metadata about information sources, search trigger, and question refinement (if enabled)
            source_info = ""
            refinement_info = ""
            
            # Add question refinement info if available and enabled
            if (Config.SHOW_REFINEMENT_INFO and 
                result.get('question_refinement') and 
                result.get('refined_question') != result.get('original_question')):
                refinement = result['question_refinement']
                if refinement.get('refinement_steps'):
                    refinement_summary = self.rag_system.question_refiner.get_refinement_summary(refinement)
                    refinement_info = f"\n\n*🔧 Câu hỏi đã được tối ưu: {refinement_summary}*"
            
            # Add search and source information (if enabled)
            if Config.SHOW_SEARCH_TRIGGER_INFO or Config.SHOW_SOURCE_INFO:
                if result.get('search_triggered'):
                    # Search was triggered due to insufficient information
                    if result.get('fallback_used') and result.get('search_results'):
                        if Config.SHOW_SEARCH_TRIGGER_INFO:
                            source_info = "\n\n*🔍➡️🌐 Không tìm thấy đủ thông tin trong tài liệu tham khảo nên đã tự động tìm kiếm trên web.*"
                        self.metrics["fallback_queries"] += 1
                    else:
                        if Config.SHOW_SEARCH_TRIGGER_INFO:
                            source_info = "\n\n*🔍 Đã kích hoạt tìm kiếm tự động.*"
                        self.metrics["fallback_queries"] += 1
                elif result.get('fallback_used'):
                    self.metrics["fallback_queries"] += 1
                    if result.get('search_results'):
                        if Config.SHOW_SOURCE_INFO:
                            source_info = "\n\n*🌐 Thông tin này được tìm kiếm từ web do không tìm thấy đủ thông tin trong cơ sở dữ liệu pháp luật nội bộ.*"
                else:
                    self.metrics["successful_queries"] += 1
                    if Config.SHOW_SOURCE_INFO:
                        source_info = f"\n\n*📚 Dựa trên {len(result.get('retrieved_documents', []))} tài liệu pháp luật.*"
            else:
                # Update metrics without showing info
                if result.get('search_triggered') or result.get('fallback_used'):
                    self.metrics["fallback_queries"] += 1
                else:
                    self.metrics["successful_queries"] += 1
            
            response += refinement_info + source_info
            
            # Store assistant response in session context
            session["messages"].append({"role": "assistant", "content": response, "timestamp": datetime.now()})
            
            # Update conversation history for display
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            # Update metrics
            response_time = time.time() - start_time
            self.metrics["average_response_time"] = (
                (self.metrics["average_response_time"] * (self.metrics["total_queries"] - 1) + response_time) 
                / self.metrics["total_queries"]
            )
            session["metrics"]["avg_response_time"] = response_time
            
            # Format retrieved documents for display
            docs_info = self._format_retrieved_documents(result.get('retrieved_documents', []))

            processing_status = f"✅ Hoàn thành ({response_time:.1f}s)"

            sources = self._serialize_retrieved_documents(result.get('retrieved_documents', []))
            meta = {
                "show_steps": bool(sources),
            }
            return history, docs_info, processing_status, sources, meta
            
        except Exception as e:
            error_response = f"❌ Lỗi xử lý: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_response})
            processing_status = f"❌ Lỗi: {str(e)[:50]}..."
            return history, "", processing_status, [], {"show_steps": False}
    
    def _serialize_retrieved_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Structured sources for Lexi web UI."""
        items = []
        for doc in documents or []:
            metadata = doc.get("metadata") or {}
            content = str(doc.get("content", ""))
            items.append({
                "title": str(doc.get("title", "Không có tiêu đề")),
                "law_id": str(metadata.get("law_id", "")),
                "excerpt": content[:280] + ("..." if len(content) > 280 else ""),
                "score": doc.get("score"),
            })
        return items
    
    def _format_retrieved_documents(self, documents):
        """Format retrieved documents with simplified styling"""
        if not documents:
            return "📄 **Không tìm thấy tài liệu tham khảo**"
        
        try:
            docs_html = f"## 📄 Tài liệu tham khảo ({len(documents)} tài liệu)\n\n"
            
            for i, doc in enumerate(documents, 1):
                # Safe access to document properties
                title = str(doc.get('title', 'Không có tiêu đề'))
                content = str(doc.get('content', ''))
                law_id = str(doc['metadata'].get('law_id', ''))
                
                # Truncate content for display
                display_content = content[:200] + "..." if len(content) > 200 else content
                
                docs_html += f"### 📄 {i}. ({law_id}) {title}\n"
                docs_html += f"**📝 Nội dung:** {display_content}\n\n"
                docs_html += "---\n\n"
            
            return docs_html
        except Exception as e:
            return f"📄 **Lỗi hiển thị tài liệu: {str(e)}**"
    
    def _format_legal_guidance(self):
        """Format legal guidance for rejected non-legal questions"""
        return """## 📚 Hướng dẫn sử dụng trợ lý pháp lý

Câu hỏi của bạn không thuộc lĩnh vực pháp luật mà tôi có thể hỗ trợ.

### ⚖️ Tôi có thể giúp bạn với:
- **Doanh nghiệp**: Thành lập, giải thể, vốn điều lệ, giấy phép kinh doanh
- **Lao động**: Hợp đồng lao động, lương, nghỉ phép, sa thải, bảo hiểm
- **Thuế**: Kê khai thuế, miễn thuế, thuế thu nhập cá nhân/doanh nghiệp
- **Bất động sản**: Mua bán nhà đất, chuyển nhượng, sổ đỏ, quyền sử dụng đất
- **Gia đình**: Hôn nhân, ly hôn, thừa kế, nuôi con, quyền con cái
- **Dân sự**: Hợp đồng, tranh chấp, bồi thường, quyền sở hữu
- **Hành chính**: Thủ tục pháp lý, giấy tờ, cơ quan nhà nước

### 💡 Gợi ý:
Hãy đặt câu hỏi cụ thể về các vấn đề pháp lý trên để nhận được hỗ trợ tốt nhất!"""
    
    def get_sample_questions(self):
        """Get categorized sample questions"""
        return {
            "🏢 Doanh nghiệp": [
                "Thủ tục thành lập doanh nghiệp như thế nào?",
                "Quy định về vốn điều lệ tối thiểu?",
                "Thủ tục giải thể doanh nghiệp?"
            ],
            "⚖️ Lao động": [
                "Quyền lợi của người lao động khi bị sa thải?",
                "Quy định về thời gian làm việc?",
                "Chế độ nghỉ phép hàng năm?"
            ],
            "💰 Thuế": [
                "Điều kiện miễn thuế thu nhập cá nhân?",
                "Cách tính thuế giá trị gia tăng?",
                "Thủ tục kê khai thuế?"
            ],
            "🏠 Bất động sản": [
                "Hợp đồng mua bán nhà đất cần giấy tờ gì?",
                "Quy trình chuyển nhượng quyền sử dụng đất?",
                "Thủ tục cấp sổ đỏ?"
            ],
            "👨‍👩‍👧‍👦 Gia đình": [
                "Thủ tục ly hôn thuận tình?",
                "Quyền thừa kế của con cái?",
                "Quy định về nuôi con nuôi?"
            ]
        }

def load_css():
    """Load CSS from external file"""
    try:
        with open('css/style.css', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        try:
            with open('css/app/style.css', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print("⚠️ Warning: CSS file not found. Using default styles.")
            return ""

def main():
    """Application entry — Lexi UI (default) or legacy Gradio."""
    ui_mode = os.getenv("UI_MODE", "lexi").strip().lower()
    chatbot = VietnameseLegalChatbot()

    if ui_mode == "gradio":
        print("🚀 Khởi động Trợ lý Pháp lý Việt Nam với Gradio...")
        interface, css = create_chatbot_interface_with(chatbot)
        try:
            interface.launch(
                share=True,
                debug=True,
                theme=gr.themes.Default(),
                css=css,
            )
        except Exception as e:
            print(f"Error launching interface: {e}")
            interface.launch()
        return

    print("🚀 Khởi động Lexi Agent UI (LawShield design)...")
    from lexi_server import run_lexi_server

    run_lexi_server(chatbot)


def create_chatbot_interface_with(chatbot: VietnameseLegalChatbot):
    """Gradio interface using an existing chatbot instance."""
    css = load_css()
    # Reuse body of create_chatbot_interface but inject chatbot — inline minimal duplicate:
    return _build_gradio_interface(chatbot, css)


def create_chatbot_interface():
    chatbot = VietnameseLegalChatbot()
    return create_chatbot_interface_with(chatbot)


def _build_gradio_interface(chatbot: VietnameseLegalChatbot, css: str):
    with gr.Blocks(
        title="Trợ lý Pháp lý Việt Nam",
        analytics_enabled=False,
    ) as interface:
        
        # Enhanced header with simple styling
        gr.HTML("""
        <div class="main-header">
            <h1>⚖️ Trợ lý Pháp lý Việt Nam</h1>
            <p>Hệ thống tư vấn pháp luật thông minh</p>
        </div>
        """)
        
        with gr.Row(elem_classes="main-container"):
            # Left sidebar - Sample questions with dropdowns
            with gr.Column(scale=2, min_width=280):
                gr.HTML('<div class="sidebar-header">💡 Câu hỏi mẫu</div>')
                
                # Sample questions as simple buttons instead of dropdowns
                sample_categories = chatbot.get_sample_questions()
                sample_buttons = []
                
                for category, questions in sample_categories.items():
                    gr.HTML(f'<div style="margin: 10px 0; font-weight: bold; color: #4285f4;">{category}</div>')
                    for question in questions[:2]:  # Limit to 2 questions per category
                        btn = gr.Button(
                            question[:40] + "..." if len(question) > 40 else question,
                            size="sm",
                            variant="secondary",
                            elem_classes="sample-question-btn"
                        )
                        sample_buttons.append((btn, question))
            
            # Center - Main chat interface (expanded)
            with gr.Column(scale=5, min_width=500):
                # Simplified chat interface
                chatbot_component = gr.Chatbot(
                    label="💬 Trợ lý Pháp lý",
                    elem_classes="chat-container-main",
                    height=500,
                )
                
                # Enhanced input area
                with gr.Row():
                    message_input = gr.Textbox(
                        placeholder="Hỏi tôi về pháp luật Việt Nam...",
                        container=False,
                        scale=5,
                        lines=1,
                        elem_classes="main-input"
                    )
                    send_button = gr.Button("📤 Gửi", variant="primary", scale=1, elem_classes="send-button")
                
                # Control buttons
                with gr.Row():
                    clear_chat_btn = gr.Button("🗑️ Xóa cuộc trò chuyện", size="sm", variant="secondary")
            
            # Right sidebar - Reference documents (expanded)
            with gr.Column(scale=3, min_width=350):
                # Enhanced documents display
                docs_display = gr.Markdown(
                    value="📄 **Tài liệu tham khảo sẽ hiển thị ở đây**",
                    label="📚 Cơ sở pháp lý",
                    elem_classes="docs-display"
                )
        
        # Simplified event handlers without State
        def normalize_history(history):
            """Gradio 6 messages format; accept legacy tuple history too."""
            normalized = []
            for item in history or []:
                if isinstance(item, dict) and item.get("role") and "content" in item:
                    normalized.append(
                        {"role": item["role"], "content": str(item["content"])}
                    )
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    if item[0]:
                        normalized.append({"role": "user", "content": str(item[0])})
                    if item[1]:
                        normalized.append(
                            {"role": "assistant", "content": str(item[1])}
                        )
            return normalized

        def handle_message(message, history):
            """Handle user message with simplified approach"""
            if not message.strip():
                return history, "", ""

            session_id = "default"
            internal_history = normalize_history(history)

            try:
                new_history, docs_info, _process_status, _sources, _meta = chatbot.process_message(
                    message, internal_history, session_id
                )
                return new_history, "", docs_info

            except Exception as e:
                error_msg = f"❌ Lỗi xử lý: {str(e)}"
                updated = list(internal_history)
                updated.append({"role": "user", "content": message})
                updated.append({"role": "assistant", "content": error_msg})
                return updated, "", ""
        
        def handle_clear_chat():
            """Clear chat"""
            return [], ""
        

        
        # Wire up sample question buttons
        for btn, question in sample_buttons:
            btn.click(
                lambda q=question: q,
                outputs=[message_input]
            )
        
        # Wire up main events
        send_button.click(
            handle_message,
            inputs=[message_input, chatbot_component],
            outputs=[chatbot_component, message_input, docs_display]
        )
        
        message_input.submit(
            handle_message,
            inputs=[message_input, chatbot_component],
            outputs=[chatbot_component, message_input, docs_display]
        )
        
        clear_chat_btn.click(
            handle_clear_chat,
            outputs=[chatbot_component, docs_display]
        )
    
    return interface, css

def main_legacy_gradio():
    """Deprecated alias — use UI_MODE=gradio."""
    os.environ["UI_MODE"] = "gradio"
    main()


if __name__ == "__main__":
    main()
