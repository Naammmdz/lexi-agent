# 3. Mã nguồn & cấu hình triển khai

## 3.1. Cấu trúc thư mục chính

```
vietnamese-legal-chatbot/
├── app.py                  # Entry point — UI Lexi / Gradio
├── lexi_server.py          # FastAPI server cho giao diện Lexi
├── config.py               # Toàn bộ hyperparameter & đường dẫn
├── setup_system.py         # Build / rebuild Qdrant + BM25 index
├── verify_setup.py         # Kiểm tra môi trường nhanh
├── requirements.txt        # Python dependencies
├── .env.example            # Mẫu biến môi trường
│
├── main/                   # Core RAG
│   ├── chatbot.py          # VietnameseLegalRAG — pipeline chính
│   ├── vector_store.py     # Qdrant client + embedding
│   ├── bm25_retriever.py   # BM25 retrieval
│   ├── reranker.py         # Cross-encoder local
│   └── api_reranker.py     # Reranker qua API (FPT Cloud)
│
├── utils/
│   ├── chat_llm.py         # Ollama unified chat (Lexi SME)
│   ├── qa_answer_generator.py  # Grounded QA + Ollama batch
│   ├── data_loader.py      # Đọc corpus JSON
│   ├── question_refiner.py   # Tinh chỉnh câu hỏi
│   ├── retrieval_scoring.py  # RRF, law shortlist, fusion
│   └── text_processor.py     # Tiền xử lý tiếng Việt
│
├── ui/lexi/                # Giao diện web Lexi Agent
│   ├── index.html
│   ├── app.js
│   └── lexi.css
│
├── tools/
│   ├── corpus/             # Build merged corpus, BM25
│   │   ├── build_hf_vbpl_gap_corpus.py   # Gap từ HF vbpl.vn
│   │   └── build_merged_corpus_v2.py     # Merge Zalo + gap
│   └── submission/         # Script tạo submission IR/QA
│
├── data/                   # Dữ liệu (tải từ Drive)
├── index/                  # BM25 pickle
└── docs/                   # Tài liệu thuyết minh nghiệm thu
```

---

## 3.2. Dependencies (requirements.txt)

| Nhóm | Thư viện | Phiên bản | Vai trò |
|------|----------|-----------|---------|
| **Web UI** | `fastapi`, `uvicorn` | latest trong venv | Lexi API server |
| | `gradio` | latest | UI thay thế (`UI_MODE=gradio`) |
| | `streamlit` | latest | Legacy UI |
| **RAG / LLM** | `langchain`, `langchain-community`, `langchain-google-genai` | latest | Prompt, Gemini fallback |
| | `qdrant-client` | latest | Vector DB |
| | `sentence-transformers` | latest | Embedding model |
| | `rank-bm25` | latest | BM25 index |
| **Xử lý tiếng Việt** | `underthesea`, `pyvi` | latest | Tokenize / NLP |
| **Dữ liệu** | `pandas`, `numpy`, `scikit-learn` | latest | Benchmark, xử lý CSV |
| **Khác** | `python-dotenv` | latest | Đọc `.env` |
| | `googlesearch-python`, `beautifulsoup4`, `requests` | latest | Google Search fallback (tuỳ chọn) |

### Phụ thuộc hệ thống (ngoài pip)

| Phần mềm | Phiên bản khuyến nghị | Bắt buộc |
|----------|----------------------|----------|
| Python | 3.11 – 3.14 | Có |
| Ollama | ≥ 0.5 | Có (chat/QA) |
| Docker | ≥ 24 | Có (chạy Qdrant) |
| Git | bất kỳ | Khuyến nghị |

Cài đặt Python:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3.3. Tệp cấu hình triển khai

| File | Mô tả |
|------|--------|
| `.env` | Biến môi trường runtime (copy từ `.env.example`) |
| `config.py` | Hằng số mặc định: retrieval top-k, fusion RRF, đường dẫn data |
| `.env.example` | Mẫu đầy đủ cho người tái hiện |

### Biến môi trường quan trọng

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `USE_MERGED_CORPUS` | `1` | Dùng corpus + index merged |
| `HYBRID_FUSION` | `rrf` | Reciprocal Rank Fusion |
| `USE_WIDE_RETRIEVAL_POOL` | `1` | Pool retrieval rộng (IR tốt nhất) |
| `OLLAMA_MODEL` | `qwen3:4b-instruct` | LLM chat/QA |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Endpoint Ollama |
| `ENABLE_RERANKING` | `0` (UI Mac) / `1` (server) | Bật cross-encoder reranker |
| `UI_MODE` | `lexi` | `lexi` \| `gradio` |
| `LEXI_PORT` | `7860` | Cổng web UI |

---

## 3.4. Entry points & lệnh vận hành

### Chạy sản phẩm (Lexi UI)

```bash
# Terminal 1 — Qdrant
docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant

# Terminal 2 — Ollama
ollama serve
ollama pull qwen3:4b-instruct

# Terminal 3 — App
cd vietnamese-legal-chatbot
cp .env.example .env
source venv/bin/activate
PYTHONUNBUFFERED=1 UI_MODE=lexi python app.py
# Mở http://127.0.0.1:7860
```

### Build index lần đầu

```bash
python setup_system.py          # dùng index có sẵn nếu đã tải BM25
python setup_system.py --rebuild   # build lại Qdrant + BM25 từ corpus
python setup_system.py --test      # + chạy 3 câu hỏi mẫu
```

### Kiểm tra môi trường

```bash
python verify_setup.py
```

### Tạo bài nộp IR / QA

```bash
# IR (đã có submission.zip tốt nhất)
venv/bin/python tools/submission/create_final_submission.py

# QA batch qua Ollama
venv/bin/python tools/submission/create_qa_submission.py \
  --backend ollama \
  --model qwen3:4b-instruct \
  --output submission_qa.zip
```

---

## 3.5. API endpoints (Lexi server)

| Method | Path | Mô tả |
|--------|------|--------|
| `GET` | `/` | Giao diện Lexi |
| `GET` | `/api/status` | Trạng thái khởi tạo hệ thống |
| `GET` | `/api/samples` | Câu hỏi mẫu theo chủ đề |
| `POST` | `/api/chat` | `{ message, history, session_id }` → trả lời + sources |

---

## 3.6. Giấy phép mã nguồn

Dự án sử dụng [MIT License](../LICENSE).

Mô hình bên thứ ba tuân theo giấy phép tương ứng trên Hugging Face / Ollama.
