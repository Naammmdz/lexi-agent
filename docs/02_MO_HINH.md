# 2. Mô hình sử dụng

## 2.1. Tổng quan kiến trúc mô hình

```
Câu hỏi người dùng
    │
    ├─► Embedding: bkai-foundation-models/vietnamese-bi-encoder  ──► Qdrant (vector)
    ├─► BM25 (rank-bm25)                                          ──► Inverted index
    │         └─► Hybrid fusion (RRF)
    │                   └─► [Tuỳ chọn] AITeamVN/Vietnamese_Reranker
    │
    └─► LLM: Qwen3-4B-Instruct (qua Ollama) + prompt grounded SME
              └─► Câu trả lời Lexi
```

---

## 2.2. Chi tiết từng mô hình

### A. Embedding — truy hồi vector

| Thuộc tính | Giá trị |
|------------|---------|
| **Tên** | `bkai-foundation-models/vietnamese-bi-encoder` |
| **Nền tảng** | [Hugging Face](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder) |
| **Loại** | Sentence Transformer (bi-encoder) |
| **Chiều vector** | 768 |
| **Vai trò** | Mã hoá câu hỏi & điều luật → tìm kiếm ngữ nghĩa trong Qdrant |
| **Cấu hình** | `config.py` → `EMBEDDING_MODEL` |

**Tải & sử dụng:**

```bash
# Tự động tải lần đầu qua sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')"
```

Checkpoint lưu tại: `~/.cache/huggingface/hub/` (không cần upload nếu máy có internet).

---

### B. Reranker — xếp hạng lại (tuỳ chọn)

| Thuộc tính | Giá trị |
|------------|---------|
| **Tên** | `AITeamVN/Vietnamese_Reranker` |
| **Nền tảng** | [Hugging Face](https://huggingface.co/AITeamVN/Vietnamese_Reranker) |
| **Loại** | Cross-encoder |
| **Vai trò** | Chấm điểm cặp (câu hỏi, đoạn luật) sau hybrid retrieval |
| **Mặc định UI Mac** | `ENABLE_RERANKING=0` (tránh treo khi load weight) |
| **Pipeline IR server** | `ENABLE_RERANKING=1` trên GPU |

**Bật reranker:**

```bash
export ENABLE_RERANKING=1
export RERANKER_DEVICE=cuda   # hoặc mps / cpu
```

---

### C. LLM — sinh câu trả lời (Chat / QA)

| Thuộc tính | Giá trị |
|------------|---------|
| **Tên** | `qwen3:4b-instruct` |
| **Phiên bản** | Qwen3 4B Instruct (non-thinking) |
| **Nền tảng** | [Ollama](https://ollama.com/library/qwen3) |
| **Kích thước** | ~2.5 GB (quantized GGUF qua Ollama) |
| **Vai trò** | Hội thoại Lexi SME + sinh câu trả lời QA grounded |
| **Cấu hình** | `.env` → `OLLAMA_MODEL=qwen3:4b-instruct` |

**Tải & chạy:**

```bash
# Cài Ollama: https://ollama.com/download
ollama serve                    # terminal 1
ollama pull qwen3:4b-instruct   # một lần
ollama list                     # kiểm tra
```

**Tham số sinh (chat):**

| Tham số | Giá trị | File |
|---------|---------|------|
| `temperature` | 0.35 | `utils/chat_llm.py` |
| `num_predict` | 512 | `CHAT_MAX_NEW_TOKENS` |
| `num_ctx` | 8192 | `OLLAMA_NUM_CTX` |

**Prompt hệ thống:** `utils/chat_llm.py` → `build_unified_sme_system_prompt()`  
**Prompt QA grounded:** `utils/qa_answer_generator.py` → `build_system_prompt()`

---

### D. Mô hình dự phòng (không dùng trong cấu hình mặc định)

| Mô hình | Khi nào dùng |
|---------|--------------|
| `gemini-2.0-flash` | `USE_LOCAL_LLM=False` + `GOOGLE_API_KEY` |
| `Qwen/Qwen3-4B-Instruct-2507` (vLLM) | Server GPU batch QA — `tools/submission/create_qa_submission.py --backend vllm` |
| Local HF transformers | `LLM_BACKEND=local_hf` — cần ~8–10 GB RAM |

---

## 2.3. Checkpoint & artifact cần chia sẻ

| Artifact | Có trên GitHub? | Cách lấy |
|----------|-----------------|----------|
| Mã nguồn | **Có** | https://github.com/Naammmdz/lexi-agent |
| `submission.zip`, `submission_qa.zip` | **Có** | Trong repo sau clone |
| Ollama `qwen3:4b-instruct` | Không (weight) | `ollama pull qwen3:4b-instruct` |
| HF embedding / reranker | Không (cache) | Tự tải khi chạy; hoặc zip `~/.cache/huggingface/` nếu offline |
| Corpus + BM25 + Qdrant snapshot | Không | Gói Drive (`r2ai-data-bundle`) — [01_MO_TA_DU_LIEU.md](01_MO_TA_DU_LIEU.md) |

### Export Qdrant snapshot (cho người tái hiện)

```bash
# Trên máy đã có collection đầy đủ
curl -X POST "http://localhost:6333/collections/bkai_biencoder_vietnamese_legal_corpus_merged/snapshots"
# Tải file snapshot từ API Qdrant và upload lên Drive
```

### Import snapshot

```bash
# Xem tài liệu Qdrant: https://qdrant.tech/documentation/database/snapshots/
```

---

## 2.4. Kết quả benchmark mô hình

### Information Retrieval (pipeline đã nộp)

| Cấu hình | ARTICLES_F2 (public) |
|----------|----------------------|
| BM25 only | ~0.55 |
| Hybrid max | ~0.58 |
| **Hybrid RRF wide + merged corpus** | **~0.631** |

File nộp: `submission.zip` (variant `rrf_swap_g008`)

### Question Answering

| Backend | Số câu | File |
|---------|--------|------|
| Ollama `qwen3:4b-instruct` | 2000/2000 | `submission_qa.zip` — xem [04_HUONG_DAN_TAI_HIEN_2000_CAU.md](04_HUONG_DAN_TAI_HIEN_2000_CAU.md) |

---

## 2.5. Yêu cầu tài nguyên

| Thành phần | RAM | Disk | GPU |
|------------|-----|------|-----|
| UI Lexi (Ollama + BM25 + Qdrant) | ≥16 GB | ~3 GB (index) + ~2.5 GB (LLM) | Tuỳ chọn (Metal/CUDA cho Ollama) |
| Pipeline IR đầy đủ (reranker) | ≥32 GB | ~5 GB | Khuyến nghị CUDA |
| Chỉ inference QA (có sẵn index) | ≥8 GB | ~3 GB | Tuỳ chọn |
