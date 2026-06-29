# Tài liệu thuyết minh sản phẩm — Lexi (Vietnamese Legal Chatbot)

**Sản phẩm:** Trợ lý AI pháp lý cho doanh nghiệp SME (Lexi Agent)  
**Cuộc thi / chương trình:** R2AI — Legal Text Retrieval & QA  
**Hạn nộp:** 17h30 ngày 30/06/2026

---

## Mục lục tài liệu

| STT | Tài liệu | File |
|-----|----------|------|
| 1 | Mô tả dữ liệu | [01_MO_TA_DU_LIEU.md](01_MO_TA_DU_LIEU.md) |
| 2 | Mô hình & checkpoint | [02_MO_HINH.md](02_MO_HINH.md) |
| 3 | Mã nguồn & cấu hình | [03_MA_NGUON_VA_CAU_HINH.md](03_MA_NGUON_VA_CAU_HINH.md) |
| 4 | Hướng dẫn cài đặt & tái hiện | [../README.md](../README.md) |

---

## Checklist nộp bài

### A. Dữ liệu (link Google Drive / OneDrive)

> **Link chia sẻ dữ liệu:** `[ĐIỀN LINK — ví dụ: https://drive.google.com/...]`

Nén và upload gói `r2ai-data-bundle/` gồm:

| Thành phần | Đường dẫn trong repo | Kích thước (ước lượng) | Bắt buộc |
|------------|----------------------|------------------------|----------|
| Corpus luật (merged) | `data/corpus/legal_corpus_merged.json` | ~664 MB | Có |
| Ánh xạ tên văn bản | `data/law_id_to_title_merged.json` | ~2 MB | Có |
| BM25 index | `index/bm25_index_merged.pkl` | ~1.3 GB | Có (hoặc build lại, xem README) |
| Stopwords | `data/utils/stopwords.txt` | <1 MB | Có |
| Tập câu hỏi R2AI | `R2AIStage1DATA.json` | ~518 KB | Có |
| Qdrant snapshot | export từ collection `bkai_biencoder_vietnamese_legal_corpus_merged` | ~2–5 GB | Khuyến nghị (hoặc build lại bằng `setup_system.py`) |
| Bài nộp IR | `submission.zip` | ~204 KB | Có |
| Bài nộp QA | `submission_qa.zip` | ~449 KB | Có |

### B. Mô hình / checkpoint (link)

> **Link checkpoint & model card:** `[ĐIỀN LINK — ví dụ: https://drive.google.com/...]`

| Mô hình | Nguồn | Ghi chú |
|---------|-------|---------|
| `qwen3:4b-instruct` | Ollama Hub | LLM sinh câu trả lời chat/QA — **không** cần upload weight, chỉ `ollama pull` |
| `bkai-foundation-models/vietnamese-bi-encoder` | Hugging Face | Embedding — tự tải khi chạy lần đầu |
| `AITeamVN/Vietnamese_Reranker` | Hugging Face | Reranker (tùy chọn, `ENABLE_RERANKING=1`) |

Chi tiết: [02_MO_HINH.md](02_MO_HINH.md)

### C. Mã nguồn

- Repository: `[ĐIỀN LINK GITHUB / GITLAB]` hoặc file `r2ai-source.zip`
- **Không** đính kèm thư mục `venv/`, `__pycache__/`, `.git/` (nếu nén tay)
- Kèm file `.env.example` (đã có trong repo)

Chi tiết cấu trúc: [03_MA_NGUON_VA_CAU_HINH.md](03_MA_NGUON_VA_CAU_HINH.md)

### D. README & tái hiện

File [README.md](../README.md) mô tả đầy đủ:

- Yêu cầu phần cứng / phần mềm
- Cài đặt từ đầu (Python, Ollama, Qdrant, dữ liệu)
- Lệnh chạy UI Lexi, pipeline IR/QA
- Cách kiểm tra (`verify_setup.py`)

---

## Kết quả đạt được (tóm tắt)

| Hạng mục | Metric | Ghi chú |
|----------|--------|---------|
| Information Retrieval (public) | ARTICLES_F2 ~ **0.631** | `submission.zip` — hybrid RRF wide + merged corpus |
| Question Answering | 2000/2000 câu | `submission_qa.zip` — grounded QA qua Ollama |
| Sản phẩm demo | Lexi UI | http://127.0.0.1:7860 — RAG + Ollama, giao diện SME |

---

## Liên hệ đội thi

| Vai trò | Họ tên | Email |
|---------|--------|-------|
| Trưởng nhóm | `[ĐIỀN]` | `[ĐIỀN]` |
| Kỹ thuật | `[ĐIỀN]` | `[ĐIỀN]` |
