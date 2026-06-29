# Tài liệu thuyết minh sản phẩm — Lexi (Vietnamese Legal Chatbot)

**Sản phẩm:** Trợ lý AI pháp lý cho doanh nghiệp SME (Lexi Agent)  
**Repository:** https://github.com/Naammmdz/lexi-agent  
**Cuộc thi / chương trình:** R2AI — Legal Text Retrieval & QA  
**Hạn nộp:** 17h30 ngày 30/06/2026

---

## Mục lục tài liệu

| STT | Tài liệu | File |
|-----|----------|------|
| **★** | **Hướng dẫn giám khảo (đọc file này trước)** | **[HUONG_DAN_GIAM_KHAO.md](HUONG_DAN_GIAM_KHAO.md)** |
| 0 | Giới thiệu sản phẩm | [00_GIOI_THIEU.md](00_GIOI_THIEU.md) |
| 1 | Mô tả dữ liệu | [01_MO_TA_DU_LIEU.md](01_MO_TA_DU_LIEU.md) |
| 2 | Mô hình & checkpoint | [02_MO_HINH.md](02_MO_HINH.md) |
| 3 | Mã nguồn & cấu hình | [03_MA_NGUON_VA_CAU_HINH.md](03_MA_NGUON_VA_CAU_HINH.md) |
| 4 | **Tái hiện pipeline 2000 câu** | [04_HUONG_DAN_TAI_HIEN_2000_CAU.md](04_HUONG_DAN_TAI_HIEN_2000_CAU.md) |
| — | Cài đặt môi trường & Lexi UI | [../README.md](../README.md) |

---

## Checklist nộp bài

### A. Dữ liệu (Google Drive)

**Link:** https://drive.google.com/drive/folders/1yrTBTV-pTdS2FObe1shBHiYmMPsxhazH?usp=drive_link

Trong folder có **`data/`** và **`index/`** (~3 GB tổng). Cách tải và copy vào repo: [HUONG_DAN_GIAM_KHAO.md](HUONG_DAN_GIAM_KHAO.md) bước 3.

| Thư mục trên Drive | Nội dung |
|--------------------|----------|
| `data/` | Corpus merged, law_id mapping, stopwords, … |
| `index/` | BM25 `bm25_index_merged.pkl` |

Qdrant vector DB: build lại bằng `setup_system.py --rebuild` (không bắt buộc có trên Drive).

**Không cần Drive** (trên https://github.com/Naammmdz/lexi-agent ): mã nguồn, `docs/`, `R2AIStage1DATA.json`, `submission.zip`, `submission_qa.zip`.

### B. Mô hình / checkpoint

Model **không** cần upload Drive — tải qua Ollama / Hugging Face khi chạy (xem [02_MO_HINH.md](02_MO_HINH.md)).

| Mô hình | Nguồn | Ghi chú |
|---------|-------|---------|
| `qwen3:4b-instruct` | Ollama Hub | `ollama pull qwen3:4b-instruct` |
| `bkai-foundation-models/vietnamese-bi-encoder` | Hugging Face | Tự tải lần đầu chạy |
| `AITeamVN/Vietnamese_Reranker` | Hugging Face | `ENABLE_RERANKING=1` trên GPU |

Qdrant snapshot (nếu có) đóng gói chung gói Drive mục A.

### C. Mã nguồn

- **Repository:** https://github.com/Naammmdz/lexi-agent  
- Clone:

**macOS / Linux:**

```bash
git clone https://github.com/Naammmdz/lexi-agent.git
cd lexi-agent
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/Naammmdz/lexi-agent.git
cd lexi-agent
```

- Kèm `.env.example`, `scripts/run_full_2000_pipeline.sh` (Mac/Linux/Git Bash; Windows PowerShell xem mục 3 trong [04_HUONG_DAN_TAI_HIEN_2000_CAU.md](04_HUONG_DAN_TAI_HIEN_2000_CAU.md))
- **Không** đính kèm `venv/`, `__pycache__/`, `data/`, `index/` (file lớn — xem mục A Drive)

Chi tiết: [03_MA_NGUON_VA_CAU_HINH.md](03_MA_NGUON_VA_CAU_HINH.md)

### D. Hướng dẫn tái hiện (bắt buộc)

Theo quy định R2AI, người đọc phải **chạy pipeline từ `R2AIStage1DATA.json`** để sinh đủ **2000 dòng** bài nộp.

**Giám khảo chỉ cần mở một file:**

👉 **[HUONG_DAN_GIAM_KHAO.md](HUONG_DAN_GIAM_KHAO.md)** — 10 bước từ clone Git → Drive → chạy pipeline → kiểm tra kết quả (Mac + Windows, tiếng Việt dễ đọc).

| Tài liệu khác | Khi nào cần |
|---------------|-------------|
| [HUONG_DAN_GIAM_KHAO.md](HUONG_DAN_GIAM_KHAO.md) | **Tái hiện bài nộp — đọc file này** |
| `scripts/run_full_2000_pipeline.sh` | Lệnh gói 5 phần xử lý (bước 8 trong hướng dẫn giám khảo) |
| [README.md](../README.md) | Tóm tắt cài đặt |
| [04_HUONG_DAN_TAI_HIEN_2000_CAU.md](04_HUONG_DAN_TAI_HIEN_2000_CAU.md) | Chi tiết kỹ thuật / PowerShell từng lệnh |
| `verify_setup.py` | Kiểm tra môi trường (bước 7) |

Lệnh chính (nằm trong hướng dẫn giám khảo, bước 8):

```bash
export USE_MERGED_CORPUS=1 HYBRID_FUSION=rrf USE_WIDE_RETRIEVAL_POOL=1 ENABLE_RERANKING=1
bash scripts/run_full_2000_pipeline.sh
```

> Phải bật xếp hạng lại (`ENABLE_RERANKING=1`) trước khi chạy. Bỏ bước tìm kiếm 2000 câu → file kết quả **khác** bản tham chiếu trên Git.

---

## Kết quả đạt được (tóm tắt)

| Hạng mục | Metric | Ghi chú |
|----------|--------|---------|
| Information Retrieval (public) | ARTICLES_F2 **0.6308** | `submission.zip` — RRF wide + zone swap |
| | DOCS_F2 0.6466 | |
| Question Answering | **2000 / 2000** câu | `submission_qa.zip` — Ollama grounded |
| Sản phẩm demo | Lexi UI | http://127.0.0.1:7860 |
