# Giới thiệu sản phẩm — Lexi (Trợ lý AI Pháp lý SME)

**Tên sản phẩm:** Lexi Agent — Vietnamese Legal Chatbot  
**Repository:** https://github.com/Naammmdz/lexi-agent  
**Chương trình:** R2AI — Truy hồi & Hỏi đáp Văn bản Pháp luật (Legal Text Retrieval & QA)  
**Đối tượng:** Doanh nghiệp vừa và nhỏ (SME) tại Việt Nam  
**Hạn nộp nghiệm thu:** 17h30 ngày 30/06/2026

---

## 1. Bối cảnh & mục tiêu

Doanh nghiệp SME thường gặp khó khăn khi tra cứu và áp dụng quy định về Luật Doanh nghiệp, thuế, lao động, hợp đồng… Lexi là **trợ lý pháp lý AI** giúp chủ doanh nghiệp, kế toán, nhân sự:

- Tra cứu nhanh **điều luật** liên quan
- Hỏi đáp **tình huống pháp lý** cụ thể
- Nhận **tư vấn sơ bộ** dựa trên văn bản pháp luật chính thống (RAG — không bịa căn cứ)

Sản phẩm tham gia chương trình R2AI với hai hạng mục:

| Hạng mục | Nội dung chấm | File nộp |
|----------|---------------|----------|
| **Information Retrieval (IR)** | Macro Precision / Recall / **F2** trên `relevant_docs`, `relevant_articles` | `submission.zip` |
| **Question Answering (QA)** | 5 tiêu chí (căn cứ, chính xác, đầy đủ, thực tiễn, rõ ràng) khi promote | `submission_qa.zip` |

---

## 2. Giải pháp kỹ thuật (tóm tắt)

```
R2AIStage1DATA.json (2000 câu hỏi)
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  RETRIEVAL (IR)                                            │
│  BM25 + Qdrant (bkai bi-encoder) → Hybrid RRF wide         │
│  → Vietnamese_Reranker → recall boost → zone swap          │
└────────────────────────────────────────────────────────────┘
         │  relevant_docs / relevant_articles
         ▼
┌────────────────────────────────────────────────────────────┐
│  GENERATION (QA)                                           │
│  Đọc nội dung điều luật từ corpus → Ollama Qwen3-4B        │
│  → Câu trả lời grounded (Căn cứ / Trả lời / Lưu ý)         │
└────────────────────────────────────────────────────────────┘
         │
         ▼
   submission.zip  +  submission_qa.zip  (mỗi file 2000 dòng)
```

**Mô hình sử dụng (open-source, < 14B):**

| Vai trò | Model |
|---------|--------|
| Embedding | `bkai-foundation-models/vietnamese-bi-encoder` |
| Reranker | `AITeamVN/Vietnamese_Reranker` |
| LLM (chat & QA) | `qwen3:4b-instruct` (Ollama) |

**Demo sản phẩm:** giao diện web Lexi Agent (`app.py`, cổng 7860) — RAG + hội thoại SME.

---

## 3. Kết quả đạt được

| Hạng mục | Kết quả | Ghi chú |
|----------|---------|---------|
| IR (public leaderboard) | **ARTICLES_F2 = 0.6308** | Hybrid RRF wide + merged corpus + zone swap |
| | DOCS_F2 = 0.6466 | |
| QA | **2000 / 2000** câu có answer | Ollama grounded generation |
| Demo | Lexi UI | http://127.0.0.1:7860 |

---

## 4. Bộ tài liệu nghiệm thu

| STT | Tài liệu | Nội dung |
|-----|----------|----------|
| 0 | [00_GIOI_THIEU.md](00_GIOI_THIEU.md) | Giới thiệu sản phẩm (file này) |
| 1 | [01_MO_TA_DU_LIEU.md](01_MO_TA_DU_LIEU.md) | Nguồn dữ liệu, corpus, index, format bài nộp |
| 2 | [02_MO_HINH.md](02_MO_HINH.md) | Mô hình embedding, reranker, LLM |
| 3 | [03_MA_NGUON_VA_CAU_HINH.md](03_MA_NGUON_VA_CAU_HINH.md) | Cấu trúc mã nguồn, dependencies, cấu hình |
| 4 | [04_HUONG_DAN_TAI_HIEN_2000_CAU.md](04_HUONG_DAN_TAI_HIEN_2000_CAU.md) | **Hướng dẫn chạy full 2000 câu từ đầu** |
| — | [THUYET_MINH_SAN_PHAM.md](THUYET_MINH_SAN_PHAM.md) | Checklist nộp Drive / Git |
| — | [../README.md](../README.md) | Cài đặt môi trường & chạy Lexi UI |

---

## 5. Tái hiện nhanh (2000 câu)

Theo quy định chương trình, người đọc phải **chạy pipeline từ `R2AIStage1DATA.json`**, không chỉ tải file nộp có sẵn.

### Chuẩn bị (một lần)

1. Clone repo:

```bash
git clone https://github.com/Naammmdz/lexi-agent.git
cd lexi-agent
pip install -r requirements.txt
```

2. Tải gói dữ liệu (~2 GB) từ Drive hoặc build corpus/index (xem [01_MO_TA_DU_LIEU.md](01_MO_TA_DU_LIEU.md))
3. Khởi động Qdrant + Ollama
4. `python verify_setup.py`

### Chạy full pipeline

```bash
cd lexi-agent
source venv/bin/activate
bash scripts/run_full_2000_pipeline.sh
```

Script trên sinh **`submission.zip`** (IR) và **`submission_qa.zip`** (IR + answer), mỗi file **2000 dòng**.

Chi tiết từng bước, thời gian ước tính, xử lý sự cố: **[04_HUONG_DAN_TAI_HIEN_2000_CAU.md](04_HUONG_DAN_TAI_HIEN_2000_CAU.md)**
