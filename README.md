# Lexi — Trợ lý AI Pháp lý cho Doanh nghiệp SME 🇻🇳⚖️

Hệ thống **RAG (Retrieval-Augmented Generation)** trả lời câu hỏi pháp luật tiếng Việt, tối ưu cho **doanh nghiệp vừa và nhỏ (SME)**. Sản phẩm gồm pipeline IR/QA cho cuộc thi R2AI và giao diện chat **Lexi Agent**.

| Hạng mục | Kết quả |
|----------|---------|
| Information Retrieval | ARTICLES_F2 ~ **0.631** (`submission.zip`) |
| Question Answering | **2000/2000** câu (`submission_qa.zip`) |
| Demo | Lexi UI tại http://127.0.0.1:7860 |

📄 **Tài liệu thuyết minh nghiệm thu:** [docs/THUYET_MINH_SAN_PHAM.md](docs/THUYET_MINH_SAN_PHAM.md)

---

## Yêu cầu môi trường

| Thành phần | Phiên bản | Ghi chú |
|------------|-----------|---------|
| **Python** | 3.11 – 3.14 | `python3 -m venv venv` |
| **Ollama** | ≥ 0.5 | LLM `qwen3:4b-instruct` |
| **Docker** | ≥ 24 | Chạy Qdrant vector DB |
| **RAM** | ≥ 16 GB | Load BM25 index ~1.3 GB + LLM |
| **Disk** | ≥ 5 GB | Corpus + index + model cache |

---

## Cài đặt từ đầu (reproduce)

### Bước 1 — Clone mã nguồn

```bash
git clone <URL_REPO_CUA_DOI> vietnamese-legal-chatbot
cd vietnamese-legal-chatbot
```

### Bước 2 — Python virtualenv & dependencies

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 3 — Tải dữ liệu & index

> Link Drive đội thi: `[ĐIỀN LINK]` — xem [docs/01_MO_TA_DU_LIEU.md](docs/01_MO_TA_DU_LIEU.md)

Giải nén vào đúng vị trí:

```
data/corpus/legal_corpus_merged.json
data/law_id_to_title_merged.json
data/utils/stopwords.txt
index/bm25_index_merged.pkl
R2AIStage1DATA.json
submission.zip
submission_qa.zip
```

**Hoặc** build lại từ corpus (HF harvest + merge ~1–2 giờ, index thêm 30–90 phút):

```bash
# Bước 1: Zalo corpus từ Kaggle → data/corpus/legal_corpus.json
# Bước 2: Gap từ Hugging Face vbpl.vn
venv/bin/python tools/corpus/build_hf_vbpl_gap_corpus.py
venv/bin/python tools/corpus/build_merged_corpus_v2.py --gap data/augmented/hf_vbpl_gap_corpus.json
python setup_system.py --rebuild
```

Nguồn HF: [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents)

### Bước 4 — Cấu hình môi trường

```bash
cp .env.example .env
# Chỉnh .env nếu cần (mặc định đã tối ưu cho IR + Lexi UI)
```

### Bước 5 — Khởi động dịch vụ phụ thuộc

```bash
# Qdrant (vector database)
docker run -d --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Ollama (LLM) — terminal riêng
ollama serve
ollama pull qwen3:4b-instruct
```

**Import Qdrant snapshot** (nếu có trên Drive, khuyến nghị thay vì rebuild):

```bash
# Xem hướng dẫn: https://qdrant.tech/documentation/database/snapshots/
```

### Bước 6 — Kiểm tra

```bash
python verify_setup.py
```

Kết quả mong đợi: ✅ Config, ✅ Ollama running, ✅ Ollama has qwen3:4b-instruct

### Bước 7 — Chạy Lexi UI

```bash
PYTHONUNBUFFERED=1 UI_MODE=lexi python app.py
```

Mở trình duyệt: **http://127.0.0.1:7860**

Lần khởi động đầu load BM25 (~60–90 giây). Sidebar hiển thị **Sẵn sàng** khi xong.

---

## Chế độ UI

| `UI_MODE` | Mô tả |
|-----------|--------|
| `lexi` (mặc định) | Giao diện Lexi Agent (FastAPI + HTML/JS) |
| `gradio` | Giao diện Gradio legacy |

```bash
UI_MODE=gradio python app.py
```

---

## Pipeline IR & QA (cuộc thi)

### Information Retrieval

Cấu hình tốt nhất (đã nộp):

```bash
export USE_MERGED_CORPUS=1
export HYBRID_FUSION=rrf
export USE_WIDE_RETRIEVAL_POOL=1
```

File nộp: `submission.zip` (~0.631 ARTICLES_F2 public)

### Question Answering

```bash
venv/bin/python tools/submission/create_qa_submission.py \
  --backend ollama \
  --model qwen3:4b-instruct \
  --output submission_qa.zip
```

Chi tiết mô hình: [docs/02_MO_HINH.md](docs/02_MO_HINH.md)

---

## Kiến trúc hệ thống

```
User → Lexi UI (FastAPI)
         → VietnameseLegalRAG
              ├─ Hybrid Retrieval (BM25 + Qdrant/bkai-bi-encoder)
              ├─ [Optional] Vietnamese_Reranker cross-encoder
              └─ Ollama Qwen3-4B-Instruct (unified SME chat + grounded QA)
```

---

## Cấu trúc mã nguồn

Xem chi tiết: [docs/03_MA_NGUON_VA_CAU_HINH.md](docs/03_MA_NGUON_VA_CAU_HINH.md)

---

## Xử lý sự cố

| Triệu chứng | Cách xử lý |
|-------------|------------|
| `address already in use :7860` | `lsof -ti :7860 \| xargs kill -9` |
| Treo sau "Loading weights" reranker | Đặt `ENABLE_RERANKING=0` trong `.env` |
| Ollama not reachable | `ollama serve` + `ollama pull qwen3:4b-instruct` |
| Qdrant connection refused | `docker start qdrant` hoặc chạy lại container |
| Không tìm thấy corpus | Tải `legal_corpus_merged.json` từ Drive |

---

## Tài liệu bổ sung

- [Thuyết minh sản phẩm (checklist nộp)](docs/THUYET_MINH_SAN_PHAM.md)
- [Mô tả dữ liệu](docs/01_MO_TA_DU_LIEU.md)
- [Mô hình & checkpoint](docs/02_MO_HINH.md)
- [Mã nguồn & cấu hình](docs/03_MA_NGUON_VA_CAU_HINH.md)
- [Submission tools](tools/submission/README.md)

---

## License

MIT — xem [LICENSE](LICENSE).
