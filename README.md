# Lexi — Trợ lý AI Pháp lý cho Doanh nghiệp SME 🇻🇳⚖️

Hệ thống **RAG (Retrieval-Augmented Generation)** trả lời câu hỏi pháp luật tiếng Việt, tối ưu cho **doanh nghiệp vừa và nhỏ (SME)**. Sản phẩm gồm pipeline IR/QA cho cuộc thi R2AI và giao diện chat **Lexi Agent**.

| Hạng mục | Kết quả |
|----------|---------|
| Information Retrieval | ARTICLES_F2 **0.6308** (`submission.zip`) |
| Question Answering | **2000/2000** câu (`submission_qa.zip`) |
| Demo | Lexi UI tại http://127.0.0.1:7860 |

**Repository:** https://github.com/Naammmdz/lexi-agent  
**Dữ liệu (Drive):** https://drive.google.com/drive/folders/1yrTBTV-pTdS2FObe1shBHiYmMPsxhazH?usp=drive_link

📄 **Bộ tài liệu nghiệm thu:** [docs/THUYET_MINH_SAN_PHAM.md](docs/THUYET_MINH_SAN_PHAM.md)  
📄 **Tái hiện full 2000 câu:** [docs/04_HUONG_DAN_TAI_HIEN_2000_CAU.md](docs/04_HUONG_DAN_TAI_HIEN_2000_CAU.md)

---

## Tài liệu hướng dẫn

| File | Mục đích |
|------|----------|
| [docs/00_GIOI_THIEU.md](docs/00_GIOI_THIEU.md) | Giới thiệu sản phẩm & kiến trúc |
| [docs/01_MO_TA_DU_LIEU.md](docs/01_MO_TA_DU_LIEU.md) | Dữ liệu, corpus, index |
| [docs/02_MO_HINH.md](docs/02_MO_HINH.md) | Mô hình embedding, reranker, LLM |
| [docs/03_MA_NGUON_VA_CAU_HINH.md](docs/03_MA_NGUON_VA_CAU_HINH.md) | Mã nguồn & cấu hình |
| [docs/04_HUONG_DAN_TAI_HIEN_2000_CAU.md](docs/04_HUONG_DAN_TAI_HIEN_2000_CAU.md) | **Chạy pipeline 2000 câu từ đầu** |

---

## Yêu cầu môi trường

| Thành phần | Phiên bản | Ghi chú |
|------------|-----------|---------|
| **Python** | 3.11 – 3.14 | `python3 -m venv venv` |
| **Ollama** | ≥ 0.5 | LLM `qwen3:4b-instruct` |
| **Docker** | ≥ 24 | Chạy Qdrant vector DB |
| **RAM** | ≥ 16 GB | Pipeline IR khuyến nghị 32 GB + GPU |
| **Disk** | ≥ 8 GB | Corpus + index + model cache |

---

## Cài đặt từ đầu

### Bước 1 — Clone mã nguồn

```bash
git clone https://github.com/Naammmdz/lexi-agent.git
cd lexi-agent
```

### Bước 2 — Python virtualenv & dependencies

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### Bước 3 — Tải `data/` và `index/` từ Google Drive

**Link:** https://drive.google.com/drive/folders/1yrTBTV-pTdS2FObe1shBHiYmMPsxhazH?usp=drive_link

Tải hai folder `data` và `index`, copy vào thư mục `lexi-agent/`. Hướng dẫn chi tiết **macOS và Windows**: [docs/04_HUONG_DAN_TAI_HIEN_2000_CAU.md](docs/04_HUONG_DAN_TAI_HIEN_2000_CAU.md) mục 2.3.

**Đã có sẵn trên Git** (sau `git clone`):

- `R2AIStage1DATA.json`
- `submission.zip`, `submission_qa.zip` (kết quả tham chiếu)
- Toàn bộ mã nguồn và `docs/`

Giải nén gói Drive vào thư mục gốc `lexi-agent/`:

```
data/corpus/legal_corpus_merged.json
data/law_id_to_title_merged.json
data/utils/stopwords.txt
index/bm25_index_merged.pkl
R2AIStage1DATA.json
```

File `submission.zip` / `submission_qa.zip` trên Git là **kết quả tham chiếu** — tái hiện phải **chạy lại pipeline** (bước 5).

**Hoặc** build corpus/index từ nguồn công khai:

```bash
venv/bin/python tools/corpus/build_hf_vbpl_gap_corpus.py
venv/bin/python tools/corpus/build_merged_corpus_v2.py --gap data/augmented/hf_vbpl_gap_corpus.json
python setup_system.py --rebuild
```

### Bước 4 — Khởi động dịch vụ

**macOS / Linux:**

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
ollama serve
ollama pull qwen3:4b-instruct
```

**Windows:** Cài [Docker Desktop](https://www.docker.com/products/docker-desktop/) và [Ollama](https://ollama.com/download). Mở Docker Desktop, rồi trong PowerShell:

```powershell
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
ollama pull qwen3:4b-instruct
```

(Ollama trên Windows thường chạy nền sau khi cài — không cần `ollama serve` riêng.)

Sau khi có `data/` từ Drive, build Qdrant index lần đầu: `python setup_system.py --rebuild` (venv đã activate).

### Bước 5 — Tái hiện bài nộp 2000 câu

**macOS / Linux:**

```bash
source venv/bin/activate
export ENABLE_RERANKING=1
bash scripts/run_full_2000_pipeline.sh
```

**Windows:** Git Bash (lệnh trên) hoặc PowerShell từng bước — xem [docs/04_HUONG_DAN_TAI_HIEN_2000_CAU.md](docs/04_HUONG_DAN_TAI_HIEN_2000_CAU.md) mục 3.

**Output:** `submission.zip` + `submission_qa.zip` (mỗi file **2000 dòng**).

Hướng dẫn chi tiết: [docs/04_HUONG_DAN_TAI_HIEN_2000_CAU.md](docs/04_HUONG_DAN_TAI_HIEN_2000_CAU.md)

### Bước 6 — Kiểm tra

```bash
python verify_setup.py
```

### Bước 7 — Chạy Lexi UI (demo)

```bash
PYTHONUNBUFFERED=1 UI_MODE=lexi python app.py
# http://127.0.0.1:7860
```

---

## Kiến trúc hệ thống

```
User → Lexi UI (FastAPI)
         → VietnameseLegalRAG
              ├─ Hybrid Retrieval (BM25 + Qdrant/bkai-bi-encoder)
              ├─ RRF fusion + [Optional] Vietnamese_Reranker
              └─ Ollama Qwen3-4B-Instruct (chat + grounded QA)
```

---

## Xử lý sự cố

| Triệu chứng | Cách xử lý |
|-------------|------------|
| `address already in use :7860` | `lsof -ti :7860 \| xargs kill -9` |
| Qdrant connection refused | `docker start qdrant` |
| Ollama not reachable | `ollama serve` + `ollama pull qwen3:4b-instruct` |
| Pipeline chậm / OOM | Chạy cache IR trên GPU; QA tách process |
| Thiếu corpus | Tải từ Drive hoặc xem [01_MO_TA_DU_LIEU.md](docs/01_MO_TA_DU_LIEU.md) |

---

## License

MIT — xem [LICENSE](LICENSE).
