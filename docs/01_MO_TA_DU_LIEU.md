# 1. Tài liệu mô tả dữ liệu

## 1.1. Nguồn dữ liệu

### Corpus pháp luật Việt Nam (chính)

| Nguồn | Mô tả | Giấy phép / truy cập |
|-------|--------|----------------------|
| **[Zalo AI Challenge 2021 — Legal Text Retrieval](https://www.kaggle.com/datasets/hariwh0/zaloai2021-legal-text-retrieval)** | Corpus gốc ~114 MB (`legal_corpus.json`, 3.271 văn bản), cấu trúc theo điều khoản | Kaggle (điều khoản dataset) |
| **[Vietnamese Legal Documents (vbpl.vn)](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents)** | Nguồn **legal DB** dùng để merge bổ sung văn bản thiếu: luật, nghị định, thông tư từ Cổng VBPL Bộ Tư pháp (~153k metadata, ~149k HTML full-text) | **CC BY 4.0** — Hugging Face |
| **Corpus merged (sản phẩm)** | `legal_corpus_merged.json` — Zalo 2021 + các văn bản QH/NĐ-CP/TT có trong HF nhưng chưa có trong Zalo (**18.243** văn bản, **~237.751** điều) | Kế thừa giấy phép nguồn gốc |

#### Cách merge corpus

```
legal_corpus.json (Zalo 2021)
        +
hf_vbpl_gap_corpus.json  ←  th1nhng0/vietnamese-legal-documents
        │
        ▼
legal_corpus_merged.json
```

Script: `tools/corpus/build_hf_vbpl_gap_corpus.py` → `tools/corpus/build_merged_corpus_v2.py`

**Trích dẫn dataset HF:**

```bibtex
@dataset{ngo_thinh_2026_vietnamese_legal,
  title     = {Vietnamese Legal Documents},
  author    = {Thịnh Ngô},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents},
}
```

### Tập câu hỏi thi đấu (R2AI)

| File | Mô tả |
|------|--------|
| `R2AIStage1DATA.json` | ~518 KB — danh sách câu hỏi pháp lý tiếng Việt cho giai đoạn IR/QA của chương trình R2AI |

### Dữ liệu phụ trợ

| File | Mô tả |
|------|--------|
| `data/law_id_to_title_merged.json` | Ánh xạ `law_id` → tên hiển thị văn bản (dùng format citation) |
| `data/utils/stopwords.txt` | Stopwords tiếng Việt cho tiền xử lý BM25 |
| `data/train/` (nếu có) | Tập train Zalo 2021 — dùng benchmark nội bộ |

---

## 1.2. Cấu trúc & định dạng dữ liệu

### Corpus JSON (`legal_corpus_merged.json`)

Định dạng: **UTF-8 JSON**, mảng các văn bản luật.

```json
[
  {
    "law_id": "01/2009/tt-bnn",
    "articles": [
      {
        "article_id": "1",
        "title": "Điều 1. Phạm vi áp dụng",
        "text": "Nội dung điều luật..."
      }
    ]
  }
]
```

| Trường | Kiểu | Mô tả |
|--------|------|--------|
| `law_id` | string | Mã định danh văn bản (số hiệu / slug) |
| `articles` | array | Danh sách điều khoản |
| `articles[].article_id` | string | Số điều |
| `articles[].title` | string | Tiêu đề điều (thường bắt đầu bằng "Điều N.") |
| `articles[].text` | string | Nội dung điều luật |

Sau khi index, mỗi **điều luật** trở thành một document:

- `id`: `{law_id}_{article_id}`
- `metadata.law_id`, `metadata.article_id`, `metadata.title`

### BM25 index (`index/bm25_index_merged.pkl`)

| Thuộc tính | Giá trị |
|------------|---------|
| Định dạng | Python pickle (rank-bm25) |
| Kích thước | ~1.3 GB |
| Số document | ~237.000 chunk/article (khớp Qdrant collection merged) |
| Tạo bởi | `tools/corpus/build_bm25_merged.py` hoặc `setup_system.py --rebuild` |

### Vector index (Qdrant)

| Thuộc tính | Giá trị |
|------------|---------|
| Collection | `bkai_biencoder_vietnamese_legal_corpus_merged` |
| Embedding | `bkai-foundation-models/vietnamese-bi-encoder` (768-dim) |
| Số điểm | ~237.522 vectors |

### Bài nộp IR (`submission.zip`)

CSV trong ZIP, mỗi dòng một câu hỏi:

| Cột | Mô tả |
|-----|--------|
| `question_id` | ID câu hỏi |
| `relevant_articles` | Danh sách tham chiếu điều luật (format `law_id\|Tên luật\|Điều N`) |

### Bài nộp QA (`submission_qa.zip`)

CSV trong ZIP:

| Cột | Mô tả |
|-----|--------|
| `question_id` | ID câu hỏi |
| `answer` | Câu trả lời tư vấn pháp lý (grounded trên điều luật truy hồi) |

---

## 1.3. Hướng dẫn truy cập / sử dụng dữ liệu

### Tải gói dữ liệu đã đóng gói

> **Link:** `[ĐIỀN LINK GOOGLE DRIVE / ONEDRIVE]`

Cấu trúc sau khi giải nén vào thư mục gốc repo:

```
vietnamese-legal-chatbot/
├── data/
│   ├── corpus/
│   │   └── legal_corpus_merged.json
│   ├── law_id_to_title_merged.json
│   └── utils/
│       └── stopwords.txt
├── index/
│   └── bm25_index_merged.pkl
├── R2AIStage1DATA.json
├── submission.zip
└── submission_qa.zip
```

### Tải corpus gốc từ Kaggle (bước 1)

```bash
# Cần tài khoản Kaggle + API token
kaggle datasets download -d hariwh0/zaloai2021-legal-text-retrieval
unzip zaloai2021-legal-text-retrieval.zip -d data/
# Đặt legal_corpus.json vào data/corpus/
```

### Tải & merge từ Hugging Face vbpl.vn (bước 2)

```bash
pip install datasets huggingface_hub

# Thu thập văn bản QH/NĐ-CP/TT thiếu so với Zalo (~ vài chục phút, cần mạng)
venv/bin/python tools/corpus/build_hf_vbpl_gap_corpus.py

# Gộp Zalo + gap → legal_corpus_merged.json
venv/bin/python tools/corpus/build_merged_corpus_v2.py \
  --gap data/augmented/hf_vbpl_gap_corpus.json
```

Dataset: [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents)

```python
from datasets import load_dataset

meta = load_dataset("th1nhng0/vietnamese-legal-documents", "metadata", split="data", streaming=True)
content = load_dataset("th1nhng0/vietnamese-legal-documents", "content", split="data", streaming=True)
```

Hoặc chạy pipeline đầy đủ (audit → HF gap → merge → re-index):

```bash
bash tools/corpus/run_p4_corpus_pipeline.sh
```

**Lưu ý:** File `legal_corpus_merged.json` đã build sẵn trong gói Drive — không bắt buộc chạy lại HF harvest khi chỉ demo/chat.

### Build lại index (nếu không tải BM25 / Qdrant snapshot)

```bash
# Yêu cầu: Qdrant đang chạy (docker run -p 6333:6333 qdrant/qdrant)
venv/bin/python setup_system.py --rebuild
```

Thời gian build: 30–90 phút tuỳ cấu hình máy.

### Sử dụng trong code

```python
from utils.data_loader import LegalDataLoader

loader = LegalDataLoader()
corpus = loader.load_legal_corpus()          # đọc JSON
documents = loader.prepare_documents_for_indexing()  # flatten theo điều luật
```

---

## 1.4. Thống kê tóm tắt

| Chỉ số | Giá trị |
|--------|---------|
| Văn bản Zalo gốc | 3.271 |
| Văn bản sau merge (HF vbpl gap) | **18.243** |
| Tổng điều khoản (merged) | **~237.751** |
| Số document đã index (Qdrant/BM25) | ~237.522 |
| Nguồn gap-fill | [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) |
| Ngôn ngữ | Tiếng Việt |
| Encoding | UTF-8 |
