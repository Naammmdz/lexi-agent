# Hướng dẫn tái hiện pipeline 2000 câu (R2AI)

> **Giám khảo:** đọc **[HUONG_DAN_GIAM_KHAO.md](HUONG_DAN_GIAM_KHAO.md)** — một file, 10 bước, dễ hiểu.  
> File này là **bản chi tiết kỹ thuật** (tham khảo thêm, lệnh PowerShell từng dòng).

Tài liệu này mô tả cách **chạy đầy đủ** pipeline từ `R2AIStage1DATA.json` để sinh bài nộp **2000 dòng**, đúng quy định chương trình R2AI (Quy trình và hướng dẫn nộp bài dự thi).

**Đầu vào bắt buộc:** `R2AIStage1DATA.json` (2000 câu hỏi)  
**Đầu ra:** `submission.zip` (IR) và `submission_qa.zip` (IR + câu trả lời đầy đủ)

---

## 1. Tổng quan luồng xử lý

```
R2AIStage1DATA.json
        │
        ├─① cache_live_retrieval.py          Retrieval 2000 câu (BM25+Qdrant+RRF+reranker)
        │
        ├─② create_cache_only_submission.py  Seed IR từ cache (không cần submission.zip cũ)
        │
        ├─③ create_recall_boost_submission.py  Thêm companion article, cap 2 điều / 1 VB
        │
        ├─④ create_rrf_zone_swap_submission.py   Zone swap top-1 (đạt ~0.631 IR)
        │        → submission.zip
        │
        └─⑤ create_qa_submission.py            Sinh answer grounded (Ollama)
                 → submission_qa.zip
```

| Bước | Script | Thời gian ước tính |
|------|--------|-------------------|
| ① Cache retrieval | `cache_live_retrieval.py` | **3–6 giờ** (GPU + reranker) |
| ②–④ Build IR | cache seed + recall boost + zone swap | **< 5 phút** |
| ⑤ Sinh QA | `create_qa_submission.py` | **3–8 giờ** (Ollama local) |

**Một lệnh gói cả pipeline:**

```bash
bash scripts/run_full_2000_pipeline.sh
```

Tùy chọn bỏ qua bước (khi đã có artifact):

```bash
SKIP_CACHE=1 bash scripts/run_full_2000_pipeline.sh   # giữ cache, chạy lại IR+QA
SKIP_IR=1 bash scripts/run_full_2000_pipeline.sh      # chỉ sinh lại QA
SKIP_QA=1 bash scripts/run_full_2000_pipeline.sh      # chỉ IR
```

---

## 2. Chuẩn bị môi trường

### 2.1. Phần cứng / phần mềm

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị (full pipeline) |
|------------|-------------------|----------------------------|
| Python | 3.11 – 3.14 | 3.11+ |
| RAM | 16 GB | 32 GB+ |
| GPU | Không bắt buộc (chậm) | NVIDIA CUDA (cache + reranker) |
| Disk | 8 GB trống | 15 GB+ |
| Docker | Qdrant | Qdrant + snapshot |
| Ollama | ≥ 0.5 | `qwen3:4b-instruct` |

### 2.2. Cài đặt Python & clone repo

**macOS / Linux (Terminal):**

```bash
git clone https://github.com/Naammmdz/lexi-agent.git
cd lexi-agent
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

**Windows (PowerShell hoặc CMD):**

```powershell
git clone https://github.com/Naammmdz/lexi-agent.git
cd lexi-agent
py -3 -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

> Cần cài sẵn: [Git](https://git-scm.com/), [Python 3.11+](https://www.python.org/downloads/) (tick **Add Python to PATH** trên Windows).

### 2.3. Tải dữ liệu từ Google Drive

**Link:** https://drive.google.com/drive/folders/1yrTBTV-pTdS2FObe1shBHiYmMPsxhazH?usp=drive_link

Folder gồm **`data/`** và **`index/`**. Sau khi tải, copy vào thư mục `lexi-agent/` sao cho có:

```
lexi-agent/data/corpus/legal_corpus_merged.json
lexi-agent/index/bm25_index_merged.pkl
```

#### macOS

1. Mở link Drive → chuột phải folder `data` → **Download** (Google có thể nén thành `.zip`).
2. Làm tương tự folder `index`.
3. Giải nén (double-click `.zip` nếu có).
4. Copy vào repo:

```bash
cd ~/Downloads/lexi-agent   # hoặc đường dẫn bạn đã clone
# Thay DOWNLOADS bằng nơi file tải về, ví dụ ~/Downloads
cp -R ~/Downloads/data ~/Downloads/index .
```

Kiểm tra:

```bash
test -f data/corpus/legal_corpus_merged.json && test -f index/bm25_index_merged.pkl && echo OK
```

#### Windows

1. Mở link Drive → tải folder `data` và `index` (Drive thường tải dạng `.zip`).
2. Giải nén trong File Explorer.
3. Copy vào repo (PowerShell, đổi đường dẫn `Downloads` nếu cần):

```powershell
cd C:\Users\TEN_BAN\lexi-agent
Copy-Item -Recurse -Force $env:USERPROFILE\Downloads\data .
Copy-Item -Recurse -Force $env:USERPROFILE\Downloads\index .
```

Kiểm tra:

```powershell
Test-Path data\corpus\legal_corpus_merged.json
Test-Path index\bm25_index_merged.pkl
```

**Đã có trên Git** (không cần Drive): `R2AIStage1DATA.json`, `submission.zip`, `submission_qa.zip`.

**Thay thế:** build lại corpus/index — [01_MO_TA_DU_LIEU.md](01_MO_TA_DU_LIEU.md).

### 2.4. Docker (Qdrant) & Ollama

Cài [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac/Windows) và [Ollama](https://ollama.com/download).

**Qdrant — macOS / Linux / Windows (PowerShell, Docker Desktop đang chạy):**

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

Lần sau khởi động lại: `docker start qdrant`

**Build Qdrant index lần đầu** (sau khi có `data/` từ Drive):

```bash
# Mac/Linux — venv đã activate
python setup_system.py --rebuild

# Windows — venv đã activate
python setup_system.py --rebuild
```

**Ollama:**

| | macOS / Linux | Windows |
|---|---------------|---------|
| Cài đặt | https://ollama.com/download | Cùng link — cài app |
| Chạy server | `ollama serve` (terminal) hoặc app menu bar | App tự chạy nền sau khi cài |
| Tải model | `ollama pull qwen3:4b-instruct` | `ollama pull qwen3:4b-instruct` |

### 2.5. Kiểm tra trước khi chạy

**macOS / Linux:**

```bash
python verify_setup.py
python -c "import json; print(len(json.load(open('R2AIStage1DATA.json'))), 'questions')"
curl -s http://127.0.0.1:6333/collections | head -c 200
test -f data/corpus/legal_corpus_merged.json && echo corpus OK
```

**Windows (PowerShell):**

```powershell
python verify_setup.py
python -c "import json; print(len(json.load(open('R2AIStage1DATA.json'))), 'questions')"
curl http://127.0.0.1:6333/collections
Test-Path data\corpus\legal_corpus_merged.json
```

Kết quả mong đợi: `verify_setup.py` pass, **2000 questions**, Qdrant phản hồi.

### 2.6. Cấu hình IR tốt nhất

Thêm vào `.env` (hoặc `export` / `$env:` trước khi chạy):

**macOS / Linux:**

```bash
USE_MERGED_CORPUS=1
HYBRID_FUSION=rrf
USE_WIDE_RETRIEVAL_POOL=1
ENABLE_RERANKING=1          # bắt buộc cho điểm IR ~0.631
RERANKER_DEVICE=cuda        # Mac Apple Silicon: mps; không GPU: cpu
```

**Windows (PowerShell):**

```powershell
$env:USE_MERGED_CORPUS="1"
$env:HYBRID_FUSION="rrf"
$env:USE_WIDE_RETRIEVAL_POOL="1"
$env:ENABLE_RERANKING="1"
$env:RERANKER_DEVICE="cuda"   # hoặc cpu
```

> **CẢNH BÁO — `ENABLE_RERANKING=1` (bắt buộc)**
>
> - Phải đặt **trước** bước ① `cache_live_retrieval.py` và giữ nguyên khi chạy `run_full_2000_pipeline.sh`.
> - Nếu log in ra `Reranking disabled in configuration` → cache **sai**, điểm IR sẽ **không** ~0.631. Dừng lại, `export ENABLE_RERANKING=1` (hoặc ghi vào `.env`), xóa cache cũ rồi chạy lại bước ①.
> - Log đúng phải có: `Reranker model loaded successfully`.
> - Không dùng `SKIP_CACHE=1` trừ khi file `data/augmented/live_retrieval_rrf_wide_merged.json` đã được build **cùng cấu hình** (reranker bật, merged corpus, RRF wide).
> - `--limit N` chỉ để pilot — **không** thay cho chạy full 2000 câu khi nghiệm thu.

---

## 3. Chạy full pipeline (khuyến nghị)

### macOS / Linux (Terminal)

```bash
cd lexi-agent
source venv/bin/activate
export USE_MERGED_CORPUS=1 HYBRID_FUSION=rrf USE_WIDE_RETRIEVAL_POOL=1
export ENABLE_RERANKING=1          # bắt buộc — xem cảnh báo mục 2.6
export RERANKER_DEVICE=mps       # Mac Apple Silicon; Linux GPU: cuda; CPU: cpu

bash scripts/run_full_2000_pipeline.sh
```

> Kiểm tra nhanh sau bước ①: mở log, tìm `Reranker model loaded successfully`. Nếu thấy `Reranking disabled` → dừng và sửa biến môi trường.

### Windows

**Cách 1 — Git Bash** (cài cùng Git for Windows): mở Git Bash trong folder `lexi-agent`, chạy lệnh giống macOS ở trên.

**Cách 2 — PowerShell** (chạy từng bước, tương đương script):

```powershell
cd lexi-agent
venv\Scripts\activate
$env:USE_MERGED_CORPUS="1"
$env:HYBRID_FUSION="rrf"
$env:USE_WIDE_RETRIEVAL_POOL="1"
$env:ENABLE_RERANKING="1"
$env:RERANKER_DEVICE="cuda"    # hoặc cpu nếu không có NVIDIA GPU

python tools\submission\cache_live_retrieval.py --input R2AIStage1DATA.json --output data\augmented\live_retrieval_rrf_wide_merged.json --mapping data\law_id_to_title_merged.json --resume

python tools\submission\create_cache_only_submission.py --input R2AIStage1DATA.json --cache data\augmented\live_retrieval_rrf_wide_merged.json --output submission_variants\submission_cache_seed_top1.zip

python tools\submission\create_recall_boost_submission.py --base submission_variants\submission_cache_seed_top1.zip --output submission_variants\submission_recall_boost_merged_vn_rerank_no_wl_tight_v1.zip --cap-articles 2 --cap-docs 1 --article-same-law-only --article-min-score 0.9 --article-min-gap-from-top1 0.03 --live-cache data\augmented\live_retrieval_rrf_wide_merged.json --mapping data\law_id_to_title_merged.json

python tools\submission\create_rrf_zone_swap_submission.py --base submission_variants\submission_recall_boost_merged_vn_rerank_no_wl_tight_v1.json --output submission_variants\rrf_swap_g008.zip --zone-cache data\augmented\live_retrieval_rrf_wide_merged.json --mapping data\law_id_to_title_merged.json --replace-min-gap 0.03 --article-min-score 0.9

copy submission_variants\rrf_swap_g008.zip submission.zip

$env:OLLAMA_MODEL="qwen3:4b-instruct"
python tools\submission\create_qa_submission.py --base submission.zip --corpus data\corpus\legal_corpus_merged.json --backend ollama --model qwen3:4b-instruct --batch-size 8 --max-articles 3 --resume --output submission_variants\qa_promote_g008_ollama.zip

copy submission_variants\qa_promote_g008_ollama.zip submission_qa.zip
```

Log (Mac/Linux): `/tmp/r2ai_full_2000_pipeline.log`

### Kết quả

| File | Nội dung | Số dòng |
|------|----------|---------|
| `submission.zip` | `id`, `question`, `relevant_docs`, `relevant_articles`, `answer` (placeholder) | **2000** |
| `submission_qa.zip` | Giống trên + `answer` grounded đầy đủ | **2000** |

### Kiểm tra sau khi chạy

**macOS / Linux:**

```bash
python3 <<'PY'
import json, zipfile
for name in ("submission.zip", "submission_qa.zip"):
    rows = json.loads(zipfile.ZipFile(name).read("results.json"))
    arts = sum(1 for r in rows if r.get("relevant_articles"))
    qa = sum(1 for r in rows if r.get("answer") and "Căn cứ pháp luật:" in r["answer"])
    print(f"{name}: total={len(rows)}  articles={arts}  qa_format={qa}")
    assert len(rows) == 2000, f"{name}: thiếu câu"
print("OK")
PY
```

**Windows (PowerShell):**

```powershell
python -c @"
import json, zipfile
for name in ('submission.zip', 'submission_qa.zip'):
    rows = json.loads(zipfile.ZipFile(name).read('results.json'))
    print(f'{name}: total={len(rows)}')
    assert len(rows) == 2000
print('OK')
"@
```

So sánh với bản tham chiếu trên Git (tùy chọn — sau khi build xong):

```bash
git show HEAD:submission.zip > /tmp/submission_ref.zip
python3 -c "
import json, zipfile
def arts(p):
    r=json.loads(zipfile.ZipFile(p).read('results.json'))
    return {x['id']: tuple(x.get('relevant_articles') or []) for x in r}
g,a=arts('/tmp/submission_ref.zip'), arts('submission.zip')
diff=sum(1 for i in g if g.get(i)!=a.get(i))
print(f'article field diffs: {diff}/2000')
print('OK match' if diff==0 else 'MISMATCH — kiểm tra ENABLE_RERANKING và chạy lại cache từ đầu')
"
```

Nếu `MISMATCH` nhiều (ví dụ >100 dòng): thường do cache thiếu reranker hoặc zone swap không áp dụng — **không** dùng file vừa build để nộp; chạy lại full pipeline từ bước ①.

### Điểm khi nộp lên BTC

| File | Metric |
|------|--------|
| `submission.zip` | **ARTICLES_F2 ≈ 0.6308** (cùng config reranker + zone swap gap 0.03) |
| `submission_qa.zip` | IR giữ nguyên ~0.631; QA chấm **5 tiêu chí** khi promote (không có F2 trong repo) |

> **Lưu ý:** Nếu chạy trên CPU / tắt reranker / cache khác máy benchmark, điểm IR có thể lệch. So sánh với `submission.zip` tham chiếu trên Git để đối chiếu.

---

## 4. Chạy từng bước (chi tiết)

### Bước ① — Cache retrieval (2000 câu)

```bash
export USE_MERGED_CORPUS=1 HYBRID_FUSION=rrf USE_WIDE_RETRIEVAL_POOL=1 ENABLE_RERANKING=1

venv/bin/python tools/submission/cache_live_retrieval.py \
  --input R2AIStage1DATA.json \
  --output data/augmented/live_retrieval_rrf_wide_merged.json \
  --mapping data/law_id_to_title_merged.json \
  --resume
```

- `--resume`: tiếp tục nếu bị ngắt giữa chừng
- Pilot: thêm `--limit 10` để test 10 câu đầu

### Bước ② — Seed submission từ cache

```bash
venv/bin/python tools/submission/create_cache_only_submission.py \
  --input R2AIStage1DATA.json \
  --cache data/augmented/live_retrieval_rrf_wide_merged.json \
  --output submission_variants/submission_cache_seed_top1.zip \
  --cap-articles 1 --cap-docs 1
```

Không cần `submission.zip` có sẵn — script đọc trực tiếp `R2AIStage1DATA.json`.

### Bước ③ — Recall boost

```bash
venv/bin/python tools/submission/create_recall_boost_submission.py \
  --base submission_variants/submission_cache_seed_top1.zip \
  --output submission_variants/submission_recall_boost_merged_vn_rerank_no_wl_tight_v1.zip \
  --cap-articles 2 --cap-docs 1 --article-same-law-only \
  --article-min-score 0.9 --article-min-gap-from-top1 0.03 \
  --live-cache data/augmented/live_retrieval_rrf_wide_merged.json \
  --mapping data/law_id_to_title_merged.json
```

### Bước ④ — Zone swap → `submission.zip`

**Zone swap** là bước hậu xử lý: thay **điều luật top-1** bằng kết quả reranker cache **chỉ khi** score ≥ 0.9 và chênh lệch ≥ 0.03 so với top-1 hiện tại (~60% câu được đổi trong bản đã nộp).

```bash
venv/bin/python tools/submission/create_rrf_zone_swap_submission.py \
  --base submission_variants/submission_recall_boost_merged_vn_rerank_no_wl_tight_v1.json \
  --output submission_variants/rrf_swap_g008.zip \
  --zone-cache data/augmented/live_retrieval_rrf_wide_merged.json \
  --mapping data/law_id_to_title_merged.json \
  --replace-min-gap 0.03 \
  --article-min-score 0.9

cp submission_variants/rrf_swap_g008.zip submission.zip
```

### Bước ⑤ — Sinh answer QA → `submission_qa.zip`

```bash
export OLLAMA_MODEL=qwen3:4b-instruct
export OLLAMA_WORKERS=6

venv/bin/python tools/submission/create_qa_submission.py \
  --base submission.zip \
  --corpus data/corpus/legal_corpus_merged.json \
  --backend ollama \
  --model qwen3:4b-instruct \
  --batch-size 8 \
  --max-articles 3 \
  --max-chars-per-article 1200 \
  --max-new-tokens 1200 \
  --resume \
  --output submission_variants/qa_promote_g008_ollama.zip

cp submission_variants/qa_promote_g008_ollama.zip submission_qa.zip
```

Pilot 5 câu:

```bash
venv/bin/python tools/submission/create_qa_submission.py \
  --base submission.zip \
  --corpus data/corpus/legal_corpus_merged.json \
  --backend ollama --model qwen3:4b-instruct \
  --limit 5 --output submission_variants/qa_pilot5.zip
```

---

## 5. Cách thay thế: một lệnh (`test_r2ai_pipeline.py`)

Script end-to-end đơn giản hơn — retrieve + answer trong một vòng lặp, **không** có recall boost / zone swap:

```bash
export USE_MERGED_CORPUS=1 HYBRID_FUSION=rrf USE_WIDE_RETRIEVAL_POOL=1
python test_r2ai_pipeline.py
```

**Output:** `results.json`, `submission.zip` (2000 dòng).

| | `test_r2ai_pipeline.py` | Pipeline đầy đủ (mục 3) |
|--|-------------------------|-------------------------|
| Từ `R2AIStage1DATA.json` | Có | Có |
| Điểm IR ~0.631 | Thường **không** | **Có** (cùng config) |
| Có `submission_qa.zip` | Không (chỉ 1 file) | Có |

Dùng cho kiểm tra nhanh format; **nghiệm thu khuyến nghị mục 3**.

---

## 6. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|-------------|-------------|------------|
| `Connection refused` :6333 | Qdrant chưa chạy | Mac/Linux: `docker start qdrant`. Windows: mở Docker Desktop rồi `docker start qdrant` |
| `Connection refused` Ollama | Ollama chưa chạy | Mac/Linux: `ollama serve`. Windows: mở app Ollama hoặc cài lại từ ollama.com |
| Cache dừng giữa chừng | Mất session / sleep | Chạy lại bước ① với `--resume` |
| QA dừng giữa chừng | Timeout / RAM | Chạy lại bước ⑤ với `--resume` |
| `corpus not found` | Thiếu merged JSON | Tải từ Drive |
| Log `Reranking disabled` | Quên `ENABLE_RERANKING=1` | Export đủ biến mục 2.6, xóa cache, chạy lại bước ① |
| `submission.zip` khác bản Git | Cache cũ / thiếu reranker / `SKIP_CACHE=1` sai | So sánh script mục 3; build lại cache full 2000 |
| Chậm / OOM | Reranker + embed cùng lúc | Chạy cache trên GPU server; QA tách process |
| `total != 2000` | `--limit` còn trong lệnh | Bỏ `--limit`, chạy lại full |

---

## 7. Định dạng `results.json` (bài nộp)

Mỗi phần tử trong mảng JSON:

```json
{
  "id": 1,
  "question": "Câu hỏi tiếng Việt...",
  "answer": "Căn cứ pháp luật: ... Trả lời: ... Lưu ý: ...",
  "relevant_docs": ["59/2020/QH14|Luật Doanh nghiệp số 59/2020/QH14"],
  "relevant_articles": ["59/2020/QH14|Luật Doanh nghiệp số 59/2020/QH14|Điều 4"]
}
```

- `relevant_docs`: `mã_văn_bản|tên_văn_bản`
- `relevant_articles`: `mã_văn_bản|tên_văn_bản|Điều N`
- File nộp: ZIP phẳng chứa `results.json` (không thư mục con)

---

## 8. Tài liệu liên quan

- [00_GIOI_THIEU.md](00_GIOI_THIEU.md) — Giới thiệu sản phẩm
- [01_MO_TA_DU_LIEU.md](01_MO_TA_DU_LIEU.md) — Mô tả dữ liệu
- [02_MO_HINH.md](02_MO_HINH.md) — Mô hình
- [03_MA_NGUON_VA_CAU_HINH.md](03_MA_NGUON_VA_CAU_HINH.md) — Mã nguồn
- [THUYET_MINH_SAN_PHAM.md](THUYET_MINH_SAN_PHAM.md) — Checklist nộp Drive/Git
