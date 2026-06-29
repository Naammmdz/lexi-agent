# Hướng dẫn giám khảo — chạy lại bài nộp 2000 câu

Tài liệu **duy nhất** để tái hiện kết quả cuộc thi R2AI. Làm **đủ các bước bên dưới theo thứ tự**, không bỏ bước, không chạy “tắt” để nhanh.

**Mục tiêu:** Từ file 2000 câu hỏi có sẵn trên Git → tự tạo lại hai file nộp:
- `submission.zip` — phần tìm điều luật (điểm IR ~0.63)
- `submission_qa.zip` — thêm câu trả lời đầy đủ cho 2000 câu

**Thời gian ước tính (máy có GPU):** khoảng **8–14 giờ** (bước tìm kiếm ~3–6 giờ, bước viết trả lời ~3–8 giờ). Máy chỉ có CPU sẽ lâu hơn nhiều.

---

## Cần tải về đâu?

| Nguồn | Link | Nội dung |
|-------|------|----------|
| **GitHub** | https://github.com/Naammmdz/lexi-agent | Mã nguồn, hướng dẫn, file 2000 câu hỏi, bản kết quả tham chiếu |
| **Google Drive** | https://drive.google.com/drive/folders/1yrTBTV-pTdS2FObe1shBHiYmMPsxhazH?usp=drive_link | Hai thư mục `data` và `index` (~3 GB) |

Trên Git **đã có sẵn** (sau khi clone): `R2AIStage1DATA.json`, `submission.zip`, `submission_qa.zip` (để đối chiếu).  
Trên Drive **phải tải thêm**: `data/` và `index/` (file nặng, không để trên Git).

---

## Máy cần cài gì?

| Phần mềm | Ghi chú |
|----------|---------|
| Git | Tải mã nguồn |
| Python 3.11 trở lên | Windows: tick “Add Python to PATH” khi cài |
| Docker Desktop | Chạy cơ sở dữ liệu vector Qdrant |
| Ollama | Chạy mô hình trả lời `qwen3:4b-instruct` |

**Khuyến nghị:** RAM ≥ 16 GB (nên 32 GB), ổ trống ≥ 15 GB, có GPU NVIDIA (Windows/Linux) hoặc Mac Apple Silicon.

---

## Bước 1 — Tải mã nguồn

**Mac / Linux (Terminal):**

```bash
git clone https://github.com/Naammmdz/lexi-agent.git
cd lexi-agent
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/Naammmdz/lexi-agent.git
cd lexi-agent
```

---

## Bước 2 — Cài thư viện Python

**Mac / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

**Windows:**

```powershell
py -3 -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

---

## Bước 3 — Tải dữ liệu từ Google Drive

1. Mở link Drive ở bảng trên.
2. Tải folder **`data`** và **`index`** (thường thành file `.zip`).
3. Giải nén.
4. Copy **cả hai folder** vào trong thư mục `lexi-agent` (cùng cấp với `README.md`).

Sau bước này phải có:
- `lexi-agent/data/corpus/legal_corpus_merged.json`
- `lexi-agent/index/bm25_index_merged.pkl`

**Kiểm tra nhanh — Mac/Linux:**

```bash
test -f data/corpus/legal_corpus_merged.json && test -f index/bm25_index_merged.pkl && echo "Dữ liệu OK"
```

**Windows (PowerShell):**

```powershell
Test-Path data\corpus\legal_corpus_merged.json
Test-Path index\bm25_index_merged.pkl
```

Cả hai lệnh đều phải trả về đúng / `True`.

---

## Bước 4 — Chạy Qdrant (Docker)

Mở **Docker Desktop** (Mac/Windows) cho đến khi báo đang chạy.

**Mac / Linux / Windows (PowerShell):**

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

Lần sau mở máy, chỉ cần:

```bash
docker start qdrant
```

---

## Bước 5 — Cài Ollama và tải mô hình trả lời

1. Cài từ https://ollama.com/download  
2. Tải mô hình:

```bash
ollama pull qwen3:4b-instruct
```

- **Mac:** có thể chạy thêm `ollama serve` trong terminal, hoặc dùng app.
- **Windows:** sau khi cài app, Ollama thường tự chạy nền.

---

## Bước 6 — Tạo chỉ mục vector (lần đầu, ~30–90 phút)

Vẫn trong thư mục `lexi-agent`, môi trường ảo đã bật (`venv`):

```bash
python setup_system.py --rebuild
```

Chờ đến khi chạy xong không báo lỗi. Bước này **chỉ làm một lần** sau khi đã có `data/` từ Drive.

---

## Bước 7 — Kiểm tra trước khi chạy pipeline

```bash
python verify_setup.py
python -c "import json; print(len(json.load(open('R2AIStage1DATA.json'))), 'câu hỏi')"
```

Kết quả mong đợi:
- `verify_setup.py` báo OK
- In ra **`2000 câu hỏi`**

---

## Bước 8 — Chạy toàn bộ quy trình (quan trọng nhất)

Đây là bước **bắt buộc** theo quy định R2AI: chạy từ đầu đến cuối, **không bỏ bước tìm kiếm** để tiết kiệm thời gian.

### Mac / Linux

```bash
cd lexi-agent
source venv/bin/activate

export USE_MERGED_CORPUS=1
export HYBRID_FUSION=rrf
export USE_WIDE_RETRIEVAL_POOL=1
export ENABLE_RERANKING=1
export RERANKER_DEVICE=mps
```

> `RERANKER_DEVICE`: Mac Apple Silicon dùng `mps`; máy có card NVIDIA dùng `cuda`; không có GPU dùng `cpu` (rất chậm).

Chạy một lệnh:

```bash
bash scripts/run_full_2000_pipeline.sh
```

### Windows

**Cách 1 — Git Bash** (cài kèm Git for Windows): mở Git Bash trong folder `lexi-agent`, chạy **y hệt** các lệnh Mac ở trên.

**Cách 2 — PowerShell:** xem danh sách lệnh từng bước trong [04_HUONG_DAN_TAI_HIEN_2000_CAU.md](04_HUONG_DAN_TAI_HIEN_2000_CAU.md) mục 3 (nội dung giống script trên, chỉ khác cách gõ đường dẫn Windows).

### Quy trình gồm 5 phần (tự chạy trong script)

| Phần | Việc làm | Thời gian |
|------|----------|-----------|
| 1 | Tìm điều luật cho 2000 câu hỏi | 3–6 giờ |
| 2 | Gom kết quả thành file nộp sơ bộ | vài giây |
| 3 | Bổ sung thêm điều luật liên quan | vài giây |
| 4 | Chỉnh lại điều luật chính ở một số câu (để đạt điểm ~0.63) | vài giây |
| 5 | Viết câu trả lời đầy đủ bằng Ollama | 3–8 giờ |

**Cảnh báo — đọc kỹ:**

- Phải bật `ENABLE_RERANKING=1` **trước** khi chạy script. Nếu trong log thấy dòng *“Reranking disabled”* → **dừng lại**, sửa cấu hình, xóa file cache cũ trong `data/augmented/live_retrieval_rrf_wide_merged.json` (nếu có), chạy lại từ đầu.
- Log đúng phải có dòng *“Reranker model loaded successfully”*.
- **Không** dùng chế độ bỏ qua bước 1 (trong code là `SKIP_CACHE=1`) khi nghiệm thu — kết quả sẽ **khác** bản đã nộp.
- Nếu bị ngắt giữa chừng: chạy lại **cùng lệnh** — script hỗ trợ tiếp tục (`--resume`).

Log chi tiết (Mac/Linux): `/tmp/r2ai_full_2000_pipeline.log`

---

## Bước 9 — Kiểm tra kết quả

```bash
python3 <<'PY'
import json, zipfile
for name in ("submission.zip", "submission_qa.zip"):
    rows = json.loads(zipfile.ZipFile(name).read("results.json"))
    print(f"{name}: {len(rows)} dòng")
    assert len(rows) == 2000
print("Đủ 2000 dòng — OK")
PY
```

**Đối chiếu với bản trên Git** (nên làm):

```bash
git show HEAD:submission.zip > /tmp/ban_tham_chieu.zip
python3 -c "
import json, zipfile
def lay_dieu_luat(path):
    r = json.loads(zipfile.ZipFile(path).read('results.json'))
    return {x['id']: tuple(x.get('relevant_articles') or []) for x in r}
g, m = lay_dieu_luat('/tmp/ban_tham_chieu.zip'), lay_dieu_luat('submission.zip')
khac = sum(1 for i in g if g[i] != m.get(i))
print(f'Số câu khác điều luật: {khac}/2000')
print('Khớp bản tham chiếu' if khac == 0 else 'CHƯA KHỚP — cần chạy lại từ bước 8, đủ 5 phần')
"
```

- **Khớp** → tái hiện thành công.
- **Khác nhiều** (ví dụ > 100 câu) → thường do bỏ bước 1 hoặc thiếu cấu hình xếp hạng lại; **không** coi là tái hiện đúng.

---

## Bước 10 — (Tùy chọn) Chạy giao diện demo Lexi

Không bắt buộc cho chấm bài nộp R2AI.

```bash
PYTHONUNBUFFERED=1 UI_MODE=lexi python app.py
```

Mở trình duyệt: http://127.0.0.1:7860

---

## Kết quả mong đợi

| File | Ý nghĩa | Điểm / ghi chú |
|------|---------|----------------|
| `submission.zip` | 2000 câu + điều luật tìm được | ARTICLES_F2 ~ **0.6308** |
| `submission_qa.zip` | Giống trên + câu trả lời đầy đủ | 2000/2000 câu có trả lời |

---

## Lỗi thường gặp

| Triệu chứng | Cách xử lý |
|-------------|------------|
| Không kết nối được cổng 6333 | Mở Docker Desktop → `docker start qdrant` |
| Ollama không phản hồi | Mac: `ollama serve`; Windows: mở app Ollama |
| Báo thiếu corpus / file dữ liệu | Tải lại `data/` và `index/` từ Drive (bước 3) |
| Log “Reranking disabled” | Đặt `ENABLE_RERANKING=1`, xóa cache cũ, chạy lại bước 8 |
| Chạy giữa chừng bị tắt máy | Chạy lại cùng lệnh bước 8 (tự tiếp tục) |
| Kết quả khác bản Git nhiều | Chạy lại **đủ** bước 8, **không** bỏ phần 1 |
| Máy báo hết RAM | Dùng máy có GPU; hoặc chạy phần 1 và phần 5 trên máy mạnh hơn |

---

## Tài liệu bổ sung (không bắt buộc đọc khi tái hiện)

Chỉ khi cần hiểu sâu hơn về dữ liệu, mô hình, hoặc cấu trúc mã:

- [THUYET_MINH_SAN_PHAM.md](THUYET_MINH_SAN_PHAM.md) — checklist nộp bài
- [01_MO_TA_DU_LIEU.md](01_MO_TA_DU_LIEU.md) — mô tả dữ liệu
- [02_MO_HINH.md](02_MO_HINH.md) — mô tả mô hình
- [04_HUONG_DAN_TAI_HIEN_2000_CAU.md](04_HUONG_DAN_TAI_HIEN_2000_CAU.md) — chi tiết kỹ thuật / lệnh Windows PowerShell
