# R2AI Submission Tools

Các script trong thư mục này phục vụ tạo và kiểm tra bài nộp R2AI. Tất cả đều tự trỏ về repo root, nên chạy từ root repo bằng `venv/bin/python tools/submission/<script>.py`.

## Augmented corpus (merge Zalo + vbpl.vn)

Nguồn legal DB công khai để bổ sung văn bản thiếu:

**https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents** (vbpl.vn, CC BY 4.0)

1. Thu thập gap (QH / NĐ-CP / Thông tư chưa có trong Zalo):

```bash
venv/bin/python tools/corpus/build_hf_vbpl_gap_corpus.py
```

Output: `data/augmented/hf_vbpl_gap_corpus.json`

2. Merge vào corpus chính:

```bash
venv/bin/python tools/corpus/build_merged_corpus_v2.py \
  --gap data/augmented/hf_vbpl_gap_corpus.json
```

Output: `data/corpus/legal_corpus_merged.json`, `data/law_id_to_title_merged.json`

Pipeline đầy đủ: `bash tools/corpus/run_p4_corpus_pipeline.sh`

### Legacy (dev nội bộ)

- `create_augmented_corpus_from_db.py` — Postgres `legal_db` local (không bắt buộc nghiệm thu)
- `build_vlsp_bulk_gap_corpus.py` — thử nghiệm VLSP2025 (đã thay bằng HF vbpl)

## Legacy / Comparison Tools

- `create_method_submission.py`: tạo variant từ BM25/hybrid/hybrid_rerank cũ.
- `create_submission_variants.py`: cắt top-k từ `results.json` có sẵn.
- `create_strict_answer_submission.py`: giữ retrieval top1 và rewrite answer ngắn.
- `normalize_submission.py`: normalize refs của `results.json` rồi rebuild `submission.zip`.
- `submission_benchmark.py`: benchmark train_qna kiểu leaderboard cho pipeline cũ.
