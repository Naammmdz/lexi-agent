# R2AI Submission Tools

Script tạo và kiểm tra bài nộp R2AI. Chạy từ thư mục gốc repo:

```bash
venv/bin/python tools/submission/<script>.py
```

---

## Tái hiện full 2000 câu (nghiệm thu)

Repo: https://github.com/Naammmdz/lexi-agent

```bash
git clone https://github.com/Naammmdz/lexi-agent.git
cd lexi-agent
bash scripts/run_full_2000_pipeline.sh
```

### Luồng từng bước

| # | Script | Output |
|---|--------|--------|
| 1 | `cache_live_retrieval.py` | `data/augmented/live_retrieval_rrf_wide_merged.json` |
| 2 | `create_cache_only_submission.py` | Seed IR (không cần `submission.zip` cũ) |
| 3 | `create_recall_boost_submission.py` | `no_wl_tight_v1` variant |
| 4 | `create_rrf_zone_swap_submission.py` | `submission.zip` (~0.631 F2) |
| 5 | `create_qa_submission.py` | `submission_qa.zip` (2000 answer) |

Hướng dẫn đầy đủ: [docs/04_HUONG_DAN_TAI_HIEN_2000_CAU.md](../../docs/04_HUONG_DAN_TAI_HIEN_2000_CAU.md)

---

## Corpus (merge Zalo + vbpl.vn)

Nguồn công khai: [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents)

```bash
venv/bin/python tools/corpus/build_hf_vbpl_gap_corpus.py
venv/bin/python tools/corpus/build_merged_corpus_v2.py --gap data/augmented/hf_vbpl_gap_corpus.json
bash tools/corpus/run_p4_corpus_pipeline.sh
```

---

## Script khác

| Script | Mục đích |
|--------|----------|
| `test_r2ai_pipeline.py` (root) | RAG end-to-end một lệnh |
| `benchmark_companion_candidate.py` | Audit offline companion |
| `submission_benchmark.py` | Benchmark train_qna |
| `clean_for_submit.py` | Dọn artifact trước nộp |
