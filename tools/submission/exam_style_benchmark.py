"""Independent benchmark for the DB-augmented R2AI ranker.

This is intentionally separate from augmented_submission_benchmark.py. The old
pseudo benchmark uses the current ranker as teacher, so it catches regressions
but can be over-optimistic. This benchmark has two independent parts:

1. train_core: labeled train_qna.csv rows whose gold law appears in the current
   DB-augmented core corpus.
2. manual_r2ai_like: compact high-confidence scenarios created from legal
   provisions and recurring R2AI question patterns.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from _paths import REPO_ROOT
from create_augmented_submission import (
    INPUT_PATH,
    build_article_lookup,
    build_or_load_index,
    load_augmented_articles,
    rank_article,
    resolve_article,
)
from utils.submission_formatter import canonical_law_id


BASE_DIR = REPO_ROOT
TRAIN_PATH = BASE_DIR / "data" / "train" / "train_qna.csv"
OUTPUT_PATH = BASE_DIR / "submission_variants" / "exam_style_benchmark.json"


MANUAL_CASES = [
    ("SME incubator tax land", "Các cơ sở ươm tạo và khu làm việc chung được hưởng hỗ trợ gì về thuế, đất đai?", [("04/2017/QH14", "12")]),
    ("SME procurement", "Doanh nghiệp nhỏ và vừa được hưởng ưu đãi gì khi tham gia đấu thầu?", [("63/2014/NĐ-CP", "6")]),
    ("Labor original certificate", "Công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì bị xử lý thế nào?", [("12/2022/NĐ-CP", "9")]),
    ("SME household conversion", "Hộ kinh doanh cần điều kiện gì để được hỗ trợ chuyển đổi thành doanh nghiệp nhỏ và vừa?", [("04/2017/QH14", "16")]),
    ("SME production premises", "Doanh nghiệp nhỏ và vừa được hỗ trợ giá thuê mặt bằng sản xuất tối đa bao lâu?", [("04/2017/QH14", "11")]),
    ("SME financial report", "Báo cáo tài chính năm của doanh nghiệp nhỏ và vừa phải trình bày thông tin chung nào?", [("133/2016/TT-BTC", "81")]),
    ("SME startup condition", "Doanh nghiệp nhỏ và vừa khởi nghiệp sáng tạo được xác định theo tiêu chí nào?", [("80/2021/NĐ-CP", "20")]),
    ("SME value chain support", "Doanh nghiệp nhỏ và vừa tham gia chuỗi giá trị được hỗ trợ những nội dung gì?", [("80/2021/NĐ-CP", "25")]),
    ("SME value chain selection", "Doanh nghiệp nhỏ và vừa tham gia chuỗi giá trị được lựa chọn hỗ trợ theo tiêu chí nào?", [("80/2021/NĐ-CP", "24")]),
    ("SME cluster selection", "Cụm liên kết ngành và doanh nghiệp nhỏ và vừa tham gia cụm được lựa chọn hỗ trợ theo tiêu chí nào?", [("80/2021/NĐ-CP", "23")]),
    ("SME fund tasks", "Quỹ phát triển doanh nghiệp nhỏ và vừa có nhiệm vụ, quyền hạn gì?", [("39/2019/NĐ-CP", "5")]),
    ("SME criteria", "Tiêu chí xác định doanh nghiệp nhỏ và vừa theo số lao động và doanh thu là gì?", [("80/2021/NĐ-CP", "5")]),
    ("SME credit access", "Doanh nghiệp nhỏ và vừa được hỗ trợ gì để nâng cao khả năng tiếp cận tín dụng?", [("04/2017/QH14", "8")]),
    ("Credit guarantee", "Quỹ bảo lãnh tín dụng căn cứ điều kiện nào để cấp bảo lãnh cho doanh nghiệp nhỏ và vừa?", [("34/2018/NĐ-CP", "16")]),
    ("SME counterpart", "Doanh nghiệp nhỏ và vừa nhận hỗ trợ có trách nhiệm gì về nguồn lực đối ứng?", [("04/2017/QH14", "28")]),
    ("SME funding sources", "Nguồn vốn hỗ trợ doanh nghiệp nhỏ và vừa gồm những nguồn nào?", [("04/2017/QH14", "6")]),
    ("Training types", "Khóa đào tạo trực tiếp về quản trị doanh nghiệp cho doanh nghiệp nhỏ và vừa gồm những loại nào?", [("05/2019/TT-BKHĐT", "3")]),
    ("Training audience", "Đối tượng học viên tham gia khóa đào tạo trực tiếp về khởi sự kinh doanh là ai?", [("05/2019/TT-BKHĐT", "3")]),
    ("Consultant cost", "Chi phí nào của tư vấn viên được ngân sách nhà nước hỗ trợ khi doanh nghiệp nhỏ và vừa nhận tư vấn?", [("54/2019/TT-BTC", "7")]),
    ("Ecommerce account support", "Doanh nghiệp nhỏ và vừa khởi nghiệp sáng tạo được hỗ trợ tài khoản trên sàn thương mại điện tử quốc tế như thế nào?", [("80/2021/NĐ-CP", "22")]),
    ("Support application", "Hồ sơ đề xuất hỗ trợ doanh nghiệp nhỏ và vừa gồm những thành phần nào?", [("80/2021/NĐ-CP", "32")]),
    ("Digital solution", "Doanh nghiệp nhỏ và vừa thuê hoặc mua giải pháp chuyển đổi số được hỗ trợ chi phí ở đâu?", [("80/2021/NĐ-CP", "11")]),
    ("Tax registration scope", "Phạm vi đăng ký thuế bao gồm những nội dung cụ thể nào?", [("105/2020/TT-BTC", "1")]),
    ("Enterprise tax code", "Mã số thuế của doanh nghiệp được quy định là mã số nào?", [("59/2020/QH14", "29")]),
    ("Tax filing extension", "Trường hợp bất khả kháng được gia hạn nộp hồ sơ khai thuế tối đa bao nhiêu ngày?", [("38/2019/QH14", "46")]),
    ("Tax overpayment", "Tiền thuế và tiền phạt nộp thừa được xử lý bằng những cách nào?", [("38/2019/QH14", "60")]),
    ("Tax exemption reduction", "Người nộp thuế thuộc trường hợp miễn giảm thuế phải làm gì để được hưởng?", [("38/2019/QH14", "79")]),
    ("Electronic tax document", "Chứng từ điện tử trong quản lý thuế được định nghĩa là gì?", [("38/2019/QH14", "94")]),
    ("E-invoice types", "Có những loại hóa đơn điện tử nào?", [("123/2020/NĐ-CP", "8")]),
    ("Tax assessment", "Trường hợp nào cơ quan thuế ấn định thuế đối với người nộp thuế?", [("38/2019/QH14", "50")]),
    ("Tax enforcement", "Cơ quan thuế có thể áp dụng những biện pháp cưỡng chế nào khi người nộp thuế nợ thuế?", [("38/2019/QH14", "125")]),
    ("Tax debt deletion bankruptcy", "Doanh nghiệp bị tuyên bố phá sản được xóa nợ tiền thuế trong trường hợp nào?", [("38/2019/QH14", "85")]),
    ("Ecommerce platform VAT", "Tổ chức quản lý nền tảng thương mại điện tử có chức năng thanh toán khấu trừ nộp thuế thay cho hộ kinh doanh theo quy định nào?", [("48/2024/QH15", "4")]),
    ("IP third party opinion", "Người thứ ba gửi ý kiến phản đối việc cấp văn bằng bảo hộ sở hữu công nghiệp theo quy định nào?", [("50/2005/QH11", "112")]),
    ("IP grant certificate", "Sau thẩm định nội dung cần làm gì để được cấp văn bằng bảo hộ sở hữu công nghiệp?", [("50/2005/QH11", "118")]),
    ("Trademark infringement", "Dùng dấu hiệu trùng với nhãn hiệu được bảo hộ cho hàng hóa dịch vụ trùng có bị coi là xâm phạm không?", [("50/2005/QH11", "129")]),
    ("IP assignment content", "Hợp đồng chuyển nhượng quyền sở hữu công nghiệp phải có nội dung chính nào?", [("50/2005/QH11", "140")]),
    ("IP license content", "Hợp đồng sử dụng đối tượng sở hữu công nghiệp phải có nội dung chủ yếu nào?", [("50/2005/QH11", "144")]),
    ("IP assignment effect", "Hợp đồng chuyển nhượng quyền sở hữu công nghiệp có hiệu lực khi nào?", [("50/2005/QH11", "148")]),
    ("Commercial penalty agreement", "Có được yêu cầu phạt vi phạm hợp đồng thương mại nếu hai bên không thỏa thuận không?", [("36/2005/QH11", "300")]),
    ("Commercial penalty cap", "Mức phạt vi phạm tối đa trong hợp đồng thương mại là bao nhiêu?", [("36/2005/QH11", "301")]),
    ("Commercial damages", "Bồi thường thiệt hại trong hợp đồng thương mại bao gồm những khoản nào?", [("36/2005/QH11", "302")]),
    ("Promotion notice", "Công ty tổ chức chương trình khuyến mại phải thông báo với cơ quan quản lý nhà nước như thế nào?", [("81/2018/NĐ-CP", "17")]),
    ("Promotion value cap", "Giá trị hàng hóa dịch vụ dùng để khuyến mại có được vượt quá 50% không?", [("81/2018/NĐ-CP", "6")]),
    ("False advertising", "Quảng cáo sai sự thật gây nhầm lẫn về khả năng kinh doanh bị cấm theo quy định nào?", [("16/2012/QH13", "8")]),
    ("Foreign ad representative office", "Văn phòng đại diện doanh nghiệp quảng cáo nước ngoài có được trực tiếp kinh doanh quảng cáo không?", [("16/2012/QH13", "41")]),
    ("Address change", "Doanh nghiệp thay đổi địa chỉ trụ sở chính sang tỉnh khác cần thủ tục đăng ký thay đổi nào?", [("168/2025/NĐ-CP", "40")]),
    ("Representative change", "Hồ sơ đăng ký thay đổi người đại diện theo pháp luật của công ty gồm giấy tờ nào?", [("168/2025/NĐ-CP", "43")]),
    ("Business suspension", "Doanh nghiệp tạm ngừng kinh doanh phải gửi thông báo trước bao nhiêu ngày làm việc?", [("168/2025/NĐ-CP", "60")]),
    ("Household suspension", "Hộ kinh doanh tạm ngừng kinh doanh từ 15 ngày trở lên phải gửi hồ sơ trước bao nhiêu ngày?", [("168/2025/NĐ-CP", "103")]),
    ("Social insurance rate", "Mức đóng bảo hiểm xã hội hằng tháng vào quỹ hưu trí và tử tuất của người lao động là bao nhiêu?", [("41/2024/QH15", "33")]),
]


def article_key(law_id: str, article_id: str) -> str:
    return f"{canonical_law_id(law_id)}|điều {str(article_id).lower()}"


def doc_key(law_id: str) -> str:
    return canonical_law_id(law_id)


def f2_for_sets(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    hits = len(pred & gold)
    precision = hits / len(pred) if pred else 0.0
    recall = hits / len(gold) if gold else 0.0
    f2 = (5 * precision * recall / (4 * precision + recall)) if precision and recall else 0.0
    return precision, recall, f2


def average_metrics(rows: Sequence[tuple[set[str], set[str], set[str], set[str]]]) -> dict[str, float]:
    article_metrics = [f2_for_sets(pred_articles, gold_articles) for pred_articles, gold_articles, _pd, _gd in rows]
    doc_metrics = [f2_for_sets(pred_docs, gold_docs) for _pa, _ga, pred_docs, gold_docs in rows]

    def avg(index: int, values: Sequence[tuple[float, float, float]]) -> float:
        return sum(row[index] for row in values) / len(values) if values else 0.0

    return {
        "count": len(rows),
        "ARTICLES_PRECISION": avg(0, article_metrics),
        "ARTICLES_RECALL": avg(1, article_metrics),
        "ARTICLES_F2MACRO": avg(2, article_metrics),
        "DOCS_PRECISION": avg(0, doc_metrics),
        "DOCS_RECALL": avg(1, doc_metrics),
        "DOCS_F2MACRO": avg(2, doc_metrics),
    }


def predict_questions(
    questions: Iterable[dict[str, Any]],
    rerank_min_delta: float,
    use_db_heuristics: bool,
    include_db_heuristic_codes: bool,
) -> list[dict[str, Any]]:
    articles = load_augmented_articles(include_db_heuristic_codes=include_db_heuristic_codes)
    lookup = build_article_lookup(articles)
    bm25, _ = build_or_load_index(articles)
    output = []
    for item in questions:
        selected = rank_article(
            item["question"],
            articles,
            bm25,
            rerank_min_delta=rerank_min_delta,
            use_db_heuristics=use_db_heuristics,
        )
        selected = resolve_article(selected, lookup)
        output.append(selected)
    return output


def train_core_rows() -> list[dict[str, Any]]:
    articles = load_augmented_articles()
    core_laws = {canonical_law_id(article["law_id"]) for article in articles}
    output = []
    with TRAIN_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold_articles = ast.literal_eval(row["relevant_articles"])
            gold_laws = {canonical_law_id(item["law_id"]) for item in gold_articles}
            if not gold_laws & core_laws:
                continue
            output.append(
                {
                    "id": row["question_id"],
                    "question": row["question"],
                    "gold": [(item["law_id"], item["article_id"]) for item in gold_articles],
                }
            )
    return output


def manual_rows() -> list[dict[str, Any]]:
    return [
        {"id": f"manual_{i + 1}", "name": name, "question": question, "gold": gold}
        for i, (name, question, gold) in enumerate(MANUAL_CASES)
    ]


def evaluate_cases(
    cases: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    sample_failures: int = 20,
) -> dict[str, Any]:
    metric_rows = []
    failures = []
    for case, pred in zip(cases, predictions):
        pred_articles = {article_key(pred.get("law_id", ""), pred.get("article_id", ""))}
        pred_docs = {doc_key(pred.get("law_id", ""))}
        gold_articles = {article_key(law_id, article_id) for law_id, article_id in case["gold"]}
        gold_docs = {doc_key(law_id) for law_id, _article_id in case["gold"]}
        metric_rows.append((pred_articles, gold_articles, pred_docs, gold_docs))
        if not pred_articles & gold_articles and len(failures) < sample_failures:
            failures.append(
                {
                    "id": case["id"],
                    "name": case.get("name", ""),
                    "question": case["question"],
                    "pred": sorted(pred_articles),
                    "gold": sorted(gold_articles),
                    "source": pred.get("source", ""),
                }
            )
    result = average_metrics(metric_rows)
    result["failures"] = failures
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--rerank-min-delta", type=float, default=0.20)
    parser.add_argument("--use-db-heuristics", action="store_true")
    parser.add_argument("--include-db-heuristic-codes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manual = manual_rows()
    train = train_core_rows()
    manual_predictions = predict_questions(
        manual,
        rerank_min_delta=args.rerank_min_delta,
        use_db_heuristics=args.use_db_heuristics,
        include_db_heuristic_codes=args.include_db_heuristic_codes,
    )
    train_predictions = predict_questions(
        train,
        rerank_min_delta=args.rerank_min_delta,
        use_db_heuristics=args.use_db_heuristics,
        include_db_heuristic_codes=args.include_db_heuristic_codes,
    )
    result = {
        "note": "Independent benchmark: labeled train_core plus manually curated R2AI-like legal scenarios.",
        "settings": {
            "rerank_min_delta": args.rerank_min_delta,
            "use_db_heuristics": args.use_db_heuristics,
            "include_db_heuristic_codes": args.include_db_heuristic_codes,
        },
        "manual_r2ai_like": evaluate_cases(manual, manual_predictions),
        "train_core": evaluate_cases(train, train_predictions),
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
