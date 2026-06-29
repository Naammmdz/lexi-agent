"""Create an agent-style legal retrieval workflow submission.

The goal of this layer is not to guess individual rows.  It applies small,
audited workflow rules that mirror how the final legal assistant should work:
classify legal intent, attach a legally necessary foundation/companion document,
and keep the existing high-precision retrieval as the anchor.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable

from _paths import REPO_ROOT
from create_augmented_submission import normalize_text
from create_domain_repair_submission import article_key, article_ref, doc_key, doc_ref, update_answer
from utils.submission_formatter import canonical_law_id, load_law_title_mapping


MAPPING_PATH = REPO_ROOT / "data" / "law_id_to_title.json"
DEFAULT_BASE = REPO_ROOT / "submission_variants" / "submission_domain_rerank_guarded_s240.zip"
DEFAULT_READY = REPO_ROOT / "submission.zip"
DEFAULT_READY_VARIANT = REPO_ROOT / "submission_variants" / "submission.zip"


FRAMEWORK_COMPANION_RULES: dict[str, dict[str, Any]] = {
    "04/2017/QH14": {
        "framework_articles": ("4", "5"),
        "framework_or_preamble": {"1", "2", "3", "4", "5"},
    },
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("results.json"))


def has_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def current_doc_keys(row: dict[str, Any]) -> set[str]:
    return {key for ref in row.get("relevant_docs", []) if (key := doc_key(ref))}


def current_article_keys(row: dict[str, Any]) -> set[tuple[str, str]]:
    return {key for ref in row.get("relevant_articles", []) if (key := article_key(ref))}


def append_article(row: dict[str, Any], mapping: dict[str, str], law_id: str, article_id: str) -> bool:
    law_id = canonical_law_id(law_id)
    doc_keys = current_doc_keys(row)
    article_keys = current_article_keys(row)
    article_id = str(article_id)
    if (law_id, article_id.lower()) in article_keys:
        return False
    if law_id not in doc_keys:
        row.setdefault("relevant_docs", []).append(doc_ref(law_id, mapping))
    row.setdefault("relevant_articles", []).append(article_ref(law_id, article_id, mapping))
    return True


def append_doc(row: dict[str, Any], mapping: dict[str, str], law_id: str) -> bool:
    law_id = canonical_law_id(law_id)
    if law_id in current_doc_keys(row):
        return False
    row.setdefault("relevant_docs", []).append(doc_ref(law_id, mapping))
    return True


def social_insurance_late_payment_rule(row: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
    """Attach the foundation Social Insurance Law to late-payment penalty rows.

    Existing retrieval often finds Decree 12/2022 Article 39, which is the
    administrative penalty.  A legal assistant still needs the underlying duty
    and handling rule in the Social Insurance Law when the query asks about
    late compulsory social-insurance payment.
    """
    q = normalize_text(row.get("question", ""))
    if not (
        "chậm đóng" in q
        and has_any(q, ("bảo hiểm xã hội", "bhxh", "bảo hiểm thất nghiệp"))
        and has_any(q, ("xử phạt", "xử lý", "bị phạt"))
    ):
        return None
    keys = current_article_keys(row)
    if ("12/2022/NĐ-CP", "39") not in keys:
        return None
    return "social_insurance_late_payment_foundation", [("41/2024/QH15", "40")]


def labor_penalty_foundation_rule(row: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
    """Attach foundation labor/OSH articles for high-confidence penalty rows."""
    q = normalize_text(row.get("question", ""))
    keys = current_article_keys(row)
    refs: list[tuple[str, str]] = []

    if ("12/2022/NĐ-CP", "9") in keys and has_any(q, ("giữ bản chính", "văn bằng", "chứng chỉ", "bằng cấp")):
        refs.append(("45/2019/QH14", "17"))

    if ("12/2022/NĐ-CP", "10") in keys and has_any(q, ("thử việc", "lương thử việc", "85%")):
        refs.append(("45/2019/QH14", "26"))

    if ("12/2022/NĐ-CP", "12") in keys and has_any(q, ("trả sổ bảo hiểm xã hội", "chấm dứt hợp đồng", "thôi việc")):
        refs.append(("45/2019/QH14", "48"))

    if ("12/2022/NĐ-CP", "13") in keys and has_any(q, ("thuê lại lao động", "lao động thuê lại")):
        refs.append(("45/2019/QH14", "57"))

    if ("12/2022/NĐ-CP", "15") in keys and "đối thoại" in q:
        refs.append(("45/2019/QH14", "63"))

    if ("12/2022/NĐ-CP", "19") in keys and has_any(q, ("phạt tiền", "cắt lương", "trừ lương", "kỷ luật")):
        refs.append(("45/2019/QH14", "127"))

    if ("12/2022/NĐ-CP", "20") in keys and has_any(q, ("báo cáo", "thống kê")):
        refs.append(("84/2015/QH13", "36"))

    if ("12/2022/NĐ-CP", "21") in keys and has_any(q, ("hồ sơ vệ sinh môi trường", "yếu tố có hại", "yếu tố nguy hiểm")):
        refs.append(("84/2015/QH13", "18"))

    if ("12/2022/NĐ-CP", "22") in keys and has_any(q, ("khám sức khỏe", "khám sức khoẻ", "bệnh nghề nghiệp")):
        refs.append(("84/2015/QH13", "21"))

    if ("12/2022/NĐ-CP", "22") in keys and has_any(q, ("phương tiện bảo vệ cá nhân", "trang cấp", "an toàn, vệ sinh lao động")):
        refs.append(("84/2015/QH13", "16"))

    if ("12/2022/NĐ-CP", "23") in keys and has_any(q, ("tai nạn lao động", "bệnh nghề nghiệp", "sơ cứu", "cấp cứu", "bồi thường")):
        refs.append(("84/2015/QH13", "38"))

    if ("12/2022/NĐ-CP", "24") in keys and has_any(q, ("máy móc", "thiết bị", "yêu cầu nghiêm ngặt", "khai báo")):
        refs.append(("84/2015/QH13", "30"))

    if ("12/2022/NĐ-CP", "25") in keys and has_any(q, ("thẻ an toàn", "huấn luyện", "yêu cầu nghiêm ngặt")):
        refs.append(("84/2015/QH13", "14"))

    if ("12/2022/NĐ-CP", "27") in keys and has_any(q, ("quan trắc môi trường lao động", "kiểm soát tác hại", "yếu tố có hại")):
        refs.append(("84/2015/QH13", "18"))

    if ("12/2022/NĐ-CP", "28") in keys and has_any(q, ("lao động nữ", "hành kinh", "nuôi con dưới 12 tháng", "mang thai")):
        refs.append(("45/2019/QH14", "137"))

    if ("12/2022/NĐ-CP", "29") in keys and has_any(q, ("lao động chưa thành niên", "chưa thành niên", "người chưa đủ 15 tuổi")):
        refs.append(("45/2019/QH14", "144"))

    if ("12/2022/NĐ-CP", "31") in keys and has_any(q, ("người lao động khuyết tật", "khuyết tật nặng", "khuyết tật")):
        refs.append(("45/2019/QH14", "160"))

    if ("12/2022/NĐ-CP", "34") in keys and has_any(q, ("đình công", "sa thải")):
        refs.append(("45/2019/QH14", "208"))

    if ("12/2022/NĐ-CP", "35") in keys and has_any(q, ("công đoàn", "tổ chức đại diện người lao động")):
        if has_any(q, ("cản trở", "không cho", "đơn phương chấm dứt", "gây bất lợi", "tham gia hoạt động công đoàn")):
            refs.append(("12/2012/QH13", "9"))
        if has_any(q, ("cán bộ công đoàn cấp trên", "công đoàn cấp trên", "tuyên truyền", "hướng dẫn", "thành lập công đoàn")):
            refs.append(("12/2012/QH13", "16"))
        if has_any(q, ("đơn phương chấm dứt", "phân biệt đối xử", "gây bất lợi")):
            refs.append(("45/2019/QH14", "175"))

    if ("12/2022/NĐ-CP", "38") in keys and "kinh phí công đoàn" in q:
        refs.append(("12/2012/QH13", "26"))
        if has_any(q, ("mức", "bao nhiêu", "chậm đóng", "đóng kinh phí", "phải đóng")):
            refs.append(("191/2013/NĐ-CP", "5"))

    if ("12/2022/NĐ-CP", "39") in keys and "chậm đóng" in q and has_any(q, ("bảo hiểm xã hội", "bhxh", "bảo hiểm thất nghiệp")):
        refs.append(("41/2024/QH15", "40"))

    if ("12/2022/NĐ-CP", "39") in keys and "trốn đóng" in q and has_any(q, ("bảo hiểm xã hội", "bhxh", "bảo hiểm thất nghiệp")):
        refs.append(("41/2024/QH15", "41"))

    if ("12/2022/NĐ-CP", "40") in keys and has_any(q, ("làm giả hồ sơ", "giả mạo hồ sơ", "sai lệch hồ sơ", "trục lợi")):
        refs.append(("41/2024/QH15", "9"))

    if ("12/2022/NĐ-CP", "16") in keys and has_any(q, ("thương lượng tập thể", "thỏa ước lao động tập thể", "thoả ước lao động tập thể")):
        if has_any(q, ("quy trình", "cung cấp thông tin", "bố trí địa điểm", "từ chối thương lượng", "tổ chức họp", "yêu cầu thương lượng")):
            refs.append(("45/2019/QH14", "70"))
        if has_any(q, ("sáp nhập", "hợp nhất", "chia", "tách", "chuyển đổi loại hình")):
            refs.append(("45/2019/QH14", "80"))
        if has_any(q, ("mở rộng phạm vi", "mở rộng áp dụng")):
            refs.append(("45/2019/QH14", "84"))
        if has_any(q, ("gia nhập", "rút khỏi")):
            refs.append(("45/2019/QH14", "85"))
        if has_any(q, ("vô hiệu", "bị tuyên bố vô hiệu")):
            refs.append(("45/2019/QH14", "86"))

    if not refs:
        return None
    clean_refs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for law_id, article_id in refs:
        key = (canonical_law_id(law_id), str(article_id).lower())
        if key in keys or key in seen:
            continue
        clean_refs.append((law_id, article_id))
        seen.add(key)

    if not clean_refs:
        return None
    return "labor_penalty_foundation", clean_refs[:2]


def tax_foundation_companion_rule(row: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
    """Attach law/decree companions for high-confidence tax penalty/assessment rows."""
    q = normalize_text(row.get("question", ""))
    keys = current_article_keys(row)
    refs: list[tuple[str, str]] = []

    if ("125/2020/NĐ-CP", "16") in keys and has_any(q, ("khai sai", "thiếu số tiền thuế", "thiếu tiền thuế", "tăng số tiền thuế được miễn")):
        refs.append(("38/2019/QH14", "142"))

    if ("125/2020/NĐ-CP", "17") in keys and "trốn thuế" in q:
        refs.append(("38/2019/QH14", "143"))

    if ("38/2019/QH14", "50") in keys and "ấn định thuế" in q:
        refs.append(("126/2020/NĐ-CP", "14"))

    if not refs:
        return None
    return "tax_foundation_companion", refs[:2]


def tax_registration_foundation_rule(row: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
    """Attach precise Tax Administration Law registration articles for tax-registration guidance."""
    q = normalize_text(row.get("question", ""))
    docs = current_doc_keys(row)
    keys = current_article_keys(row)
    refs: list[tuple[str, str]] = []

    if any(law_id == "38/2019/QH14" for law_id, _article_id in keys):
        return None
    if "105/2020/TT-BTC" not in docs and not has_any(q, ("đăng ký thuế", "mã số thuế", "hiệu lực mã số thuế")):
        return None

    if has_any(q, ("phạm vi đăng ký thuế", "đối tượng đăng ký thuế", "cấp mã số thuế")):
        refs.append(("38/2019/QH14", "30"))
    if has_any(q, ("hồ sơ đăng ký thuế lần đầu", "đăng ký thuế lần đầu")):
        refs.append(("38/2019/QH14", "31"))
    if has_any(q, ("địa điểm nộp hồ sơ", "nộp hồ sơ đăng ký thuế")):
        refs.append(("38/2019/QH14", "32"))
    if has_any(q, ("thời hạn đăng ký thuế", "bao lâu phải đăng ký thuế")):
        refs.append(("38/2019/QH14", "33"))
    if has_any(q, ("giấy chứng nhận đăng ký thuế", "cấp giấy chứng nhận")):
        refs.append(("38/2019/QH14", "34"))
    if has_any(q, ("sử dụng mã số thuế", "mã số thuế của doanh nghiệp", "cho người khác sử dụng mã số thuế")):
        refs.append(("38/2019/QH14", "35"))
    if has_any(q, ("thay đổi thông tin đăng ký thuế", "thay đổi thông tin")):
        refs.append(("38/2019/QH14", "36"))
    if has_any(q, ("tạm ngừng hoạt động", "tạm ngừng kinh doanh")) and "mã số thuế" in q:
        refs.append(("38/2019/QH14", "37"))
    if has_any(q, ("tổ chức lại doanh nghiệp", "chia", "tách", "hợp nhất", "sáp nhập")) and "đăng ký thuế" in q:
        refs.append(("38/2019/QH14", "38"))
    if has_any(q, ("chấm dứt hiệu lực mã số thuế", "ngưng sử dụng mã số thuế")):
        refs.append(("38/2019/QH14", "39"))
    if has_any(q, ("khôi phục mã số thuế", "khôi phục tình trạng pháp lý")) and "mã số thuế" in q:
        refs.append(("38/2019/QH14", "40"))

    clean_refs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for law_id, article_id in refs:
        key = (canonical_law_id(law_id), str(article_id).lower())
        if key in keys or key in seen:
            continue
        clean_refs.append((law_id, article_id))
        seen.add(key)

    if not clean_refs:
        return None
    return "tax_registration_foundation", clean_refs[:2]


def tax_doc_article_fill_rule(row: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
    """Fill precise Tax Administration Law articles when the law doc is already present."""
    q = normalize_text(row.get("question", ""))
    docs = current_doc_keys(row)
    keys = current_article_keys(row)
    refs: list[tuple[str, str]] = []

    if any(law_id == "38/2019/QH14" for law_id, _article_id in keys):
        return None
    if not ({"38/2019/QH14", "125/2020/NĐ-CP", "126/2020/NĐ-CP"} & docs):
        return None

    if has_any(q, ("thẩm quyền ra quyết định xử phạt", "thẩm quyền xử phạt")):
        refs.append(("38/2019/QH14", "139"))

    if has_any(q, ("miễn tiền phạt", "được miễn tiền phạt")):
        refs.append(("38/2019/QH14", "140"))

    if has_any(q, ("thời hiệu xử phạt", "thời hiệu thi hành quyết định xử phạt")):
        refs.append(("38/2019/QH14", "137"))

    if has_any(q, ("chậm nộp tiền phạt", "tiền chậm nộp")) and "tiền phạt" in q:
        refs.append(("38/2019/QH14", "138"))

    if has_any(q, ("biên bản vi phạm hành chính", "vi phạm hành chính điện tử")):
        refs.append(("38/2019/QH14", "136"))

    if has_any(q, ("gia hạn nộp thuế", "thời gian được gia hạn")):
        refs.append(("38/2019/QH14", "63" if "khó khăn đặc biệt" in q else "62"))

    if has_any(q, ("xóa nợ thuế", "xoá nợ thuế", "xóa nợ tiền thuế", "xoá nợ tiền thuế")):
        refs.append(("38/2019/QH14", "85"))

    if has_any(q, ("cưỡng chế", "thu hồi giấy chứng nhận đăng ký doanh nghiệp", "ngừng sử dụng hóa đơn", "ngừng sử dụng hoá đơn")):
        if has_any(q, ("thu hồi giấy chứng nhận đăng ký doanh nghiệp", "thu hồi giấy chứng nhận đăng ký kinh doanh")):
            refs.append(("38/2019/QH14", "135"))
        elif has_any(q, ("ngừng sử dụng hóa đơn", "ngừng sử dụng hoá đơn")):
            refs.append(("38/2019/QH14", "132"))
        else:
            refs.append(("38/2019/QH14", "125"))

    if has_any(q, ("hoàn thuế", "hoàn thuế gtgt", "hoàn thuế giá trị gia tăng")):
        refs.append(("38/2019/QH14", "71"))
        if has_any(q, ("tiếp nhận hồ sơ", "hồ sơ điện tử", "quy trình tiếp nhận")):
            refs.append(("38/2019/QH14", "72"))
        if has_any(q, ("hoàn thuế trước", "kiểm tra trước hoàn thuế", "phân loại")):
            refs.append(("38/2019/QH14", "73"))
        if "thẩm quyền" in q or "cơ quan nào" in q:
            refs.append(("38/2019/QH14", "76"))

    if has_any(q, ("chậm nộp hồ sơ khai thuế", "hồ sơ khai thuế")):
        refs.append(("38/2019/QH14", "141"))

    if has_any(q, ("hóa đơn", "hoá đơn")) and has_any(q, ("xử phạt", "vi phạm", "cho mượn", "không lập", "sử dụng hóa đơn")):
        refs.append(("38/2019/QH14", "146"))

    clean_refs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for law_id, article_id in refs:
        key = (canonical_law_id(law_id), str(article_id).lower())
        if key in keys or key in seen:
            continue
        clean_refs.append((law_id, article_id))
        seen.add(key)

    if not clean_refs:
        return None
    return "tax_doc_article_fill", clean_refs[:2]


def tax_invoice_foundation_rule(row: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
    """Attach precise Tax Administration Law foundations for invoice/e-document rows."""
    q = normalize_text(row.get("question", ""))
    docs = current_doc_keys(row)
    keys = current_article_keys(row)
    refs: list[tuple[str, str]] = []

    if "123/2020/NĐ-CP" not in docs:
        return None
    if not has_any(q, ("hóa đơn", "hoá đơn", "chứng từ", "biên lai", "máy tính tiền")):
        return None

    if has_any(q, ("chứng từ", "biên lai", "khấu trừ")):
        refs.append(("38/2019/QH14", "94"))

    if has_any(q, ("cơ sở dữ liệu", "cung cấp thông tin", "tra cứu", "cổng thông tin")):
        refs.append(("38/2019/QH14", "93"))

    if has_any(q, ("đăng ký sử dụng", "có mã", "không có mã", "máy tính tiền", "ngừng sử dụng")):
        refs.append(("38/2019/QH14", "91"))

    if has_any(q, ("lập hóa đơn", "lập hoá đơn", "nội dung", "định dạng", "sai", "xử lý", "hủy", "huỷ", "tiêu hủy", "tiêu huỷ", "loại hóa đơn", "loại hoá đơn")):
        refs.append(("38/2019/QH14", "90"))

    if not refs and has_any(q, ("hóa đơn điện tử", "hoá đơn điện tử")):
        refs.append(("38/2019/QH14", "89"))

    clean_refs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for law_id, article_id in refs:
        key = (canonical_law_id(law_id), str(article_id).lower())
        if key in keys or key in seen:
            continue
        clean_refs.append((law_id, article_id))
        seen.add(key)

    if not clean_refs:
        return None
    return "tax_invoice_foundation", clean_refs[:2]


def ip_infringement_companion_rule(row: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
    """Attach Decree 65 infringement-element articles for IP infringement rows."""
    q = normalize_text(row.get("question", ""))
    keys = current_article_keys(row)
    refs: list[tuple[str, str]] = []

    if ("50/2005/QH11", "126") in keys and has_any(q, ("kiểu dáng công nghiệp", "xâm phạm")):
        refs.append(("65/2023/NĐ-CP", "76"))

    if ("50/2005/QH11", "129") in keys and has_any(q, ("nhãn hiệu", "xâm phạm")):
        refs.append(("65/2023/NĐ-CP", "77"))

    if not refs:
        return None
    return "ip_infringement_companion", refs[:2]


def ip_foundation_law_rule(row: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
    """Attach precise IP Law articles when Decree 65 procedure articles are the anchor."""
    q = normalize_text(row.get("question", ""))
    docs = current_doc_keys(row)
    keys = current_article_keys(row)
    refs: list[tuple[str, str]] = []

    if "65/2023/NĐ-CP" not in docs:
        return None
    if not has_any(
        q,
        (
            "sở hữu công nghiệp",
            "sở hữu trí tuệ",
            "quyền sở hữu trí tuệ",
            "nhãn hiệu",
            "sáng chế",
            "kiểu dáng",
            "chỉ dẫn địa lý",
            "văn bằng bảo hộ",
            "bằng độc quyền",
            "cục sở hữu trí tuệ",
            "công báo sở hữu công nghiệp",
            "đơn pct",
            "đơn madrid",
            "đơn la hay",
            "tên thương mại",
            "xâm phạm quyền",
            "xâm phạm sở hữu",
            "tạm dừng thông quan",
        ),
    ):
        return None
    if any(law_id == "50/2005/QH11" for law_id, _article_id in keys):
        return None

    if has_any(q, ("phản đối", "ý kiến phản đối", "ý kiến của người thứ ba", "người thứ ba")):
        refs.append(("50/2005/QH11", "112"))

    if has_any(q, ("công bố", "công báo", "quyết định cấp văn bằng", "dự định cấp văn bằng")):
        refs.append(("50/2005/QH11", "99"))

    if has_any(q, ("tách đơn", "rút đơn", "sửa đổi", "bổ sung đơn", "chuyển đổi đơn")):
        refs.append(("50/2005/QH11", "115"))

    if has_any(q, ("thẩm định hình thức", "tính hợp lệ của đơn")):
        refs.append(("50/2005/QH11", "109"))

    if has_any(q, ("thẩm định nội dung", "điều kiện bảo hộ", "khả năng cấp văn bằng")):
        refs.append(("50/2005/QH11", "114"))

    if has_any(q, ("duy trì hiệu lực", "gia hạn hiệu lực", "gia hạn giấy chứng nhận", "gia hạn cho giấy chứng nhận")):
        refs.append(("50/2005/QH11", "94"))

    if has_any(q, ("chấm dứt hiệu lực", "chấm dứt trước thời hạn")):
        refs.append(("50/2005/QH11", "95"))

    if has_any(q, ("hủy bỏ hiệu lực", "huỷ bỏ hiệu lực")):
        refs.append(("50/2005/QH11", "96"))

    if not refs and has_any(q, ("hồ sơ đăng ký", "ngày nộp đơn", "chấp nhận ngày nộp đơn", "tài liệu trong đơn", "đơn đăng ký")):
        if "nhãn hiệu" in q:
            refs.append(("50/2005/QH11", "105"))
        elif "kiểu dáng" in q:
            refs.append(("50/2005/QH11", "103"))
        elif "sáng chế" in q:
            refs.append(("50/2005/QH11", "102"))
        elif "chỉ dẫn địa lý" in q:
            refs.append(("50/2005/QH11", "106"))
        else:
            refs.append(("50/2005/QH11", "100"))

    if has_any(q, ("chuyển nhượng quyền sở hữu công nghiệp", "hợp đồng chuyển nhượng")):
        refs.append(("50/2005/QH11", "140"))

    if has_any(q, ("chuyển quyền sử dụng", "cho đối tác thuê quyền sử dụng", "hợp đồng sử dụng", "sửa đổi nội dung hợp đồng", "gia hạn hợp đồng")):
        refs.append(("50/2005/QH11", "144"))

    if has_any(q, ("hiệu lực hợp đồng", "đăng ký hợp đồng", "hợp đồng chuyển giao")):
        refs.append(("50/2005/QH11", "148"))

    if has_any(q, ("ngăn cấm đối thủ", "quyền ngăn cấm", "không có quyền cấm")):
        refs.append(("50/2005/QH11", "125"))

    if has_any(q, ("hàng hóa bị nghi ngờ xâm phạm", "tạm dừng thông quan", "kiểm soát hàng hóa", "cơ quan hải quan", "hải quan")):
        refs.append(("50/2005/QH11", "216"))
        if has_any(q, ("nghĩa vụ bảo đảm", "bảo đảm tài chính", "chuẩn bị giấy tờ", "người yêu cầu")):
            refs.append(("50/2005/QH11", "217"))

    clean_refs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for law_id, article_id in refs:
        key = (canonical_law_id(law_id), str(article_id).lower())
        if key in keys or key in seen:
            continue
        clean_refs.append((law_id, article_id))
        seen.add(key)

    if not clean_refs:
        return None
    return "ip_foundation_law", clean_refs[:2]


def ip_doc_article_fill_rule(row: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
    """Fill precise IP Law articles when the IP Law doc is already present."""
    q = normalize_text(row.get("question", ""))
    docs = current_doc_keys(row)
    keys = current_article_keys(row)
    refs: list[tuple[str, str]] = []

    if any(law_id == "50/2005/QH11" for law_id, _article_id in keys):
        return None
    if not ({"50/2005/QH11", "65/2023/NĐ-CP"} & docs):
        return None

    if has_any(q, ("chủ sở hữu đối tượng sở hữu công nghiệp", "chủ sở hữu sáng chế", "chủ sở hữu nhãn hiệu")):
        refs.append(("50/2005/QH11", "121"))

    if "tính mới của kiểu dáng công nghiệp" in q:
        refs.append(("50/2005/QH11", "65"))

    if has_any(q, ("điều kiện chung đối với kiểu dáng", "đăng ký kiểu dáng", "đặc điểm tạo dáng")):
        refs.append(("50/2005/QH11", "63"))

    if has_any(q, ("giải pháp kỹ thuật", "đăng ký sáng chế")):
        refs.append(("50/2005/QH11", "58"))

    if has_any(q, ("bản dịch tiếng việt", "tài liệu nước ngoài", "hồ sơ xác lập quyền", "đơn đăng ký")):
        refs.append(("50/2005/QH11", "100"))

    if has_any(q, ("đơn la hay", "khôi phục thời hạn", "tách đơn", "thống nhất")):
        refs.append(("50/2005/QH11", "115"))

    if has_any(q, ("đại diện sở hữu công nghiệp", "kiểm tra nghiệp vụ", "chứng chỉ hành nghề")):
        if has_any(q, ("điều kiện", "hành nghề", "kiểm tra nghiệp vụ")):
            refs.append(("50/2005/QH11", "155"))
        else:
            refs.append(("50/2005/QH11", "151"))

    if has_any(q, ("quyền đăng ký", "tài liệu chứng minh quyền đăng ký")):
        if "nhãn hiệu" in q:
            refs.append(("50/2005/QH11", "87"))
        elif has_any(q, ("sáng chế", "kiểu dáng", "thiết kế bố trí")):
            refs.append(("50/2005/QH11", "86"))

    if has_any(q, ("xâm phạm quyền sở hữu công nghiệp", "hàng giả mạo sở hữu trí tuệ", "bị kết luận là xâm phạm", "đặt tên doanh nghiệp", "trùng với nhãn hiệu")):
        refs.append(("50/2005/QH11", "211"))
        if "nhãn hiệu" in q or "tên thương mại" in q or "tên doanh nghiệp" in q:
            refs.append(("50/2005/QH11", "129"))
        if "bí mật kinh doanh" in q:
            refs.append(("50/2005/QH11", "127"))

    if has_any(q, ("giám định sở hữu trí tuệ", "kết quả giám định", "hợp đồng giám định")):
        refs.append(("50/2005/QH11", "201"))

    if has_any(q, ("ngăn chặn đối thủ", "cấp văn bằng bảo hộ", "hủy bỏ văn bằng", "huỷ bỏ văn bằng")):
        refs.append(("50/2005/QH11", "112"))
        refs.append(("50/2005/QH11", "96"))

    if has_any(q, ("nhượng quyền thương mại", "nhãn hiệu", "chuyển quyền sử dụng")):
        refs.append(("50/2005/QH11", "141"))

    clean_refs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for law_id, article_id in refs:
        key = (canonical_law_id(law_id), str(article_id).lower())
        if key in keys or key in seen:
            continue
        clean_refs.append((law_id, article_id))
        seen.add(key)

    if not clean_refs:
        return None
    return "ip_doc_article_fill", clean_refs[:2]


def sme_procurement_companion_rule(row: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
    """Attach SME-law procurement preference when retrieval only finds procurement law."""
    q = normalize_text(row.get("question", ""))
    keys = current_article_keys(row)
    if not (has_any(q, ("doanh nghiệp nhỏ và vừa", "doanh nghiệp siêu nhỏ", "doanh nghiệp nhỏ")) and "đấu thầu" in q):
        return None
    if ("04/2017/QH14", "13") in keys:
        return None
    if not (("43/2013/QH13", "14") in keys or "63/2014/NĐ-CP" in current_doc_keys(row)):
        return None
    return "sme_procurement_companion", [("04/2017/QH14", "13")]


def sme_foundation_law_rule(row: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
    """Attach SME Support Law foundations when Decree 80 is the detailed anchor."""
    q = normalize_text(row.get("question", ""))
    docs = current_doc_keys(row)
    keys = current_article_keys(row)
    refs: list[tuple[str, str]] = []

    if "80/2021/NĐ-CP" not in docs:
        return None
    if not has_any(q, ("doanh nghiệp nhỏ và vừa", "doanh nghiệp siêu nhỏ", "doanh nghiệp nhỏ", "doanh nghiệp vừa", "nhỏ và vừa")):
        return None

    if has_any(q, ("tiêu chí", "xác định", "quy mô", "siêu nhỏ", "doanh nghiệp nhỏ", "doanh nghiệp vừa")):
        refs.append(("04/2017/QH14", "4"))

    if "hỗ trợ" in q:
        refs.append(("04/2017/QH14", "5"))

    if has_any(q, ("công nghệ", "chuyển đổi số", "ươm tạo", "cơ sở kỹ thuật", "khu làm việc chung", "sở hữu trí tuệ")):
        refs.append(("04/2017/QH14", "12"))

    if has_any(q, ("tư vấn pháp luật", "tư vấn pháp lý", "tư vấn", "thông tin")):
        refs.append(("04/2017/QH14", "14"))

    if has_any(q, ("đào tạo", "nhân lực", "quản trị", "khởi sự kinh doanh")):
        refs.append(("04/2017/QH14", "15"))

    if has_any(q, ("khởi nghiệp sáng tạo", "startup")):
        refs.append(("04/2017/QH14", "17"))

    if has_any(q, ("cụm liên kết", "chuỗi giá trị")):
        refs.append(("04/2017/QH14", "19"))

    if has_any(q, ("quỹ phát triển", "vay vốn từ nguồn vốn của quỹ", "cho vay")):
        refs.append(("04/2017/QH14", "20"))

    if has_any(q, ("mặt bằng", "cơ sở ươm tạo")):
        refs.append(("04/2017/QH14", "11"))

    clean_refs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for law_id, article_id in refs:
        key = (canonical_law_id(law_id), str(article_id).lower())
        if key in keys or key in seen:
            continue
        clean_refs.append((law_id, article_id))
        seen.add(key)

    if not clean_refs:
        return None
    return "sme_foundation_law", clean_refs[:2]


def framework_companion_rule(row: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
    """Attach framework articles after a specific article of the same law is anchored."""
    keys = current_article_keys(row)
    refs: list[tuple[str, str]] = []
    for law_id, cfg in FRAMEWORK_COMPANION_RULES.items():
        law_articles = {article for current_law_id, article in keys if current_law_id == law_id}
        if not law_articles:
            continue
        if law_articles <= cfg["framework_or_preamble"]:
            continue
        refs.extend((law_id, article) for article in cfg["framework_articles"] if article not in law_articles)
    if not refs:
        return None
    return "framework_companion", refs[:2]


def doc_only_companion_rule(row: dict[str, Any], broad: bool = False) -> tuple[str, list[str]] | None:
    """Add direct companion documents without adding uncertain article labels."""
    q = normalize_text(row.get("question", ""))
    docs = current_doc_keys(row)
    refs: list[str] = []

    if "125/2020/NĐ-CP" in docs and "38/2019/QH14" not in docs and has_any(
        q,
        (
            "khai sai",
            "trốn thuế",
            "chậm nộp hồ sơ khai thuế",
            "miễn tiền phạt",
            "thời hiệu xử phạt",
            "chậm nộp tiền phạt",
            "vi phạm hành chính về thuế",
        ),
    ):
        refs.append("38/2019/QH14")
    if "126/2020/NĐ-CP" in docs and "38/2019/QH14" not in docs and has_any(q, ("ấn định thuế", "cưỡng chế", "gia hạn nộp thuế", "hoàn thuế", "xóa nợ", "xoá nợ")):
        refs.append("38/2019/QH14")
    if broad and "105/2020/TT-BTC" in docs and "38/2019/QH14" not in docs and has_any(q, ("đăng ký thuế", "mã số thuế", "chấm dứt hiệu lực mã số thuế")):
        refs.append("38/2019/QH14")
    if broad and "123/2020/NĐ-CP" in docs and "38/2019/QH14" not in docs and has_any(q, ("hóa đơn", "hoá đơn", "chứng từ điện tử")):
        refs.append("38/2019/QH14")
    if "65/2023/NĐ-CP" in docs and "50/2005/QH11" not in docs and has_any(q, ("sở hữu công nghiệp", "sở hữu trí tuệ", "nhãn hiệu", "sáng chế", "kiểu dáng", "văn bằng bảo hộ", "đơn đăng ký")):
        refs.append("50/2005/QH11")
    if broad and "168/2025/NĐ-CP" in docs and "59/2020/QH14" not in docs and "hộ kinh doanh" not in q and has_any(q, ("doanh nghiệp", "công ty", "chi nhánh", "văn phòng đại diện", "người đại diện", "vốn điều lệ")):
        refs.append("59/2020/QH14")
    if "12/2022/NĐ-CP" in docs:
        if broad and "45/2019/QH14" not in docs and has_any(q, ("người lao động", "nhân viên", "hợp đồng", "thử việc", "tiền lương", "kỷ luật", "đối thoại", "thương lượng", "thỏa ước", "thoả ước")):
            refs.append("45/2019/QH14")
        if broad and "84/2015/QH13" not in docs and has_any(q, ("an toàn", "vệ sinh lao động", "tai nạn lao động", "bệnh nghề nghiệp", "khám sức khỏe", "khám sức khoẻ", "độc hại", "nguy hiểm")):
            refs.append("84/2015/QH13")
    if broad and "80/2021/NĐ-CP" in docs and "04/2017/QH14" not in docs and has_any(q, ("doanh nghiệp nhỏ và vừa", "nhỏ và vừa", "hỗ trợ")):
        refs.append("04/2017/QH14")

    clean_refs = []
    seen = set()
    for law_id in refs:
        law_id = canonical_law_id(law_id)
        if law_id not in docs and law_id not in seen:
            clean_refs.append(law_id)
            seen.add(law_id)
    if not clean_refs:
        return None
    return "doc_only_companion", clean_refs[:2]


def propose_refs(row: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
    reasons: list[str] = []
    refs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set(current_article_keys(row))
    for rule in (
        social_insurance_late_payment_rule,
        sme_procurement_companion_rule,
        labor_penalty_foundation_rule,
        tax_foundation_companion_rule,
        tax_registration_foundation_rule,
        ip_infringement_companion_rule,
        ip_foundation_law_rule,
    ):
        result = rule(row)
        if result:
            reason, rule_refs = result
            added_for_rule = False
            for law_id, article_id in rule_refs:
                key = (canonical_law_id(law_id), str(article_id).lower())
                if key in seen:
                    continue
                refs.append((law_id, article_id))
                seen.add(key)
                added_for_rule = True
            if added_for_rule:
                reasons.append(reason)
    if not refs:
        return None
    return "+".join(reasons), refs[:4]


def create_submission(
    base_zip: Path,
    output_zip: Path,
    debug_path: Path,
    copy_to_submission: bool,
    doc_only: str = "off",
) -> dict[str, Any]:
    mapping = load_law_title_mapping(MAPPING_PATH)
    rows = load_rows(base_zip)
    debug_rows: list[dict[str, Any]] = []

    for row in rows:
        proposal = propose_refs(row)
        if proposal:
            reason, refs = proposal
            before_docs = list(row.get("relevant_docs", []))
            before_articles = list(row.get("relevant_articles", []))
            added = []
            for law_id, article_id in refs:
                if append_article(row, mapping, law_id, article_id):
                    added.append(f"{canonical_law_id(law_id)}|{article_id}")
            if added:
                update_answer(row)
                debug_rows.append(
                    {
                        "id": row.get("id"),
                        "reason": reason,
                        "question": row.get("question", ""),
                        "added": " || ".join(added),
                        "before_docs": " || ".join(before_docs),
                        "after_docs": " || ".join(row.get("relevant_docs", [])),
                        "before_articles": " || ".join(before_articles),
                        "after_articles": " || ".join(row.get("relevant_articles", [])),
                    }
                )

        if doc_only == "off":
            continue
        doc_proposal = doc_only_companion_rule(row, broad=doc_only == "broad")
        if not doc_proposal:
            continue
        reason, refs = doc_proposal
        before_docs = list(row.get("relevant_docs", []))
        before_articles = list(row.get("relevant_articles", []))
        added_docs = []
        for law_id in refs:
            if append_doc(row, mapping, law_id):
                added_docs.append(canonical_law_id(law_id))
        if not added_docs:
            continue
        debug_rows.append(
            {
                "id": row.get("id"),
                "reason": reason,
                "question": row.get("question", ""),
                "added": " || ".join(added_docs),
                "before_docs": " || ".join(before_docs),
                "after_docs": " || ".join(row.get("relevant_docs", [])),
                "before_articles": " || ".join(before_articles),
                "after_articles": " || ".join(row.get("relevant_articles", [])),
            }
        )

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_zip.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")

    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with debug_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "reason", "question", "added", "before_docs", "after_docs", "before_articles", "after_articles"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(debug_rows)

    if copy_to_submission:
        shutil.copyfile(output_zip, DEFAULT_READY)
        shutil.copyfile(output_zip, DEFAULT_READY_VARIANT)

    return {
        "rows": len(rows),
        "changed_rows": len(debug_rows),
        "doc_refs": sum(len(row.get("relevant_docs", [])) for row in rows),
        "article_refs": sum(len(row.get("relevant_articles", [])) for row in rows),
        "multi_doc_rows": sum(1 for row in rows if len(row.get("relevant_docs", [])) > 1),
        "multi_article_rows": sum(1 for row in rows if len(row.get("relevant_articles", [])) > 1),
        "output": str(output_zip),
        "debug": str(debug_path),
        "ready": str(DEFAULT_READY) if copy_to_submission else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--output", default=str(REPO_ROOT / "submission_variants" / "submission_agent_workflow.zip"))
    parser.add_argument("--debug", default=str(REPO_ROOT / "submission_variants" / "submission_agent_workflow_debug.csv"))
    parser.add_argument("--copy-to-submission", action="store_true")
    parser.add_argument("--doc-only", choices=("off", "conservative", "broad"), default="off")
    args = parser.parse_args()
    stats = create_submission(Path(args.base), Path(args.output), Path(args.debug), args.copy_to_submission, args.doc_only)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
