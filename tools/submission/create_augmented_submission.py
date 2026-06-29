"""Create an R2AI submission using DB-augmented retrieval plus legal rules.

This intentionally avoids the English MS-MARCO reranker used by the original
pipeline. The submission is top-1 article focused because the leaderboard
showed article precision matters more than adding top-k recall.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
import math
import pickle
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rank_bm25 import BM25Okapi

from _paths import REPO_ROOT
from utils.submission_formatter import (
    article_label,
    canonical_law_id,
    format_law_title,
    get_mapping_title,
    load_law_title_mapping,
)
from create_augmented_corpus_from_db import MANUAL_SEED_CODES


BASE_DIR = REPO_ROOT
AUGMENTED_CORPUS = BASE_DIR / "data" / "augmented" / "db_seed_articles.json"
MAPPING_PATH = BASE_DIR / "data" / "law_id_to_title.json"
INPUT_PATH = BASE_DIR / "R2AIStage1DATA.json"
OUTPUT_DIR = BASE_DIR / "submission_variants"
INDEX_PATH = BASE_DIR / "data" / "augmented" / "augmented_bm25.pkl"
DEBUG_PATH = OUTPUT_DIR / "submission_augmented_rerank_top1_debug.csv"
DB_HEURISTIC_RULES_PATH = BASE_DIR / "data" / "augmented" / "db_heuristic_rules.json"
RERANK_MIN_DELTA = 0.20

TITLE_OVERRIDES = {
    "105/2020/TT-BTC": "Thông tư 105/2020/TT-BTC Hướng dẫn về đăng ký thuế",
    "54/2019/TT-BTC": (
        "Thông tư 54/2019/TT-BTC Hướng dẫn lập dự toán, quản lý, sử dụng và quyết toán "
        "kinh phí ngân sách nhà nước hỗ trợ doanh nghiệp nhỏ và vừa sử dụng dịch vụ tư vấn"
    ),
}


STOPWORDS = {
    "các", "của", "và", "với", "khi", "thì", "là", "gì", "nào", "những",
    "được", "phải", "cần", "có", "không", "trong", "cho", "về", "theo",
    "nếu", "như", "tại", "sao", "một", "công", "ty", "doanh", "nghiệp",
    "hỏi", "tôi", "hiện", "nay", "bao", "nhiêu", "thế", "nào",
}

INTENT_TERMS = {
    "condition": (
        ("điều kiện", "đáp ứng", "tiêu chí", "xác định", "được hỗ trợ"),
        ("điều kiện", "tiêu chí", "xác định", "đối tượng", "lựa chọn"),
    ),
    "procedure": (
        ("hồ sơ", "thủ tục", "trình tự", "đăng ký", "nộp hồ sơ", "đề nghị"),
        ("hồ sơ", "trình tự", "thủ tục", "đăng ký", "tiếp nhận", "đề nghị"),
    ),
    "penalty": (
        ("phạt", "xử phạt", "vi phạm", "mức phạt", "chế tài"),
        ("phạt", "xử phạt", "vi phạm", "biện pháp khắc phục", "hình thức xử phạt"),
    ),
    "support": (
        ("hỗ trợ", "mức hỗ trợ", "chi phí", "ngân sách", "ưu đãi"),
        ("hỗ trợ", "mức hỗ trợ", "chi phí", "ngân sách", "ưu đãi", "nội dung hỗ trợ"),
    ),
    "rights": (
        ("quyền", "nghĩa vụ", "trách nhiệm", "được làm gì", "phải làm gì"),
        ("quyền", "nghĩa vụ", "trách nhiệm", "quyền và nghĩa vụ"),
    ),
    "contract": (
        ("hợp đồng", "nội dung hợp đồng", "chuyển nhượng", "sử dụng"),
        ("hợp đồng", "nội dung", "hiệu lực", "chuyển nhượng", "sử dụng"),
    ),
}

DB_RULE_SKIP = {
    # Too broad: these often match detailed invoice/registration questions better
    # handled by article BM25 inside the correct domain law.
    "dang_ky_kinh_doanh",
    "hoa_don_dien_tu",
    "khau_tru_thue",
}

DB_RULE_ALIASES = {
    "mat_chay_hong_hoa_don": ("mất cháy hỏng hóa đơn", "mất hóa đơn", "cháy hóa đơn", "hỏng hóa đơn"),
    "doanh_nghiep_xa_hoi": ("doanh nghiệp xã hội",),
    "thay_doi_dia_chi": ("thay đổi địa chỉ", "đổi địa chỉ trụ sở"),
    "thu_hoi_giay_phep_kinh_doanh": ("thu hồi giấy chứng nhận đăng ký", "thu hồi giấy phép kinh doanh"),
    "khuyen_mai_giam_gia_qua_50": ("khuyến mại", "giảm giá quá 50", "giảm giá trên 50"),
    "su_kien_bat_kha_khang": ("sự kiện bất khả kháng", "bất khả kháng"),
    "don_phuong_cham_dut_hop_dong": ("đơn phương chấm dứt hợp đồng lao động",),
    "ky_hop_dong_lao_dong": ("ký hợp đồng lao động", "giao kết hợp đồng lao động"),
    "quay_roi_tinh_duc_noi_lam_viec": ("quấy rối tình dục",),
    "quang_cao_sai_su_that": ("quảng cáo sai sự thật", "quảng cáo gian dối"),
    "nghi_thai_san_nam": ("lao động nam", "vợ sinh con", "thai sản"),
    "giai_the_doanh_nghiep": ("giải thể doanh nghiệp", "đăng ký giải thể"),
    "tam_ngung_kinh_doanh": ("tạm ngừng kinh doanh", "tạm ngừng hoạt động kinh doanh"),
    "boi_thuong_thiet_hai": ("bồi thường thiệt hại",),
    "phat_vi_pham_hop_dong": ("phạt vi phạm hợp đồng",),
    "boi_thuong_chi_phi_dao_tao": ("bồi thường chi phí đào tạo", "chi phí đào tạo"),
    "dong_bhxh_bat_buoc": ("đóng bảo hiểm xã hội bắt buộc", "tham gia bảo hiểm xã hội bắt buộc"),
    "muc_dong_bhxh": ("mức đóng bảo hiểm xã hội", "tỷ lệ đóng bảo hiểm xã hội"),
    "cham_nop_thue": ("chậm nộp thuế", "chậm nộp tiền thuế"),
    "hoan_thue_gtgt": ("hoàn thuế giá trị gia tăng", "hoàn thuế gtgt"),
    "uu_dai_thue_tndn": ("ưu đãi thuế thu nhập doanh nghiệp", "ưu đãi thuế tndn"),
    "gia_tri_phap_ly_hop_dong_dien_tu": ("giá trị pháp lý hợp đồng điện tử", "hợp đồng điện tử có giá trị"),
    "dang_ky_chi_nhanh_nuoc_ngoai": ("đăng ký chi nhánh nước ngoài", "chi nhánh ở nước ngoài"),
    "dang_ky_nhan_hieu": ("đăng ký nhãn hiệu",),
    "hoan_thue_xuat_khau": ("hoàn thuế xuất khẩu", "hoàn thuế hàng hóa xuất khẩu"),
    "chu_ky_so_hop_dong": ("chữ ký số", "hợp đồng điện tử"),
    "website_thuong_mai_dien_tu": ("website thương mại điện tử",),
    "chuyen_doi_loai_hinh_doanh_nghiep": ("chuyển đổi loại hình doanh nghiệp",),
    "thay_doi_nguoi_dai_dien": ("thay đổi người đại diện", "đổi người đại diện"),
    "hop_dong_mua_ban": ("hợp đồng mua bán",),
    "thanh_lap_doanh_nghiep": ("thành lập doanh nghiệp", "đăng ký thành lập doanh nghiệp"),
}


@dataclass(frozen=True)
class Rule:
    name: str
    law_id: str
    article_id: str
    all_terms: tuple[str, ...] = ()
    any_terms: tuple[str, ...] = ()
    none_terms: tuple[str, ...] = ()
    priority: int = 100


RULES: list[Rule] = [
    Rule("sme_incubator_tax_land", "04/2017/QH14", "12", ("ươm tạo",), ("khu làm việc chung", "thuế", "đất"), priority=500),
    Rule("sme_procurement_preference", "63/2014/NĐ-CP", "6", ("đấu thầu",), ("doanh nghiệp nhỏ", "nhỏ và vừa"), priority=500),
    Rule("labor_keep_original_cert", "12/2022/NĐ-CP", "9", ("giữ bản chính",), ("bằng cấp", "văn bằng", "chứng chỉ", "giấy tờ"), priority=500),
    Rule("sme_household_conversion", "04/2017/QH14", "16", ("hộ kinh doanh", "chuyển đổi"), ("doanh nghiệp nhỏ", "nhỏ và vừa"), priority=450),
    Rule("sme_production_premises", "04/2017/QH14", "11", ("mặt bằng sản xuất",), ("giá thuê", "thuê mặt bằng"), priority=500),
    Rule("sme_financial_report_common", "133/2016/TT-BTC", "81", ("báo cáo tài chính",), ("thông tin chung", "trình bày"), priority=480),
    Rule("labor_late_social_insurance", "12/2022/NĐ-CP", "39", ("chậm đóng", "bảo hiểm xã hội"), priority=500),
    Rule(
        "sme_startup_condition",
        "80/2021/NĐ-CP",
        "20",
        ("khởi nghiệp sáng tạo",),
        ("điều kiện", "lựa chọn", "giải thưởng", "tiêu chí"),
        ("cụm liên kết ngành", "đăng ký thành lập", "người đại diện", "hồ sơ đăng ký doanh nghiệp"),
        priority=460,
    ),
    Rule("sme_value_chain_support", "80/2021/NĐ-CP", "25", ("chuỗi giá trị",), ("nội dung hỗ trợ", "chi phí", "đào tạo"), priority=460),
    Rule("sme_value_chain_selection", "80/2021/NĐ-CP", "24", ("chuỗi giá trị",), ("lựa chọn", "hình thức", "tiêu chí"), priority=455),
    Rule("sme_cluster_selection", "80/2021/NĐ-CP", "23", ("cụm liên kết ngành",), ("lựa chọn", "tiêu chí", "hình thức"), priority=455),
    Rule("sme_fund_tasks", "39/2019/NĐ-CP", "5", ("quỹ phát triển doanh nghiệp nhỏ và vừa",), ("chức năng", "nhiệm vụ", "hỗ trợ"), priority=500),
    Rule("sme_criteria", "80/2021/NĐ-CP", "5", ("xác định", "doanh nghiệp nhỏ"), ("tiêu chí", "số lao động", "tổng nguồn vốn", "doanh thu"), priority=455),
    Rule("sme_credit_access", "04/2017/QH14", "8", ("tiếp cận tín dụng",), ("hỗ trợ",), priority=470),
    Rule("sme_credit_guarantee_conditions", "34/2018/NĐ-CP", "16", ("quỹ bảo lãnh tín dụng",), ("điều kiện", "cấp bảo lãnh"), priority=500),
    Rule("sme_responsibility_counterpart", "04/2017/QH14", "28", ("nguồn lực đối ứng",), ("trách nhiệm", "nhận hỗ trợ"), priority=500),
    Rule("sme_support_funding_sources", "04/2017/QH14", "6", ("nguồn vốn hỗ trợ",), ("nhỏ và vừa", "doanh nghiệp nhỏ"), priority=500),
    Rule("sme_training_types", "05/2019/TT-BKHĐT", "3", ("khóa đào tạo trực tiếp",), ("bao gồm", "loại khóa", "gồm", "những loại"), priority=430),
    Rule("sme_ministry_finance_role", "04/2017/QH14", "23", ("bộ tài chính",), ("thuế", "kế toán", "trách nhiệm"), priority=480),
    Rule("sme_consulting_support", "80/2021/NĐ-CP", "13", ("hỗ trợ tư vấn",), ("nội dung", "mức hỗ trợ", "tư vấn viên"), priority=460),
    Rule("sme_training_audience", "05/2019/TT-BKHĐT", "3", ("đối tượng", "khóa đào tạo trực tiếp"), ("khởi sự kinh doanh",), priority=430),
    Rule("sme_consultant_cost", "54/2019/TT-BTC", "7", ("tư vấn viên",), ("chi phí", "ngân sách nhà nước", "hỗ trợ tư vấn"), priority=520),
    Rule("sme_ecommerce_account", "80/2021/NĐ-CP", "22", ("sàn thương mại điện tử quốc tế",), ("duy trì tài khoản", "khởi nghiệp sáng tạo"), priority=480),
    Rule("sme_support_application", "80/2021/NĐ-CP", "32", ("hồ sơ",), ("đề xuất hỗ trợ", "nhiều nội dung hỗ trợ"), priority=450),
    Rule("sme_digital_solution", "80/2021/NĐ-CP", "11", ("giải pháp chuyển đổi số",), ("thuê", "mua", "hỗ trợ chi phí"), priority=500),
    Rule("tax_company_code", "59/2020/QH14", "29", ("mã số thuế", "doanh nghiệp"), ("mã số nào", "quy định là"), priority=500),
    Rule("tax_registration_scope", "105/2020/TT-BTC", "1", ("phạm vi đăng ký thuế",), ("nội dung", "cụ thể"), priority=520),
    Rule("tax_extension_force_majeure", "38/2019/QH14", "46", ("bất khả kháng", "hồ sơ khai thuế"), ("gia hạn",), priority=500),
    Rule("tax_overpayment", "38/2019/QH14", "60", ("nộp thừa",), ("tiền thuế", "tiền phạt"), priority=500),
    Rule(
        "tax_exemption_reduction",
        "38/2019/QH14",
        "79",
        (),
        ("được hưởng miễn thuế", "được hưởng giảm thuế", "được hưởng miễn giảm thuế", "làm gì để được hưởng"),
        ("hồ sơ", "thời hạn", "nộp hồ sơ", "gửi cơ quan", "bằng hình thức", "thiên tai", "hỏa hoạn", "lao động nữ"),
        priority=480,
    ),
    Rule("tax_electronic_document", "38/2019/QH14", "94", ("chứng từ điện tử",), ("quản lý thuế", "định nghĩa"), priority=500),
    Rule("tax_einvoice_types", "123/2020/NĐ-CP", "8", ("loại hóa đơn",), ("hóa đơn điện tử",), priority=500),
    Rule("tax_assessment_cases", "38/2019/QH14", "50", ("ấn định thuế",), ("trường hợp", "cơ quan thuế"), priority=500),
    Rule("tax_refund_wrong_collection", "38/2019/QH14", "61", ("hoàn trả", "thuế"), ("thu không đúng", "khiếu nại"), priority=500),
    Rule("tax_enforcement_measures", "38/2019/QH14", "125", ("nợ thuế",), ("biện pháp cưỡng chế", "cưỡng chế"), priority=500),
    Rule("tax_delete_debt_bankruptcy", "38/2019/QH14", "85", ("xóa", "tiền thuế nợ"), ("phá sản", "tuyên bố phá sản"), priority=470),
    Rule("tax_buyer_foreign_address", "123/2020/NĐ-CP", "10", ("khách hàng nước ngoài",), ("địa chỉ người mua", "lập hóa đơn"), priority=460),
    Rule("tax_ecommerce_platform_household_withholding", "48/2024/QH15", "4", ("nền tảng thương mại điện tử", "hộ kinh doanh"), ("khấu trừ", "nộp thuế thay"), priority=540),
    Rule("commercial_penalty_requires_agreement", "36/2005/QH11", "300", ("phạt vi phạm hợp đồng",), ("không thỏa thuận", "có được yêu cầu"), priority=520),
    Rule("commercial_penalty_cap", "36/2005/QH11", "301", ("phạt vi phạm",), ("tối đa", "bao nhiêu", "mức phạt"), priority=520),
    Rule("commercial_damages", "36/2005/QH11", "302", ("bồi thường thiệt hại",), ("hợp đồng thương mại", "hủy bỏ hợp đồng", "huỷ bỏ hợp đồng"), priority=500),
    Rule("electronic_signature", "20/2023/QH15", "22", ("hợp đồng", "chữ ký"), ("chữ ký số", "chữ ký điện tử"), priority=500),
    Rule("electronic_contract_validity", "20/2023/QH15", "34", ("hợp đồng điện tử",), ("giá trị pháp lý", "công nhận", "không ký hợp đồng giấy"), priority=500),
    Rule("promotion_obligations", "36/2005/QH11", "96", ("khuyến mại", "nghĩa vụ"), ("khách hàng", "cơ bản"), priority=510),
    Rule(
        "promotion_customer_notice_methods",
        "36/2005/QH11",
        "98",
        ("thông báo khuyến mại",),
        ("hàng hóa", "hàng hoá", "dịch vụ", "khách hàng", "cách thức"),
        ("sở công thương", "cơ quan quản lý", "cơ quan nhà nước"),
        priority=515,
    ),
    Rule(
        "promotion_notice",
        "81/2018/NĐ-CP",
        "17",
        ("khuyến mại",),
        ("sở công thương", "cơ quan quản lý", "cơ quan nhà nước"),
        ("khách hàng", "cách thức"),
        priority=500,
    ),
    Rule("promotion_value_cap", "81/2018/NĐ-CP", "6", ("khuyến mại",), ("50%", "quá 50", "trên 50", "hạn mức"), priority=500),
    Rule("advertising_false", "16/2012/QH13", "8", ("quảng cáo",), ("sai sự thật", "gian dối", "nhầm lẫn"), priority=500),
    Rule("foreign_ad_rep_office", "16/2012/QH13", "41", ("văn phòng đại diện", "quảng cáo"), ("trực tiếp", "kinh doanh"), priority=500),
    Rule(
        "enterprise_address_change",
        "168/2025/NĐ-CP",
        "40",
        ("thay đổi địa chỉ",),
        ("trụ sở", "địa chỉ trụ sở"),
        ("văn bằng", "bằng độc quyền", "nhãn hiệu", "sáng chế", "hộ kinh doanh"),
        priority=480,
    ),
    Rule(
        "enterprise_representative_change",
        "168/2025/NĐ-CP",
        "43",
        ("thay đổi người đại diện theo pháp luật",),
        ("hồ sơ", "đăng ký"),
        ("quỹ phát triển doanh nghiệp nhỏ và vừa",),
        priority=480,
    ),
    Rule(
        "enterprise_business_suspension",
        "168/2025/NĐ-CP",
        "60",
        ("tạm ngừng kinh doanh",),
        ("hồ sơ", "thông báo", "trước bao nhiêu ngày", "thủ tục", "quy trình"),
        (
            "mã số thuế",
            "nhà cung cấp",
            "giao dịch",
            "nợ thuế",
            "giải thể",
            "tổ chức lại",
            "sáp nhập",
            "chi nhánh đã ngừng",
            "ngừng hoạt động quá 12 tháng",
            "thu hồi giấy chứng nhận",
            "đăng ký thuế",
        ),
        priority=480,
    ),
    Rule("household_business_suspension", "168/2025/NĐ-CP", "103", ("hộ kinh doanh", "tạm ngừng kinh doanh"), priority=490),
    Rule("social_insurance_rate", "41/2024/QH15", "33", ("mức đóng bảo hiểm xã hội",), ("tỷ lệ", "bao nhiêu phần trăm", "hằng tháng"), priority=480),
    Rule("ip_third_party_opinion", "50/2005/QH11", "112", ("người thứ ba",), ("ý kiến phản đối", "văn bằng bảo hộ"), priority=500),
    Rule("ip_grant_certificate", "50/2005/QH11", "118", ("thẩm định nội dung",), ("cấp văn bằng bảo hộ", "thông báo kết quả"), priority=500),
    Rule("ip_trademark_infringement", "50/2005/QH11", "129", ("nhãn hiệu",), ("xâm phạm", "dấu hiệu trùng"), priority=500),
    Rule("ip_assignment_content", "50/2005/QH11", "140", ("hợp đồng chuyển nhượng", "sở hữu công nghiệp"), ("nội dung", "nội dung chính"), priority=500),
    Rule("ip_temporary_rights", "50/2005/QH11", "131", ("sáng chế", "nộp đơn"), ("kinh doanh", "quyền làm gì", "người khác"), priority=500),
    Rule("ip_license_content", "50/2005/QH11", "144", ("hợp đồng sử dụng", "sở hữu công nghiệp"), ("nội dung chủ yếu", "soạn thảo"), priority=500),
    Rule("ip_industrial_design_infringement", "50/2005/QH11", "126", ("kiểu dáng công nghiệp",), ("xâm phạm", "không xin phép", "được bảo hộ"), priority=500),
    Rule("ip_assignment_effect", "50/2005/QH11", "148", ("hợp đồng chuyển nhượng", "sở hữu công nghiệp"), ("hiệu lực",), priority=500),
    Rule("ip_plant_variety_maintenance", "88/2010/NĐ-CP", "24", ("duy trì hiệu lực",), ("bằng bảo hộ giống cây trồng", "lệ phí"), priority=450),
    Rule("labor_try_wage", "12/2022/NĐ-CP", "10", ("thử việc",), ("85%", "lương"), priority=500),
    Rule("labor_hygiene_record", "12/2022/NĐ-CP", "21", ("hồ sơ vệ sinh môi trường lao động",), ("yếu tố có hại",), priority=500),
    Rule("labor_health_check", "12/2022/NĐ-CP", "22", ("khám sức khỏe định kỳ",), ("nhân viên", "người lao động"), priority=450),
    Rule("labor_salary_discipline", "12/2022/NĐ-CP", "19", ("phạt tiền", "cắt lương"), ("kỷ luật lao động",), priority=500),
    Rule("labor_female_menstruation", "12/2022/NĐ-CP", "28", ("lao động nữ", "hành kinh"), ("30 phút",), priority=500),
    Rule("labor_strict_safety_machine", "12/2022/NĐ-CP", "24", ("yêu cầu nghiêm ngặt", "an toàn lao động"), ("máy móc", "khai báo"), priority=500),
    Rule("labor_environment_monitoring", "12/2022/NĐ-CP", "27", ("quan trắc môi trường lao động",), ("phạt",), priority=500),
    Rule("labor_safety_card", "12/2022/NĐ-CP", "25", ("thẻ an toàn",), ("yêu cầu nghiêm ngặt", "an toàn"), priority=500),
    Rule("labor_penalty_forms", "12/2022/NĐ-CP", "3", ("hình thức xử phạt",), ("lao động", "bảo hiểm xã hội"), priority=470),
    Rule("labor_safety_report", "12/2022/NĐ-CP", "20", ("báo cáo", "an toàn, vệ sinh lao động"), ("không đúng thời hạn", "không báo cáo"), priority=500),
    Rule("labor_minor_logbook", "12/2022/NĐ-CP", "29", ("lao động chưa thành niên",), ("sổ theo dõi",), priority=500),
    Rule("labor_strike_dismissal", "12/2022/NĐ-CP", "34", ("sa thải", "đình công"), priority=500),
    Rule("labor_union_fee", "12/2022/NĐ-CP", "38", ("kinh phí công đoàn",), ("chậm đóng",), priority=500),
    Rule("labor_first_aid", "12/2022/NĐ-CP", "23", ("sơ cứu",), ("cấp cứu", "tai nạn lao động"), priority=500),
    Rule("labor_bhxh_book_return", "12/2022/NĐ-CP", "12", ("sổ bảo hiểm xã hội",), ("chấm dứt hợp đồng", "không trả"), priority=470),
    Rule("labor_disability_overtime", "12/2022/NĐ-CP", "31", ("người lao động khuyết tật",), ("làm thêm giờ", "không đồng ý"), priority=500),
]


def normalize_text(text: str) -> str:
    text = str(text or "").lower().replace("ð", "đ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_accents(text: str) -> str:
    text = normalize_text(text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    tokens = re.findall(r"[0-9a-zA-ZÀ-ỹĐđ/%.-]+", text)
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) >= 3}


def query_phrases(question: str) -> list[str]:
    tokens = tokenize(question)
    phrases: list[str] = []
    for size in (4, 3, 2):
        for i in range(0, max(len(tokens) - size + 1, 0)):
            phrase = " ".join(tokens[i : i + size])
            if len(phrase) >= 8:
                phrases.append(phrase)
    return phrases[:40]


def rule_match(rule: Rule, question: str) -> bool:
    q = normalize_text(question)
    if any(term in q for term in rule.none_terms):
        return False
    if rule.all_terms and not all(term in q for term in rule.all_terms):
        return False
    if rule.any_terms and not any(term in q for term in rule.any_terms):
        return False
    return True


def find_rule(question: str) -> Optional[Rule]:
    matches = [rule for rule in RULES if rule_match(rule, question)]
    if not matches:
        return None
    return sorted(matches, key=lambda rule: rule.priority, reverse=True)[0]


@lru_cache(maxsize=1)
def load_db_heuristic_rules() -> tuple[dict[str, Any], ...]:
    if not DB_HEURISTIC_RULES_PATH.exists():
        return ()
    try:
        rows = json.loads(DB_HEURISTIC_RULES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ()
    return tuple(row for row in rows if row.get("target_document_code") and row.get("target_article_number"))


def db_heuristic_codes() -> set[str]:
    return {canonical_law_id(row["target_document_code"]) for row in load_db_heuristic_rules()}


def keyword_tokens(keyword: str) -> list[str]:
    ignored = {"sme", "va", "voi", "cho", "cua", "ve", "tai", "trong"}
    return [part for part in strip_accents(keyword).split("_") if len(part) > 2 and part not in ignored]


def db_rule_match(row: dict[str, Any], question: str) -> bool:
    keyword = str(row.get("intent_keyword", ""))
    if keyword in DB_RULE_SKIP:
        return False

    q = strip_accents(question)
    q_tokens = set(tokenize(q))
    aliases = DB_RULE_ALIASES.get(keyword, ())
    for alias in aliases:
        alias_ascii = strip_accents(alias)
        alias_tokens = [token for token in tokenize(alias_ascii) if len(token) > 2]
        if alias_ascii in q or (alias_tokens and all(token in q_tokens for token in alias_tokens)):
            return True

    tokens = keyword_tokens(keyword)
    if len(tokens) < 2:
        return False
    if keyword in {"thanh_lap_doanh_nghiep", "hop_dong_mua_ban"} and len(tokens) < 3:
        return False
    return all(token in q_tokens for token in tokens)


def find_db_heuristic_rule(question: str) -> Optional[dict[str, Any]]:
    matches = [row for row in load_db_heuristic_rules() if db_rule_match(row, question)]
    if not matches:
        return None
    return sorted(matches, key=lambda row: float(row.get("boost_score") or 0), reverse=True)[0]


def load_augmented_articles(include_db_heuristic_codes: bool = False) -> list[dict[str, Any]]:
    articles = json.loads(AUGMENTED_CORPUS.read_text(encoding="utf-8"))
    core_laws = {canonical_law_id(code) for code in MANUAL_SEED_CODES}
    if include_db_heuristic_codes:
        core_laws.update(db_heuristic_codes())
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for article in articles:
        law_id = canonical_law_id(article["law_id"])
        if law_id not in core_laws:
            continue
        key = (law_id, str(article["article_id"]))
        current = deduped.get(key)
        if current is None or status_bonus(article) > status_bonus(current):
            article["law_id"] = key[0]
            deduped[key] = article
    return list(deduped.values())


def status_bonus(article: dict[str, Any]) -> float:
    status = normalize_text(article.get("status", ""))
    if status == "effective":
        return 0.18
    if status == "partially_expired":
        return 0.05
    if status == "expired":
        return -0.12
    return 0.0


def article_text(article: dict[str, Any]) -> str:
    law_id = article.get("law_id", "")
    article_id = article.get("article_id", "")
    return " ".join(
        [
            str(law_id),
            str(law_id),
            f"Điều {article_id}",
            str(article.get("document_title", "")),
            str(article.get("document_title", "")),
            str(article.get("title", "")),
            str(article.get("title", "")),
            str(article.get("title", "")),
            str(article.get("content", ""))[:2200],
        ]
    )


def build_or_load_index(articles: Sequence[dict[str, Any]], force_rebuild: bool = False):
    if INDEX_PATH.exists() and not force_rebuild:
        with INDEX_PATH.open("rb") as f:
            data = pickle.load(f)
        if data.get("count") == len(articles):
            return data["bm25"], data["tokenized"]

    tokenized = [tokenize(article_text(article)) for article in articles]
    bm25 = BM25Okapi(tokenized, b=0.55, k1=1.35)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("wb") as f:
        pickle.dump({"count": len(articles), "bm25": bm25, "tokenized": tokenized}, f)
    return bm25, tokenized


def domain_boost(question: str, article: dict[str, Any]) -> float:
    q = normalize_text(question)
    law_id = canonical_law_id(article.get("law_id", ""))
    title = normalize_text(article.get("title", "") + " " + article.get("document_title", ""))
    boost = status_bonus(article)

    if any(term in q for term in ["doanh nghiệp nhỏ", "nhỏ và vừa", "hỗ trợ"]):
        if law_id in {"04/2017/QH14", "80/2021/NĐ-CP", "39/2019/NĐ-CP", "34/2018/NĐ-CP", "38/2018/NĐ-CP", "05/2019/TT-BKHĐT", "54/2019/TT-BTC"}:
            boost += 0.45
    if "thuế" in q or "hóa đơn" in q or "chứng từ" in q:
        if law_id in {"38/2019/QH14", "126/2020/NĐ-CP", "123/2020/NĐ-CP", "125/2020/NĐ-CP", "133/2016/TT-BTC"}:
            boost += 0.45
    if "lao động" in q or "nhân viên" in q or "bảo hiểm xã hội" in q:
        if law_id in {"12/2022/NĐ-CP", "45/2019/QH14", "41/2024/QH15", "84/2015/QH13", "85/2015/NĐ-CP"}:
            boost += 0.45
    if "sở hữu" in q or "nhãn hiệu" in q or "sáng chế" in q or "kiểu dáng" in q or "văn bằng bảo hộ" in q:
        if law_id in {"50/2005/QH11", "65/2023/NĐ-CP", "103/2006/NĐ-CP", "99/2013/NĐ-CP", "22/2018/NĐ-CP"}:
            boost += 0.45
    if "đấu thầu" in q and law_id in {"43/2013/QH13", "63/2014/NĐ-CP"}:
        boost += 0.60
    if "hợp đồng" in q and law_id in {"91/2015/QH13", "36/2005/QH11", "45/2019/QH14", "50/2005/QH11"}:
        boost += 0.25

    query_tokens = set(tokenize(question))
    title_hits = sum(1 for token in query_tokens if token in title)
    boost += min(title_hits * 0.04, 0.45)
    return boost


def article_relevance_boost(question: str, article: dict[str, Any]) -> float:
    q = normalize_text(question)
    title = normalize_text(article.get("title", ""))
    content = normalize_text(str(article.get("content", ""))[:1800])
    doc_title = normalize_text(article.get("document_title", ""))
    title_text = f"{title} {doc_title}"
    full_text = f"{title_text} {content}"
    q_tokens = content_tokens(question)
    if not q_tokens:
        return 0.0

    title_tokens = content_tokens(title_text)
    content_token_set = content_tokens(content)
    title_overlap = len(q_tokens & title_tokens) / max(len(q_tokens), 1)
    content_overlap = len(q_tokens & content_token_set) / max(len(q_tokens), 1)
    boost = title_overlap * 0.95 + content_overlap * 0.35

    phrase_boost = 0.0
    for phrase in query_phrases(question):
        if phrase in title_text:
            phrase_boost += 0.22
        elif phrase in content:
            phrase_boost += 0.075
    boost += min(phrase_boost, 0.95)

    for question_terms, article_terms in INTENT_TERMS.values():
        if any(term in q for term in question_terms) and any(term in full_text for term in article_terms):
            boost += 0.28

    article_id = str(article.get("article_id", "")).strip().lower()
    explicit_article = re.search(r"điều\s+([0-9]+[a-z]?)", q)
    if explicit_article and explicit_article.group(1) == article_id:
        boost += 2.0

    if article_id in {"1", "2", "3"} and not any(term in q for term in ["phạm vi", "đối tượng", "giải thích", "hình thức xử phạt"]):
        boost -= 0.18
    return boost


def rerank_within_law(
    question: str,
    selected_idx: int,
    articles: Sequence[dict[str, Any]],
    scores: Sequence[float],
    max_score: float,
    min_delta: float = RERANK_MIN_DELTA,
) -> dict[str, Any]:
    selected = articles[selected_idx]
    law_id = canonical_law_id(selected.get("law_id", ""))
    selected_article_id = str(selected.get("article_id", ""))
    best_article = selected
    best_score = -math.inf
    original_score = -math.inf

    for idx, article in enumerate(articles):
        if canonical_law_id(article.get("law_id", "")) != law_id:
            continue
        normalized_score = float(scores[idx]) / max_score if max_score else 0.0
        final_score = (
            normalized_score * 0.55
            + domain_boost(question, article) * 0.45
            + article_relevance_boost(question, article)
        )
        if str(article.get("article_id", "")) == selected_article_id:
            original_score = final_score
        if final_score > best_score:
            best_score = final_score
            best_article = article

    delta = best_score - original_score
    if str(best_article.get("article_id", "")) != selected_article_id and delta < min_delta:
        output = dict(selected)
        output["rank_score"] = original_score
        output["source"] = "augmented_bm25_same_law_low_delta"
    else:
        output = dict(best_article)
        output["rank_score"] = best_score
        if str(best_article.get("article_id", "")) == selected_article_id:
            output["source"] = "augmented_bm25_same_law"
        else:
            output["source"] = "augmented_bm25_same_law_rerank"
    output["original_article_id"] = selected_article_id
    output["original_article_score"] = original_score
    output["best_candidate_article_id"] = str(best_article.get("article_id", ""))
    output["best_candidate_score"] = best_score
    output["rerank_delta"] = delta
    return output


def rank_article(
    question: str,
    articles: Sequence[dict[str, Any]],
    bm25,
    rerank_min_delta: float = RERANK_MIN_DELTA,
    use_db_heuristics: bool = False,
) -> dict[str, Any]:
    rule = find_rule(question)
    if rule:
        return {
            "law_id": canonical_law_id(rule.law_id),
            "article_id": str(rule.article_id),
            "title": f"Điều {rule.article_id}",
            "document_title": "",
            "source": f"rule:{rule.name}",
        }

    db_rule = find_db_heuristic_rule(question) if use_db_heuristics else None
    if db_rule:
        article_id = str(db_rule["target_article_number"])
        return {
            "law_id": canonical_law_id(db_rule["target_document_code"]),
            "article_id": article_id,
            "title": f"Điều {article_id}",
            "document_title": "",
            "source": f"db_rule:{db_rule.get('intent_keyword', '')}",
            "rank_score": float(db_rule.get("boost_score") or 0),
        }

    scores = bm25.get_scores(tokenize(question))
    if len(scores) == 0:
        return articles[0]

    # Inspect only top lexical candidates, then add legal-domain boosts.
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:60]
    best_idx = top_indices[0]
    best_score = -math.inf
    max_score = float(scores[top_indices[0]]) if top_indices else 1.0
    max_score = max(max_score, 1e-9)
    for idx in top_indices:
        article = articles[idx]
        normalized_score = float(scores[idx]) / max_score
        final_score = normalized_score + domain_boost(question, article)
        if final_score > best_score:
            best_score = final_score
            best_idx = idx
    selected = rerank_within_law(question, best_idx, articles, scores, max_score, min_delta=rerank_min_delta)
    selected["law_rank_score"] = best_score
    return selected


def build_article_lookup(articles: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(canonical_law_id(article["law_id"]), str(article["article_id"])): article for article in articles}


def resolve_article(article: dict[str, Any], lookup: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    key = (canonical_law_id(article.get("law_id", "")), str(article.get("article_id", "")))
    if key in lookup:
        resolved = dict(lookup[key])
        for field in (
            "source",
            "rank_score",
            "law_rank_score",
            "original_article_id",
            "original_article_score",
            "best_candidate_article_id",
            "best_candidate_score",
            "rerank_delta",
        ):
            if field in article:
                resolved[field] = article[field]
        return resolved
    return article


def make_refs(article: dict[str, Any], mapping: dict[str, str]) -> tuple[list[str], list[str]]:
    law_id = canonical_law_id(article.get("law_id", ""))
    raw_title = TITLE_OVERRIDES.get(law_id) or article.get("document_title") or get_mapping_title(mapping, law_id)
    law_title = format_law_title(law_id, raw_title)
    article_id = str(article.get("article_id", "")).strip()
    label = f"Điều {article_id}" if article_id else article_label(article.get("title", ""), "")
    return [f"{law_id}|{law_title}"], [f"{law_id}|{law_title}|{label}"]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|(?<=;)\s+", text)
    return [part.strip(" -") for part in parts if len(part.strip()) >= 20]


def answer_from_article(question: str, article: dict[str, Any], article_ref: str) -> str:
    law_id, _title, label = article_ref.split("|", 2)
    title = str(article.get("title", "") or label).strip()
    content = str(article.get("content", "") or title)
    q_tokens = content_tokens(question)
    sentences = split_sentences(content)
    scored = []
    for sentence in sentences[:30]:
        s_tokens = content_tokens(sentence)
        overlap = len(q_tokens & s_tokens)
        if overlap:
            scored.append((overlap, len(sentence), sentence))
    if scored:
        selected = [item[2] for item in sorted(scored, key=lambda item: (-item[0], item[1]))[:3]]
    else:
        selected = sentences[:3]

    summary = " ".join(selected)
    summary = re.sub(r"\s+", " ", summary).strip()
    if len(summary) > 950:
        summary = summary[:950].rsplit(" ", 1)[0].strip() + "."
    if not summary:
        summary = "Áp dụng nội dung của điều luật này để xác định quyền, nghĩa vụ, điều kiện hoặc chế tài tương ứng."

    return f"Căn cứ pháp luật: {law_id}|{label}. Trả lời: {summary}"


def create_submission(
    input_path: Path,
    output_zip: Path,
    force_rebuild: bool = False,
    debug_output: Optional[Path] = DEBUG_PATH,
    rerank_min_delta: float = RERANK_MIN_DELTA,
    use_db_heuristics: bool = False,
    include_db_heuristic_codes: bool = False,
) -> None:
    mapping = load_law_title_mapping(MAPPING_PATH)
    articles = load_augmented_articles(include_db_heuristic_codes=include_db_heuristic_codes)
    lookup = build_article_lookup(articles)
    bm25, _tokenized = build_or_load_index(articles, force_rebuild=force_rebuild)
    questions = json.loads(input_path.read_text(encoding="utf-8"))

    rows = []
    debug_rows = []
    rule_count = 0
    rerank_count = 0
    for item in questions:
        selected = rank_article(
            item["question"],
            articles,
            bm25,
            rerank_min_delta=rerank_min_delta,
            use_db_heuristics=use_db_heuristics,
        )
        if str(selected.get("source", "")).startswith("rule:"):
            rule_count += 1
        if selected.get("source") == "augmented_bm25_same_law_rerank":
            rerank_count += 1
        selected = resolve_article(selected, lookup)
        doc_refs, article_refs = make_refs(selected, mapping)
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": answer_from_article(item["question"], selected, article_refs[0]),
                "relevant_docs": doc_refs,
                "relevant_articles": article_refs,
            }
        )
        debug_rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "law_id": canonical_law_id(selected.get("law_id", "")),
                "article_id": str(selected.get("article_id", "")),
                "source": selected.get("source", ""),
                "rank_score": selected.get("rank_score", ""),
                "law_rank_score": selected.get("law_rank_score", ""),
                "original_article_id": selected.get("original_article_id", ""),
                "original_article_score": selected.get("original_article_score", ""),
                "best_candidate_article_id": selected.get("best_candidate_article_id", ""),
                "best_candidate_score": selected.get("best_candidate_score", ""),
                "rerank_delta": selected.get("rerank_delta", ""),
                "article_title": selected.get("title", ""),
            }
        )

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_zip.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")
    if debug_output:
        debug_output.parent.mkdir(parents=True, exist_ok=True)
        with debug_output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()))
            writer.writeheader()
            writer.writerows(debug_rows)
    print(
        f"rows={len(rows)} rules={rule_count} reranked={rerank_count} "
        f"min_delta={rerank_min_delta} db_heuristics={use_db_heuristics} "
        f"db_codes={include_db_heuristic_codes} articles={len(articles)}"
    )
    print(f"wrote={output_zip}")
    if debug_output:
        print(f"debug={debug_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "submission_augmented_rerank_top1.zip"))
    parser.add_argument("--debug-output", default=str(DEBUG_PATH))
    parser.add_argument("--rerank-min-delta", type=float, default=RERANK_MIN_DELTA)
    parser.add_argument("--use-db-heuristics", action="store_true")
    parser.add_argument("--include-db-heuristic-codes", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_submission(
        Path(args.input),
        Path(args.output),
        force_rebuild=args.rebuild_index,
        debug_output=Path(args.debug_output) if args.debug_output else None,
        rerank_min_delta=args.rerank_min_delta,
        use_db_heuristics=args.use_db_heuristics,
        include_db_heuristic_codes=args.include_db_heuristic_codes,
    )


if __name__ == "__main__":
    main()
