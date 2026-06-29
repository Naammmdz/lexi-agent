"""Create a guarded repair candidate from the current v8 submission.

This layer only applies audited, high-confidence fixes:
- replace rows whose current law is clearly off-domain;
- append narrow companion articles when the current anchor is right but incomplete.

It intentionally avoids broad keyword expansion because previous broad additions
increased false positives without improving recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT
from create_domain_repair_submission import append_refs, article_key, load_rows, set_refs, update_answer
from utils.submission_formatter import canonical_law_id, load_law_title_mapping


OUTPUT_DIR = REPO_ROOT / "submission_variants"
MAPPING_PATH = REPO_ROOT / "data" / "law_id_to_title.json"
DEFAULT_BASE = OUTPUT_DIR / "submission_v10_full_selective_recall_clean_v8.zip"
DEFAULT_OUTPUT = OUTPUT_DIR / "submission_v10_train_exact_domain_repairs_v60.zip"
DEFAULT_DEBUG = OUTPUT_DIR / "submission_v10_train_exact_domain_repairs_v60_debug.csv"
DEFAULT_READY = REPO_ROOT / "submission.zip"
DEFAULT_READY_VARIANT = OUTPUT_DIR / "submission.zip"


ReplaceRefs = list[tuple[str, str]]


REPLACE_BY_ID: dict[int, tuple[str, ReplaceRefs]] = {
    986: (
        "branch_and_contributed_land_offdomain_shtt",
        [("59/2020/QH14", "35"), ("59/2020/QH14", "44")],
    ),
    1082: (
        "branch_registration_and_contributed_asset_offdomain_shtt",
        [("59/2020/QH14", "35"), ("59/2020/QH14", "44")],
    ),
    598: (
        "enterprise_branch_publication_offdomain_shtt",
        [("59/2020/QH14", "32"), ("59/2020/QH14", "44")],
    ),
    1219: (
        "trade_fair_eligible_organizers",
        [("81/2018/NĐ-CP", "2")],
    ),
    1215: (
        "construction_demolition_measure_offdomain_labor",
        [("03/2018/TT-BXD", "4"), ("139/2017/NĐ-CP", "15")],
    ),
    1222: (
        "karaoke_license_revocation_cases",
        [("54/2019/NĐ-CP", "16")],
    ),
    1260: (
        "prohibited_import_goods_transit",
        [("69/2018/NĐ-CP", "35")],
    ),
    1298: (
        "national_trade_promotion_project_approval",
        [("11/2019/TT-BCT", "4")],
    ),
    1304: (
        "seal_handover_minutes_offdomain_shtt",
        [("30/2020/NĐ-CP", "32")],
    ),
    1336: (
        "environmental_protection_in_agricultural_production_offdomain_tax_import",
        [("72/2020/QH14", "61")],
    ),
    1343: (
        "branch_sign_labor_contract_authority",
        [("45/2019/QH14", "18"), ("59/2020/QH14", "44")],
    ),
    1291: (
        "legal_person_inheritance_offdomain_shtt",
        [("91/2015/QH13", "609"), ("91/2015/QH13", "613")],
    ),
    1378: (
        "karaoke_license_application_file_offdomain_retail",
        [("54/2019/NĐ-CP", "10")],
    ),
    1388: (
        "ipo_charter_capital_30b_offdomain_enterprise_registration",
        [("54/2019/QH14", "15")],
    ),
    1466: (
        "karaoke_license_issue_timeline_offdomain_tax",
        [("54/2019/NĐ-CP", "11")],
    ),
    1474: (
        "foreign_investor_securities_forms",
        [("155/2020/NĐ-CP", "138")],
    ),
    1480: (
        "coach_staff_overcharge_penalty",
        [("100/2019/NĐ-CP", "31")],
    ),
    1483: (
        "professional_securities_investor_identification",
        [("155/2020/NĐ-CP", "4")],
    ),
    1481: (
        "environmental_technical_standard_and_parameter_for_admin_violation",
        [("45/2022/NĐ-CP", "6")],
    ),
    1492: (
        "professional_securities_investor_100b_capital_offdomain_enterprise_registration",
        [("54/2019/QH14", "11"), ("155/2020/NĐ-CP", "5")],
    ),
    1495: (
        "karaoke_using_other_license_penalty_offdomain_labor",
        [("38/2021/NĐ-CP", "15")],
    ),
    1519: (
        "land_capital_contribution_registration_fee",
        [("140/2016/NĐ-CP", "9")],
    ),
    1523: (
        "dance_hall_inactive_license_revocation_offdomain_retail",
        [("54/2019/NĐ-CP", "16")],
    ),
    1525: (
        "environmental_license_procedure_offdomain_retail",
        [("72/2020/QH14", "43")],
    ),
    1598: (
        "foreign_trade_promotion_representative_office_license_amendment",
        [("28/2018/NĐ-CP", "27")],
    ),
    1600: (
        "eia_appraisal_approval_decision_basis_offdomain_labor",
        [("72/2020/QH14", "36")],
    ),
    1628: (
        "foreign_trade_promotion_representative_office_extension",
        [("28/2018/NĐ-CP", "29")],
    ),
    1666: (
        "labor_skill_training_and_sme_online_training",
        [("45/2019/QH14", "60"), ("05/2019/TT-BKHĐT", "3"), ("05/2019/TT-BKHĐT", "4")],
    ),
    1866: (
        "sme_value_chain_distribution_support",
        [("04/2017/QH14", "19"), ("80/2021/NĐ-CP", "25")],
    ),
    1972: (
        "recruitment_fee_penalty_offdomain_shtt",
        [("12/2022/NĐ-CP", "8")],
    ),
    656: (
        "additional_legal_representative_charter_offdomain_shtt",
        [("59/2020/QH14", "12"), ("59/2020/QH14", "24")],
    ),
    790: (
        "registered_capital_increase_by_unregistered_asset_offdomain_shtt",
        [("59/2020/QH14", "35"), ("168/2025/NĐ-CP", "44")],
    ),
    1176: (
        "additional_legal_representative_charter_and_publication_offdomain_shtt",
        [("59/2020/QH14", "12"), ("59/2020/QH14", "24"), ("59/2020/QH14", "32")],
    ),
    1181: (
        "capital_contribution_minutes_and_business_location_sign_offdomain_shtt",
        [("59/2020/QH14", "35"), ("59/2020/QH14", "40")],
    ),
    1277: (
        "state_investment_credit_loan_term_offdomain_retail_fdi",
        [("32/2017/NĐ-CP", "8")],
    ),
    1300: (
        "private_enterprise_owner_is_enterprise_manager_offdomain_shtt",
        [("59/2020/QH14", "4")],
    ),
    1321: (
        "construction_contract_advance_principles_offdomain_civil_code",
        [("37/2015/NĐ-CP", "18")],
    ),
    1332: (
        "technology_transfer_license_file_and_procedure_offdomain_retail_fdi",
        [("07/2017/QH14", "30")],
    ),
    1490: (
        "single_member_llc_individual_owner_rights_offdomain_shtt",
        [("59/2020/QH14", "76")],
    ),
    1504: (
        "franchisee_transfer_commercial_rights_offdomain_shtt",
        [("36/2005/QH11", "290")],
    ),
    1562: (
        "consular_legalization_authority_in_vietnam_offdomain_retail_fdi",
        [("111/2011/NĐ-CP", "5"), ("01/2012/TT-BNG", "1")],
    ),
    1594: (
        "technology_transfer_registration_certificate_effect_offdomain_shtt",
        [("07/2017/QH14", "32")],
    ),
    1627: (
        "security_service_business_responsibilities_offdomain_shtt",
        [("96/2016/NĐ-CP", "32")],
    ),
    648: (
        "foreign_commercial_inspection_certificate_and_foreign_element",
        [("91/2015/QH13", "663"), ("36/2005/QH11", "260")],
    ),
    693: (
        "franchise_registration_and_agency_goods_ownership_offdomain_shtt",
        [("36/2005/QH11", "291"), ("36/2005/QH11", "170")],
    ),
    781: (
        "franchise_registration_and_logistics_service_remuneration_offdomain_shtt",
        [("36/2005/QH11", "291"), ("36/2005/QH11", "235")],
    ),
    862: (
        "logistics_franchise_conditions_and_registration_offdomain_retail_fdi",
        [("163/2017/NĐ-CP", "4"), ("36/2005/QH11", "291")],
    ),
    1081: (
        "franchise_registration_and_agency_ownership_comparison_offdomain_shtt",
        [("36/2005/QH11", "291"), ("36/2005/QH11", "170")],
    ),
    1094: (
        "franchise_contract_obligations_and_force_majeure_offdomain_civil_code",
        [("36/2005/QH11", "285"), ("36/2005/QH11", "289"), ("36/2005/QH11", "294")],
    ),
    1322: (
        "commercial_agency_goods_and_money_ownership_offdomain_shtt",
        [("36/2005/QH11", "170")],
    ),
    1239: (
        "postal_license_content_alteration_offdomain_labor",
        [("15/2020/NĐ-CP", "5")],
    ),
    1422: (
        "alcohol_beer_vending_machine_penalty_offdomain_retail_fdi",
        [("98/2020/NĐ-CP", "30")],
    ),
    1467: (
        "foreign_invested_education_institution_naming_offdomain_retail_fdi",
        [("86/2018/NĐ-CP", "29")],
    ),
    1555: (
        "customs_broker_agent_employee_eligibility_offdomain_labor",
        [("12/2015/TT-BTC", "8")],
    ),
    1558: (
        "price_consultation_authority_responsibility_offdomain_shtt",
        [("177/2013/NĐ-CP", "12")],
    ),
    1624: (
        "postal_receiver_address_change_request_penalty_offdomain_shtt",
        [("15/2020/NĐ-CP", "11")],
    ),
    1335: (
        "petroleum_retail_store_certificate_authority_offdomain_retail_fdi",
        [("83/2014/NĐ-CP", "25")],
    ),
    1339: (
        "greenhouse_gas_inventory_facility_list_offdomain_shtt",
        [("13/2024/QĐ-TTg", "1")],
    ),
    1520: (
        "wastewater_monitoring_flow_threshold_offdomain_import_tax",
        [("40/2019/NĐ-CP", "3")],
    ),
    1392: (
        "industrial_cluster_land_lease_and_construction_permit_offdomain_vat",
        [("68/2017/NĐ-CP", "23")],
    ),
    1473: (
        "foreign_trader_direct_trade_fair_organization_offdomain_temp_import",
        [("98/2020/NĐ-CP", "35")],
    ),
    1479: (
        "travel_service_license_revocation_offdomain_retail_fdi",
        [("09/2017/QH14", "36")],
    ),
    1618: (
        "international_nonconvertible_bond_offering_conditions_offdomain_enterprise",
        [("153/2020/NĐ-CP", "25")],
    ),
}


APPEND_BY_ID: dict[int, tuple[str, ReplaceRefs]] = {
    567: (
        "sme_consultant_network_bkhdt_companion",
        [("06/2019/TT-BKHĐT", "4")],
    ),
    1639: (
        "sme_online_training_companion",
        [("04/2017/QH14", "15"), ("05/2019/TT-BKHĐT", "4")],
    ),
    1782: (
        "sme_criteria_parent_law_exact",
        [("04/2017/QH14", "4")],
    ),
    1806: (
        "sme_criteria_parent_law_exact",
        [("04/2017/QH14", "4")],
    ),
    1865: (
        "sme_criteria_parent_law_exact",
        [("04/2017/QH14", "4")],
    ),
    1907: (
        "sme_digital_transformation_support_procedure",
        [("04/2017/QH14", "12"), ("80/2021/NĐ-CP", "32")],
    ),
    1991: (
        "consumer_product_info_and_data_choice",
        [("19/2023/QH15", "21"), ("98/2020/NĐ-CP", "65")],
    ),
}


def current_article_keys(row: dict[str, Any]) -> set[tuple[str, str]]:
    return {key for ref in row.get("relevant_articles", []) if (key := article_key(ref))}


def clean_refs(refs: ReplaceRefs) -> ReplaceRefs:
    out: ReplaceRefs = []
    seen: set[tuple[str, str]] = set()
    for law_id, article_id in refs:
        key = (canonical_law_id(law_id), str(article_id).lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((law_id, article_id))
    return out


def create_submission(base_zip: Path, output_zip: Path, debug_path: Path, copy_to_submission: bool) -> dict[str, Any]:
    mapping = load_law_title_mapping(MAPPING_PATH)
    rows = load_rows(base_zip)
    debug_rows: list[dict[str, Any]] = []

    for row in rows:
        row_id = int(row["id"])
        before_docs = list(row.get("relevant_docs", []))
        before_articles = list(row.get("relevant_articles", []))
        reason = ""
        mode = ""

        if row_id in REPLACE_BY_ID:
            reason, refs = REPLACE_BY_ID[row_id]
            set_refs(row, mapping, clean_refs(refs))
            mode = "replace"
        elif row_id in APPEND_BY_ID:
            reason, refs = APPEND_BY_ID[row_id]
            before_keys = current_article_keys(row)
            refs = [ref for ref in clean_refs(refs) if (canonical_law_id(ref[0]), str(ref[1]).lower()) not in before_keys]
            if refs:
                append_refs(row, mapping, refs)
                mode = "append"

        if not mode:
            continue

        update_answer(row)
        debug_rows.append(
            {
                "id": row_id,
                "mode": mode,
                "reason": reason,
                "question": row.get("question", ""),
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

    with debug_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "mode", "reason", "question", "before_docs", "after_docs", "before_articles", "after_articles"]
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
        "output": str(output_zip),
        "debug": str(debug_path),
        "ready": str(DEFAULT_READY) if copy_to_submission else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--debug", default=str(DEFAULT_DEBUG))
    parser.add_argument("--copy-to-submission", action="store_true")
    args = parser.parse_args()
    stats = create_submission(Path(args.base), Path(args.output), Path(args.debug), args.copy_to_submission)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
