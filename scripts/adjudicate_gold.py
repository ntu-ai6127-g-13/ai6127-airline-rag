"""Create an auditable AI-assisted adjudicated gold set.

The source export remains immutable.  The final table contains the original
three labels, the adjudicated label, the decision basis, and eligibility.  A
human team member must review the rows marked ``manual_rule`` before calling
this an official human gold set; the assignment requires manual annotation.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


USERS = ("sunsijie", "sunyuxin", "zhangyuxuan")
ASPECTS = {"seat", "staff", "food", "entertainment", "ground", "value", "overall", "other"}

# These seven subjectivity cases have no two-of-three majority.  The decisions
# follow the project's objective-vs-evaluative and R2/R3 rules.
SUBJ_OVERRIDES = {
    "E0189": ("objective", "three-way: request stated as a fact; no evaluation"),
    "E0605": ("objective", "three-way: asking to buy food is factual"),
    "E0613": ("objective", "three-way: check-in request is factual"),
    "E0673": ("objective", "three-way: gate-change action is factual"),
    "E0755": ("objective", "three-way: waiting and sitting are factual"),
    "E0841": ("subjective", "R2: wrong-seat situation is an evaluative service problem"),
    "E0989": ("subjective", "hope/request for refurbished IFE expresses preference"),
}

# Polarity cases with no majority or a clear R15/R16 direction.  Positive and
# negative are only assigned after the final subjectivity is known.
POL_OVERRIDES = {
    "E0085": ("positive", "no comparison = favourable value assessment"),
    "E0094": ("negative", "R16: failed entertainment outweighs compensation"),
    "E0145": ("negative", "switching to another airline is dissatisfaction"),
    "E0171": ("negative", "R15: limitation after but is the final sentiment"),
    "E0178": ("negative", "R16: delay is the negative service issue"),
    "E0212": ("positive", "dinner being rolled out is the favourable event"),
    "E0230": ("negative", "R16: insufficient luggage space is the negative issue"),
    "E0264": ("negative", "check-in location/agent description is negative"),
    "E0367": ("negative", "missing/fragmented queue is a service defect"),
    "E0439": ("positive", "different carrier produced a more enjoyable flight"),
    "E0516": ("negative", "R15: budget-airline comparison after but is negative"),
    "E0569": ("positive", "R14: not wasting money is favourable"),
    "E0581": ("negative", "R15: expectation after but is negative"),
    "E0708": ("positive", "better low-cholesterol choice is favourable"),
    "E0957": ("negative", "delay remains a negative outcome despite fault caveat"),
    "E1012": ("negative", "limitation after but is negative for long flights"),
    "E1073": ("positive", "R15: purchase option after but is favourable"),
    "E1098": ("positive", "accepting responsibility after but is not a complaint"),
}

# Guideline-based aspect decisions for every three-way aspect conflict.
ASPECT_OVERRIDES = {
    "E0009": ("value", "R13: explicit value-for-money complaint"),
    "E0057": ("food", "R16: same polarity, first evaluated aspect"),
    "E0088": ("overall", "R16: price/ground/entertainment are three aspects"),
    "E0179": ("value", "R16: value appears before comfort"),
    "E0224": ("other", "aircraft-level comfort without explicit seat target"),
    "E0233": ("other", "aircraft-level comfort without explicit seat target"),
    "E0235": ("seat", "explicit sitting/seat comfort"),
    "E0238": ("overall", "general experience evaluation"),
    "E0249": ("overall", "overall airline preference precedes staff detail"),
    "E0274": ("overall", "R16: three or more aspects"),
    "E0290": ("staff", "explicit interpersonal service behaviour"),
    "E0315": ("overall", "insufficient aspect detail; overall is the least-specific valid class"),
    "E0318": ("value", "cost/travel expense is the evaluated aspect"),
    "E0327": ("overall", "overall airline preference precedes price condition"),
    "E0355": ("ground", "checkout counter is ground service"),
    "E0358": ("overall", "comparative overall airline experience"),
    "E0369": ("overall", "cabin crowding is a general experience"),
    "E0372": ("overall", "R16: food, value and entertainment"),
    "E0377": ("seat", "the issue is people occupying the narrator's seats"),
    "E0394": ("value", "customer-service request maps to value in the guideline"),
    "E0455": ("ground", "hand luggage/overhead-bin handling"),
    "E0463": ("staff", "first evaluated service aspect; bag fee is secondary"),
    "E0563": ("value", "comparison turns on the same price"),
    "E0569": ("value", "explicit wasted-money evaluation"),
    "E0619": ("other", "route/operator availability without a service aspect"),
    "E0630": ("staff", "negative evaluation is the person's indolence"),
    "E0639": ("value", "resolved customer-service matter"),
    "E0641": ("ground", "disembarkation/flight operation"),
    "E0644": ("overall", "rating/problem without specific aspect"),
    "E0659": ("value", "compensation maps to value"),
    "E0663": ("overall", "general accusation without a specific service target"),
    "E0679": ("ground", "flight duration/waiting-time concern"),
    "E0686": ("overall", "R16: post-but clause contains aircraft/seat/food"),
    "E0692": ("value", "generic customer-service quality, not cabin crew"),
    "E0760": ("overall", "general positive experience"),
    "E0802": ("value", "complaint/customer-service handling"),
    "E0803": ("value", "generic customer-service quality, not cabin crew"),
    "E0823": ("value", "customer-service representatives in price/service comparison"),
    "E0841": ("seat", "explicit wrong-seat interaction"),
    "E0864": ("value", "coupon/compensation rather than food quality"),
    "E0866": ("ground", "flight-deck slot/operation concern"),
    "E0883": ("overall", "general airline-experience comparison"),
    "E0928": ("staff", "explicit person/service evaluation"),
    "E0931": ("value", "customer-service request without cabin-crew target"),
    "E0937": ("staff", "explicit rude interpersonal behaviour"),
    "E0942": ("staff", "explicit crew behaviour/threat"),
    "E0951": ("ground", "airport arrival/timing"),
    "E0989": ("entertainment", "explicit IFE target"),
    "E1028": ("ground", "waiting-time/airport-operation consequence"),
    "E1041": ("overall", "R15: after-but overall negative outcome"),
    "E1049": ("food", "explicit dinner/meal absence"),
    "E1078": ("value", "lower-cost condition appears before general assessment"),
}


def mode(values: list[object]) -> str:
    vals = [str(v) for v in values if pd.notna(v) and str(v) not in ("", "nan", "NA")]
    if not vals:
        return ""
    counts = Counter(vals)
    best = counts.most_common()
    if len(best) > 1 and best[0][1] == best[1][1]:
        return ""
    return best[0][0]


def pairwise_metrics(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(USERS, 2):
        x = frame[[f"{field}_{a}", f"{field}_{b}"]].copy()
        x = x.replace({"": pd.NA, "nan": pd.NA, "NA": pd.NA}).dropna()
        if x.empty:
            continue
        agreement = float((x.iloc[:, 0] == x.iloc[:, 1]).mean())
        kappa = float(cohen_kappa_score(x.iloc[:, 0], x.iloc[:, 1]))
        rows.append({"field": field, "annotator_a": a, "annotator_b": b,
                     "n": len(x), "agreement": agreement, "cohen_kappa": kappa})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--sentences", type=Path, default=Path("data/eval/to_label.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/eval/to_label_meta.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/final_gold"))
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    sentences = pd.read_csv(args.sentences).rename(columns={"id": "sentence_id"})
    meta = pd.read_csv(args.metadata).rename(columns={"id": "sentence_id"})
    base = sentences[["sentence_id", "text"]].merge(meta, on="sentence_id", how="left")
    rows = []
    for _, base_row in base.iterrows():
        sid = base_row["sentence_id"]
        group = raw[raw["sentence_id"] == sid]
        row = {k: base_row[k] for k in base_row.index if k != "sentence_id"}
        row["sentence_id"] = sid
        for u in USERS:
            g = group[group["username"] == u]
            if g.empty:
                raise SystemExit(f"missing annotation: {u}/{sid}")
            r = g.iloc[-1]
            for f in ("subjectivity", "polarity", "aspect", "note"):
                row[f"{f}_{u}"] = r[f]

        subj_values = [row[f"subjectivity_{u}"] for u in USERS]
        pol_values = [row[f"polarity_{u}"] for u in USERS]
        aspect_values = [row[f"aspect_{u}"] for u in USERS]
        unclear_votes = sum(v == "unclear" for v in subj_values)
        row["unclear_votes"] = unclear_votes
        row["drop_reason"] = "at_least_two_unclear" if unclear_votes >= 2 else ""

        if sid in SUBJ_OVERRIDES:
            final_subj, subj_basis = SUBJ_OVERRIDES[sid]
            subj_status = "manual_rule"
        else:
            final_subj = mode(subj_values)
            subj_basis = "majority_2_of_3" if len(set(subj_values)) > 1 else "unanimous"
            subj_status = "majority" if subj_basis != "unanimous" else "unanimous"
        if not final_subj:
            raise SystemExit(f"no subjectivity decision for {sid}")

        if final_subj != "subjective":
            final_pol = "NA"
            pol_basis = "subjectivity_is_not_subjective"
        elif sid in POL_OVERRIDES:
            final_pol, pol_basis = POL_OVERRIDES[sid]
        else:
            final_pol = mode(pol_values)
            if not final_pol:
                # This is only a fallback for an unusual missing/three-way
                # case; every such row is included in the audit log.
                nonempty = [str(v) for v in pol_values if pd.notna(v) and str(v) not in ("", "NA", "nan")]
                final_pol = nonempty[0] if nonempty else "NA"
                pol_basis = "manual_fallback_from_available_label"
            else:
                pol_basis = "majority_2_of_3" if len(set(str(v) for v in pol_values)) > 1 else "unanimous"

        if sid in ASPECT_OVERRIDES:
            final_aspect, aspect_basis = ASPECT_OVERRIDES[sid]
            aspect_status = "manual_rule"
        else:
            final_aspect = mode(aspect_values)
            aspect_basis = "majority_2_of_3" if len(set(aspect_values)) > 1 else "unanimous"
            aspect_status = "majority" if aspect_basis != "unanimous" else "unanimous"
        if final_aspect not in ASPECTS and not (unclear_votes >= 2 and final_aspect == "unclear"):
            raise SystemExit(f"invalid final aspect {final_aspect!r} for {sid}")

        row.update({
            "final_subjectivity": final_subj,
            "final_polarity": final_pol,
            "final_aspect": final_aspect,
            "final_label": final_subj if final_subj != "subjective" else final_pol,
            "subjectivity_basis": subj_basis,
            "polarity_basis": pol_basis,
            "aspect_basis": aspect_basis,
            "decision_status": "drop" if unclear_votes >= 2 else (
                "manual_rule" if "manual_rule" in (subj_status, aspect_status) or sid in POL_OVERRIDES else (
                    "majority" if any(len(set(str(v) for v in vals)) > 1 for vals in (subj_values, pol_values, aspect_values)) else "unanimous"
                )
            ),
            "gold_eligible": unclear_votes < 2,
        })
        rows.append(row)

    all_rows = pd.DataFrame(rows)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out / "adjudication_log_all_1100.csv", index=False, encoding="utf-8-sig")
    all_rows[~all_rows.gold_eligible].to_csv(out / "excluded_from_gold.csv", index=False, encoding="utf-8-sig")

    eligible = all_rows[all_rows.gold_eligible].copy()
    eligible["label"] = eligible["final_label"]
    final_columns = [
        "text", "label", "sentence_id", "chunk_id", "review_id",
        "subjectivity", "polarity", "aspect", "airline", "cabin", "date",
        "final_subjectivity", "final_polarity", "final_aspect",
        "decision_status", "subjectivity_basis", "polarity_basis", "aspect_basis",
        "unclear_votes",
        "subjectivity_sunsijie", "polarity_sunsijie", "aspect_sunsijie",
        "subjectivity_sunyuxin", "polarity_sunyuxin", "aspect_sunyuxin",
        "subjectivity_zhangyuxuan", "polarity_zhangyuxuan", "aspect_zhangyuxuan",
    ]
    final_columns = [c for c in final_columns if c in eligible.columns]
    eligible[final_columns].to_csv(out / "gold_final.csv", index=False, encoding="utf-8-sig")
    eligible[final_columns].to_excel(out / "gold_final.xlsx", index=False)

    # Build a simple wide raw table for all three fields and calculate metrics
    # on the original pre-adjudication labels, both before and after exclusions.
    wide = base[["sentence_id"]].copy()
    for field in ("subjectivity", "polarity", "aspect"):
        p = raw.pivot(index="sentence_id", columns="username", values=field).reset_index()
        p = p.rename(columns={u: f"{field}_{u}" for u in USERS})
        wide = wide.merge(p, on="sentence_id", how="left")
    metric_frames = []
    for field in ("subjectivity", "polarity", "aspect"):
        metric_frames.append(pairwise_metrics(wide, field))
        eligible_ids = set(eligible["sentence_id"])
        metric_frames.append(pairwise_metrics(wide[wide.sentence_id.isin(eligible_ids)], field).assign(scope="eligible_1085"))
    metric_table = pd.concat(metric_frames, ignore_index=True)
    metric_table.to_csv(out / "agreement_pairwise.csv", index=False)

    triple_rates = {}
    for field in ("subjectivity", "aspect"):
        p = wide[[f"{field}_{u}" for u in USERS]]
        triple_rates[field] = float((p.nunique(axis=1) == 1).mean())
    polarity_wide = wide[[f"polarity_{u}" for u in USERS]].fillna("NA")
    triple_rates["polarity_with_NA"] = float((polarity_wide.nunique(axis=1) == 1).mean())
    full_triple = []
    for _, r in wide.iterrows():
        full_triple.append(
            len({
                tuple(r[f"{field}_{u}"] if pd.notna(r[f"{field}_{u}"]) else "NA" for field in ("subjectivity", "polarity", "aspect"))
                for u in USERS
            }) == 1
        )
    pairwise_summary = {}
    for field in ("subjectivity", "polarity", "aspect"):
        q = metric_table[(metric_table.field == field) & metric_table.scope.isna()]
        pairwise_summary[field] = {
            "mean_pairwise_agreement": float(q.agreement.mean()),
            "mean_pairwise_cohen_kappa": float(q.cohen_kappa.mean()),
            "pairs": q.to_dict("records"),
        }
    agreement_summary = {
        "raw_sentences": int(len(wide)),
        "eligible_sentences": int(len(eligible)),
        "pairwise": pairwise_summary,
        "three_annotator_exact_agreement": triple_rates,
        "full_three_field_exact_agreement": float(sum(full_triple) / len(full_triple)),
        "full_three_field_exact_count": int(sum(full_triple)),
        "note": "Polarity pairwise metrics in agreement_pairwise.csv condition on both labels being non-NA; polarity_with_NA treats NA as an explicit category.",
    }
    pd.Series(agreement_summary, dtype="object").to_json(
        out / "agreement_summary.json", force_ascii=False, indent=2
    )

    summary = {
        "raw_sentences": int(len(all_rows)),
        "raw_records": int(len(raw)),
        "eligible_gold_sentences": int(len(eligible)),
        "excluded_sentences": int((~all_rows.gold_eligible).sum()),
        "excluded_reason_counts": all_rows.loc[~all_rows.gold_eligible, "drop_reason"].value_counts().to_dict(),
        "decision_status_counts": all_rows.decision_status.value_counts().to_dict(),
        "aspect_classes": sorted(ASPECTS),
        "ai_assisted_adjudication": True,
        "human_confirmation_required": "Review the adjudication_log_all_1100.csv rows with decision_status=manual_rule before submission.",
    }
    pd.Series(summary, dtype="object").to_json(out / "gold_summary.json", force_ascii=False, indent=2)
    print(summary)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
