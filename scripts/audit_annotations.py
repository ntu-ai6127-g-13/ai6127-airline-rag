"""Audit and prepare a three-annotator evaluation-set arbitration.

This script never overwrites the raw export.  It produces a wide, auditable
conflict table plus a proposed majority label.  Majority labels are a draft:
the assignment still requires a human/group decision for unresolved cases,
especially three-way aspect disagreements.

Example:
    python scripts/audit_annotations.py \
      --raw path/to/annotations_raw.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


USERS = ("sunsijie", "sunyuxin", "zhangyuxuan")
ASPECTS = (
    "seat", "staff", "food", "entertainment", "ground", "value", "overall", "other"
)
FIELDS = ("subjectivity", "polarity", "aspect")

# A conservative, guideline-based draft for the 52 three-way aspect cases.
# These are deliberately labelled as human-confirmation items in the output;
# they must not be presented as independent human annotation in the report.
THREE_WAY_DRAFTS = {
    "E0009": ("value", "R13: explicit value-for-money complaint"),
    "E0057": ("food", "R16: two same-polarity aspects; first evaluated aspect"),
    "E0088": ("overall", "R16: price/ground/entertainment are three aspects"),
    "E0179": ("value", "R16: value appears before comfort"),
    "E0224": ("other", "aircraft-level comfort without an explicit seat/service target"),
    "E0233": ("other", "aircraft-level comfort without an explicit seat target"),
    "E0235": ("seat", "explicitly about sitting/seat comfort"),
    "E0238": ("overall", "general experience evaluation"),
    "E0249": ("overall", "overall airline preference comes before staff detail"),
    "E0274": ("overall", "R16: three or more aspects; overall bucket"),
    "E0290": ("staff", "explicit interpersonal service behaviour"),
    "E0315": ("unclear", "missing referent for 'same levels'; human confirmation required"),
    "E0318": ("value", "cost/travel expense is the explicit evaluated aspect"),
    "E0327": ("overall", "overall airline preference precedes price condition"),
    "E0355": ("ground", "checkout counter is a ground-service interaction"),
    "E0358": ("overall", "comparative overall airline experience"),
    "E0369": ("overall", "cabin crowding/seat movement is a general experience"),
    "E0372": ("overall", "R16: food, value and entertainment are three aspects"),
    "E0377": ("seat", "the evaluated issue is people occupying the narrator's seats"),
    "E0394": ("value", "customer-service request; guideline maps this to value"),
    "E0455": ("ground", "hand-luggage/overhead-bin handling is ground/baggage"),
    "E0463": ("staff", "first evaluated service aspect; checked-bag fees are secondary"),
    "E0563": ("value", "comparison explicitly turns on the same price"),
    "E0569": ("value", "explicit wasted-money evaluation"),
    "E0619": ("other", "route/operator availability with no specified service aspect"),
    "E0630": ("staff", "negative evaluation is the person's indolence"),
    "E0639": ("value", "resolved customer-service matter"),
    "E0641": ("ground", "disembarkation/flight operation"),
    "E0644": ("overall", "rating/problem with no specific service aspect"),
    "E0659": ("value", "compensation is mapped to value by the guideline"),
    "E0663": ("unclear", "missing context for what caused the situation; human confirmation required"),
    "E0679": ("ground", "flight duration/waiting-time concern"),
    "E0686": ("overall", "R16: post-but clause contains aircraft/seat/food aspects"),
    "E0692": ("value", "generic customer-service quality, not cabin crew"),
    "E0760": ("overall", "general positive experience"),
    "E0802": ("value", "complaint/customer-service handling"),
    "E0803": ("value", "generic customer-service quality, not cabin crew"),
    "E0823": ("value", "customer-service representatives in a price/service comparison"),
    "E0841": ("seat", "explicit wrong-seat interaction"),
    "E0864": ("value", "coupon/compensation rather than food quality"),
    "E0866": ("ground", "flight-deck slot/operation concern"),
    "E0883": ("overall", "general airline-experience comparison"),
    "E0928": ("staff", "explicit person/service evaluation"),
    "E0931": ("value", "customer-service request with no cabin-crew target"),
    "E0937": ("staff", "explicit rude interpersonal behaviour"),
    "E0942": ("staff", "explicit crew behaviour/threat"),
    "E0951": ("ground", "airport arrival/timing"),
    "E0989": ("entertainment", "explicit IFE target"),
    "E1028": ("ground", "waiting-time/airport-operation consequence"),
    "E1041": ("overall", "R16: after-but overall negative outcome"),
    "E1049": ("food", "explicit dinner/meal absence"),
    "E1078": ("value", "lower-cost condition appears before general assessment"),
}


def majority(values: list[str | float]) -> tuple[str, int, int]:
    values = [str(v) for v in values if pd.notna(v) and str(v) != ""]
    if not values:
        return "", 0, 0
    counts = Counter(values)
    best = counts.most_common()
    # A deterministic tie marker is safer than silently selecting a label.
    if len(best) > 1 and best[0][1] == best[1][1]:
        return "", best[0][1], len(counts)
    return best[0][0], best[0][1], len(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--sentences",
        type=Path,
        default=Path("data/eval/to_label.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/eval/to_label_meta.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/annotation_audit"),
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    required = {"username", "sentence_id", *FIELDS, "note", "updated_at"}
    missing = required - set(raw.columns)
    if missing:
        raise SystemExit(f"raw export missing columns: {sorted(missing)}")
    if raw.duplicated(["username", "sentence_id"]).any():
        raise SystemExit("duplicate (username, sentence_id) records found")
    unknown = set(raw["username"].dropna()) - set(USERS)
    if unknown:
        raise SystemExit(f"unknown annotators: {sorted(unknown)}")

    sentences = pd.read_csv(args.sentences)
    if "id" not in sentences or "text" not in sentences:
        raise SystemExit("sentences file must contain id and text")
    meta = pd.read_csv(args.metadata)
    if "id" not in meta or "chunk_id" not in meta:
        raise SystemExit("metadata file must contain id and chunk_id")
    base = sentences.rename(columns={"id": "sentence_id"})[["sentence_id", "text"]]
    base = base.merge(meta.rename(columns={"id": "sentence_id"}), on="sentence_id", how="left")

    by_id = {sid: group for sid, group in raw.groupby("sentence_id", sort=False)}
    rows: list[dict[str, object]] = []
    for sid in base["sentence_id"]:
        group = by_id.get(sid, pd.DataFrame(columns=raw.columns))
        row: dict[str, object] = {"sentence_id": sid}
        base_row = base.loc[base["sentence_id"] == sid].iloc[0]
        for col in ("text", "chunk_id", "review_id", "airline", "cabin", "date"):
            if col in base_row:
                row[col] = base_row[col]
        for user in USERS:
            match = group[group["username"] == user]
            record = match.iloc[-1] if len(match) else None
            for field in FIELDS:
                row[f"{field}_{user}"] = record[field] if record is not None else ""
            row[f"note_{user}"] = record["note"] if record is not None else ""

        values = {field: [row[f"{field}_{u}"] for u in USERS] for field in FIELDS}
        row["annotator_count"] = sum(bool(row[f"subjectivity_{u}"]) for u in USERS)
        for field in FIELDS:
            nonempty = [str(v) for v in values[field] if pd.notna(v) and str(v)]
            row[f"{field}_unique"] = "|".join(sorted(set(nonempty)))
            row[f"{field}_conflict"] = len(set(nonempty)) > 1
            label, votes, nlabels = majority(values[field])
            row[f"proposed_{field}"] = label
            row[f"proposed_{field}_votes"] = votes
            row[f"{field}_label_count"] = nlabels

        aspect_values = [str(v) for v in values["aspect"] if pd.notna(v) and str(v)]
        row["aspect_three_way"] = len(set(aspect_values)) == 3
        row["unclear_votes"] = sum(v == "unclear" for v in values["subjectivity"])
        row["drop_from_gold_by_guideline"] = row["unclear_votes"] >= 2
        row["any_conflict"] = any(row[f"{field}_conflict"] for field in FIELDS)
        row["needs_human_arbitration"] = bool(
            row["any_conflict"] or row["drop_from_gold_by_guideline"]
        )
        rows.append(row)

    wide = pd.DataFrame(rows)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out / "annotations_wide.csv", index=False, encoding="utf-8-sig")
    wide[wide["any_conflict"] | wide["drop_from_gold_by_guideline"]].to_csv(
        out / "conflicts_all.csv", index=False, encoding="utf-8-sig"
    )
    aspect = wide[wide["aspect_conflict"]].copy()
    aspect.to_csv(out / "aspect_conflicts_383.csv", index=False, encoding="utf-8-sig")
    aspect[aspect["aspect_three_way"]].to_csv(
        out / "aspect_conflicts_three_way_52.csv", index=False, encoding="utf-8-sig"
    )
    # Safe automatic part: two-of-three majority only. Three-way cases remain
    # explicitly unresolved so an AI suggestion cannot be mistaken for a human
    # arbitration decision.
    proposed = aspect.copy()
    proposed["proposal_status"] = proposed.apply(
        lambda r: "needs_human_arbitration"
        if r["aspect_three_way"]
        else "majority_2_of_3",
        axis=1,
    )
    proposed["final_aspect"] = proposed["proposed_aspect"]
    proposed.loc[proposed["aspect_three_way"], "final_aspect"] = ""
    proposed["draft_aspect"] = proposed["proposed_aspect"]
    proposed["draft_basis"] = proposed.apply(
        lambda r: "majority_2_of_3"
        if not r["aspect_three_way"]
        else THREE_WAY_DRAFTS.get(str(r["sentence_id"]), ("", "missing draft"))[1],
        axis=1,
    )
    for sid, (draft, _) in THREE_WAY_DRAFTS.items():
        proposed.loc[proposed["sentence_id"] == sid, "draft_aspect"] = draft
    proposed["human_confirmation_required"] = proposed["aspect_three_way"]
    proposed["draft_action"] = proposed.apply(
        lambda r: "confirm_human_draft"
        if r["aspect_three_way"]
        else "adopt_majority_after_group_signoff",
        axis=1,
    )
    proposed.to_csv(out / "aspect_arbitration_proposals.csv", index=False, encoding="utf-8-sig")
    missing_drafts = set(aspect.loc[aspect["aspect_three_way"], "sentence_id"]) - set(THREE_WAY_DRAFTS)
    if missing_drafts:
        raise SystemExit(f"missing three-way draft decisions: {sorted(missing_drafts)}")

    # A non-final, reproducible candidate gold table.  It uses majority labels
    # for ordinary 2-of-3 cases and the explicit drafts above for three-way
    # aspect conflicts.  `gold_eligible` excludes the 15 sentences with >=2
    # unclear votes under the repository's R11 rule.
    candidate = wide.copy()
    candidate["candidate_subjectivity"] = candidate["proposed_subjectivity"]
    candidate["candidate_polarity"] = candidate["proposed_polarity"]
    candidate["candidate_aspect"] = candidate["proposed_aspect"]
    for sid, (draft, _) in THREE_WAY_DRAFTS.items():
        candidate.loc[candidate["sentence_id"] == sid, "candidate_aspect"] = draft
    candidate["candidate_label"] = candidate["candidate_subjectivity"]
    subjective = candidate["candidate_subjectivity"] == "subjective"
    candidate.loc[subjective, "candidate_label"] = candidate.loc[subjective, "candidate_polarity"]
    # A draft `unclear` aspect cannot be used as a normal aspect label.  Keep
    # those rows outside the candidate gold set until a human decides whether
    # the sentence is genuinely unresolvable or should be assigned overall/
    # other under the guideline.
    candidate["gold_eligible"] = (
        ~candidate["drop_from_gold_by_guideline"]
        & candidate["candidate_aspect"].ne("unclear")
    )
    candidate["candidate_status"] = "majority_or_guideline_draft"
    candidate.loc[candidate["drop_from_gold_by_guideline"], "candidate_status"] = "drop_pending_group_confirmation"
    candidate.loc[candidate["aspect_three_way"], "candidate_status"] = "three_way_draft_pending_human_confirmation"
    candidate.loc[
        candidate["candidate_aspect"].eq("unclear"), "candidate_status"
    ] = "unclear_aspect_pending_human_confirmation"
    candidate.to_csv(out / "gold_candidate_1100.csv", index=False, encoding="utf-8-sig")

    summary = {
        "raw_rows": int(len(raw)),
        "unique_sentences": int(raw["sentence_id"].nunique()),
        "annotators": {u: int((raw["username"] == u).sum()) for u in USERS},
        "all_three_annotated": int((wide["annotator_count"] == 3).sum()),
        "any_conflict": int(wide["any_conflict"].sum()),
        "aspect_conflicts": int(len(aspect)),
        "aspect_three_way": int(aspect["aspect_three_way"].sum()),
        "unclear_two_or_more": int(wide["drop_from_gold_by_guideline"].sum()),
        "aspect_values_seen": sorted(
            {v for v in raw["aspect"].dropna().astype(str) if v}
        ),
        "normal_aspects_expected": list(ASPECTS),
    }
    pd.Series(summary, dtype="object").to_json(out / "audit_summary.json", force_ascii=False, indent=2)
    print(summary)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
