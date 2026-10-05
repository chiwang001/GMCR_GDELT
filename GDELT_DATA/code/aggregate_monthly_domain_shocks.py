#!/usr/bin/env python3
"""Build leakage-safe monthly China-US GDELT shocks for the trade-conflict case."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DOMAINS = (
    "diplomacy",
    "economy",
    "law",
    "politics",
    "security",
    "society_humanitarian",
)
COUNTRY_LABELS = {"CHN": "CN", "USA": "US"}
SCOPES = ("all_bilateral", "trade_context", "direct_trade")
MONTH_PATTERN = re.compile(r"gdelt_events_(\d{6})_bilateral\.csv$")

# CAMEO root codes describe actions, not topics. Fractional memberships keep
# broad actions available to the GMCR preference model without pretending that
# every diplomatic event is exclusively diplomatic.
ROOT_DOMAIN_WEIGHTS: dict[int, dict[str, float]] = {
    1: {"diplomacy": 0.60, "politics": 0.40},
    2: {"diplomacy": 0.50, "politics": 0.20, "society_humanitarian": 0.30},
    3: {"diplomacy": 0.80, "politics": 0.20},
    4: {"diplomacy": 0.80, "politics": 0.20},
    5: {"diplomacy": 1.00},
    6: {"diplomacy": 0.20, "economy": 0.60, "society_humanitarian": 0.20},
    7: {"diplomacy": 0.10, "economy": 0.20, "society_humanitarian": 0.70},
    8: {"diplomacy": 0.50, "politics": 0.30, "security": 0.20},
    9: {"law": 0.70, "politics": 0.30},
    10: {"diplomacy": 0.40, "economy": 0.20, "politics": 0.40},
    11: {"diplomacy": 0.60, "politics": 0.40},
    12: {"diplomacy": 0.60, "politics": 0.40},
    13: {"diplomacy": 0.30, "security": 0.70},
    14: {"politics": 0.60, "security": 0.10, "society_humanitarian": 0.30},
    15: {"security": 1.00},
    16: {"diplomacy": 0.60, "economy": 0.25, "politics": 0.15},
    17: {"economy": 0.20, "law": 0.25, "politics": 0.25, "security": 0.30},
    18: {"security": 0.80, "society_humanitarian": 0.20},
    19: {"security": 0.90, "society_humanitarian": 0.10},
    20: {"security": 0.70, "society_humanitarian": 0.30},
}

# Explicit CAMEO economic actions. These form the high-precision direct-trade
# subset. The supplied files contain no article text/GKG themes, so generic
# consultations cannot honestly be labelled as trade negotiations.
DIRECT_TRADE_CODES = {
    "021",  # appeal for economic cooperation
    "031",  # intend to engage in economic cooperation
    "061",  # cooperate economically
    "071",  # provide economic aid
    "085",  # ease economic sanctions/boycott/embargo
    "101",  # demand economic cooperation
    "121",  # reject economic cooperation
    "162",  # reduce or stop material aid
    "163",  # impose embargo, boycott, or sanctions
    "171",  # seize or damage property
    "172",  # impose administrative sanctions/restrictions
}
ECONOMIC_ACTOR_TYPES = {"BUS", "MNC", "AGR"}
TRADE_CONTEXT_ROOTS = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17}

BASE_DOMAIN_OVERRIDES: dict[str, dict[str, float]] = {
    "021": {"diplomacy": 0.25, "economy": 0.65, "politics": 0.10},
    "023": {"diplomacy": 0.20, "economy": 0.10, "society_humanitarian": 0.70},
    "024": {"diplomacy": 0.20, "law": 0.20, "politics": 0.60},
    "025": {"diplomacy": 0.25, "law": 0.15, "politics": 0.60},
    "026": {"diplomacy": 0.15, "law": 0.45, "politics": 0.15, "society_humanitarian": 0.25},
    "027": {"diplomacy": 0.35, "politics": 0.10, "security": 0.55},
    "028": {"diplomacy": 0.75, "politics": 0.10, "security": 0.15},
    "031": {"diplomacy": 0.25, "economy": 0.65, "politics": 0.10},
    "033": {"diplomacy": 0.20, "economy": 0.15, "society_humanitarian": 0.65},
    "034": {"diplomacy": 0.20, "law": 0.20, "politics": 0.60},
    "035": {"diplomacy": 0.25, "law": 0.15, "politics": 0.60},
    "036": {"diplomacy": 0.15, "law": 0.45, "politics": 0.15, "society_humanitarian": 0.25},
    "037": {"diplomacy": 0.35, "politics": 0.10, "security": 0.55},
    "038": {"diplomacy": 0.75, "politics": 0.10, "security": 0.15},
    "061": {"diplomacy": 0.15, "economy": 0.80, "politics": 0.05},
    "062": {"diplomacy": 0.15, "politics": 0.05, "security": 0.80},
    "063": {"diplomacy": 0.15, "law": 0.75, "politics": 0.10},
    "064": {"diplomacy": 0.15, "politics": 0.10, "security": 0.75},
    "071": {"diplomacy": 0.10, "economy": 0.75, "society_humanitarian": 0.15},
    "072": {"diplomacy": 0.10, "economy": 0.10, "security": 0.80},
    "073": {"diplomacy": 0.10, "economy": 0.10, "society_humanitarian": 0.80},
    "074": {"diplomacy": 0.15, "law": 0.25, "politics": 0.10, "society_humanitarian": 0.50},
    "075": {"diplomacy": 0.15, "security": 0.25, "society_humanitarian": 0.60},
    "081": {"diplomacy": 0.20, "law": 0.20, "politics": 0.60},
    "085": {"diplomacy": 0.20, "economy": 0.70, "politics": 0.10},
    "086": {"diplomacy": 0.55, "law": 0.15, "politics": 0.20, "security": 0.10},
    "087": {"diplomacy": 0.20, "politics": 0.05, "security": 0.75},
    "090": {"law": 0.70, "politics": 0.20, "security": 0.10},
    "091": {"law": 0.75, "politics": 0.15, "security": 0.10},
    "092": {"law": 0.55, "politics": 0.15, "society_humanitarian": 0.30},
    "101": {"diplomacy": 0.25, "economy": 0.60, "politics": 0.15},
    "104": {"diplomacy": 0.15, "law": 0.20, "politics": 0.65},
    "105": {"diplomacy": 0.20, "law": 0.15, "politics": 0.65},
    "106": {"diplomacy": 0.10, "law": 0.45, "politics": 0.15, "society_humanitarian": 0.30},
    "115": {"diplomacy": 0.10, "law": 0.70, "politics": 0.20},
    "121": {"diplomacy": 0.25, "economy": 0.60, "politics": 0.15},
    "123": {"diplomacy": 0.15, "economy": 0.15, "politics": 0.10, "society_humanitarian": 0.60},
    "124": {"diplomacy": 0.15, "law": 0.20, "politics": 0.65},
    "125": {"diplomacy": 0.20, "law": 0.15, "politics": 0.65},
    "127": {"diplomacy": 0.25, "politics": 0.10, "security": 0.65},
    "128": {"diplomacy": 0.70, "politics": 0.10, "security": 0.20},
    "136": {"diplomacy": 0.10, "politics": 0.10, "security": 0.80},
    "162": {"diplomacy": 0.20, "economy": 0.60, "politics": 0.20},
    "163": {"diplomacy": 0.10, "economy": 0.75, "politics": 0.15},
    "166": {"diplomacy": 0.35, "law": 0.15, "politics": 0.35, "security": 0.15},
    "171": {"economy": 0.30, "law": 0.25, "politics": 0.20, "security": 0.25},
    "172": {"economy": 0.20, "law": 0.25, "politics": 0.30, "security": 0.25},
    "173": {"law": 0.35, "politics": 0.20, "security": 0.30, "society_humanitarian": 0.15},
    "174": {"diplomacy": 0.15, "law": 0.25, "politics": 0.20, "society_humanitarian": 0.40},
    "175": {"law": 0.20, "politics": 0.25, "security": 0.25, "society_humanitarian": 0.30},
}

REQUIRED_COLUMNS = {
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Actor1Code", "Actor1Name",
    "Actor1CountryCode", "Actor1Type1Code", "Actor2Code", "Actor2Name",
    "Actor2CountryCode", "Actor2Type1Code", "IsRootEvent", "EventCode",
    "EventBaseCode", "EventRootCode", "QuadClass", "GoldsteinScale",
    "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "ActionGeo_CountryCode", "ActionGeo_FeatureID", "SourceFileTimestamp",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=here.parent / "input")
    parser.add_argument("--output-dir", type=Path, default=here.parent / "output")
    parser.add_argument(
        "--demo-input",
        type=Path,
        default=here.parents[1] / "DynamicGMCR" / "input" / "monthly_country_domain_impact.csv",
    )
    parser.add_argument("--demo-scope", choices=SCOPES, default="trade_context")
    parser.add_argument("--stock-decay", type=float, default=0.70)
    return parser.parse_args()


def month_from_path(path: Path) -> int:
    match = MONTH_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Unexpected input filename: {path.name}")
    return int(match.group(1))


def code3(value: object) -> str | None:
    value = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(value) else f"{int(value):03d}"


def domain_weights(base: str | None, root: int) -> tuple[dict[str, float], str]:
    if base in BASE_DOMAIN_OVERRIDES:
        return BASE_DOMAIN_OVERRIDES[base], "base_override"
    return ROOT_DOMAIN_WEIGHTS[root], "root_fallback"


def load_events(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    qc = []
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        month = month_from_path(path)
        input_rows = len(frame)
        duplicate_ids = int(frame["GLOBALEVENTID"].duplicated().sum())
        frame = frame.drop_duplicates("GLOBALEVENTID").copy()
        bilateral = (
            frame["Actor1CountryCode"].isin(COUNTRY_LABELS)
            & frame["Actor2CountryCode"].isin(COUNTRY_LABELS)
            & frame["Actor1CountryCode"].ne(frame["Actor2CountryCode"])
        )
        root = pd.to_numeric(frame["IsRootEvent"], errors="coerce").eq(1)
        root_code = pd.to_numeric(frame["EventRootCode"], errors="coerce")
        mapped = root_code.isin(ROOT_DOMAIN_WEIGHTS)
        used = frame.loc[bilateral & root & mapped].copy()
        used["month"] = month  # information-arrival month, not referenced-event month
        used["source_file"] = path.name
        frames.append(used)
        qc.append({
            "month": month,
            "source_file": path.name,
            "input_rows": input_rows,
            "duplicate_event_ids": duplicate_ids,
            "non_bilateral_rows": int((~bilateral).sum()),
            "non_root_rows": int((bilateral & ~root).sum()),
            "unmapped_root_rows": int((bilateral & root & ~mapped).sum()),
            "used_rows": len(used),
            "data_status": "observed" if input_rows else "missing_input",
        })
    return pd.concat(frames, ignore_index=True), pd.DataFrame(qc)


def prepare_events(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["base_code"] = result["EventBaseCode"].map(code3)
    result["root_code"] = pd.to_numeric(result["EventRootCode"], errors="raise").astype(int)
    result["source_country"] = result["Actor1CountryCode"].map(COUNTRY_LABELS)
    result["target_country"] = result["Actor2CountryCode"].map(COUNTRY_LABELS)
    result["direction"] = result["source_country"] + "_to_" + result["target_country"]
    result["mapping_level"] = np.where(
        result["base_code"].isin(BASE_DOMAIN_OVERRIDES), "base_override", "root_fallback"
    )

    for column in ("GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone", "QuadClass"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["signed_intensity"] = result["GoldsteinScale"].fillna(0).clip(-10, 10) / 10
    result["tone_scaled"] = result["AvgTone"].fillna(0).clip(-10, 10) / 10
    result["is_material"] = result["QuadClass"].isin([2, 4])
    result["is_conflict"] = result["QuadClass"].isin([3, 4])

    # Cap media reach so one heavily syndicated story cannot determine a month.
    mentions = result["NumMentions"].fillna(0).clip(0, 50)
    sources = result["NumSources"].fillna(0).clip(0, 10)
    articles = result["NumArticles"].fillna(0).clip(0, 50)
    result["coverage_weight_raw"] = (
        np.sqrt(np.log1p(mentions) * np.log1p(articles))
        * (1 + 0.35 * np.log1p(sources))
    )
    medians = result.groupby(["month", "direction"])["coverage_weight_raw"].transform("median")
    result["coverage_weight"] = result["coverage_weight_raw"] / medians.clip(lower=1e-6)

    source_date = pd.to_datetime(
        result["SourceFileTimestamp"].astype("Int64").astype(str).str[:8],
        format="%Y%m%d", errors="coerce",
    )
    result["information_date"] = source_date
    result["week_from_month_end"] = (
        (source_date.dt.days_in_month - source_date.dt.day) // 7
    ).clip(0, 4).fillna(4).astype(int)
    event_date = pd.to_datetime(
        result["SQLDATE"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    age = (source_date - event_date).dt.days
    valid_age = age.where(age.between(0, 3650))
    result["referenced_event_age_days"] = valid_age
    result["is_retrospective"] = valid_age.gt(31).fillna(True)
    result["timeliness_weight"] = np.select(
        [valid_age.le(31), valid_age.le(180), valid_age.notna()], [1.0, 0.5, 0.2], default=0.5
    )
    result["event_weight"] = result["coverage_weight"] * result["timeliness_weight"]

    action_geo = result["ActionGeo_CountryCode"].fillna("")
    result["action_geo_role"] = np.select(
        [action_geo.eq("CH"), action_geo.eq("US")], ["CN", "US"], default="third_or_unknown"
    )
    result["is_third_country_action"] = ~action_geo.isin(["CH", "US"])

    actor_economic = (
        result["Actor1Type1Code"].isin(ECONOMIC_ACTOR_TYPES)
        | result["Actor2Type1Code"].isin(ECONOMIC_ACTOR_TYPES)
    )
    if "UrlTradeKeyword" in result:
        url_trade_keyword = (
            result["UrlTradeKeyword"].astype("string").str.lower().isin(["true", "1"])
        )
    else:
        url_trade_keyword = pd.Series(False, index=result.index)
    result["is_direct_trade"] = result["base_code"].isin(DIRECT_TRADE_CODES)
    result["is_economic_actor"] = actor_economic
    result["url_trade_keyword"] = url_trade_keyword
    result["is_trade_context"] = (
        result["is_direct_trade"]
        | (actor_economic & result["root_code"].isin(TRADE_CONTEXT_ROOTS))
        | (url_trade_keyword & result["root_code"].isin({1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 16, 17}))
        | result["base_code"].isin({"090", "091", "115", "166"})
    )
    result["trade_relevance"] = np.select(
        [result["is_direct_trade"], actor_economic & result["is_trade_context"], result["is_trade_context"]],
        [1.0, 0.70, 0.45], default=0.0,
    )
    if "TradeRelevance" in result:
        downloaded_relevance = pd.to_numeric(result["TradeRelevance"], errors="coerce").fillna(0.0)
        result["trade_relevance"] = np.maximum(result["trade_relevance"], downloaded_relevance)
    return result


def event_audit(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "GLOBALEVENTID", "month", "SQLDATE", "SourceFileTimestamp", "direction",
        "Actor1Code", "Actor1Name", "Actor1Type1Code", "Actor2Code", "Actor2Name",
        "Actor2Type1Code", "EventCode", "base_code", "root_code", "EventRootName",
        "QuadClass", "QuadClassName", "GoldsteinScale", "signed_intensity", "AvgTone",
        "NumMentions", "NumSources", "NumArticles", "coverage_weight", "timeliness_weight", "event_weight",
        "referenced_event_age_days", "is_retrospective", "action_geo_role",
        "is_third_country_action", "mapping_level", "is_economic_actor",
        "url_trade_keyword", "is_direct_trade", "is_trade_context",
        "trade_relevance", "SOURCEURL", "source_file",
    ]
    return frame.loc[:, [column for column in columns if column in frame.columns]]


def explode_domains(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in frame.itertuples(index=False):
        weights, _ = domain_weights(item.base_code, int(item.root_code))
        for domain, weight in weights.items():
            rows.append({
                "month": int(item.month),
                "target_country": item.target_country,
                "domain": domain,
                "domain_weight": float(weight),
                "signed_intensity": float(item.signed_intensity),
                "tone_scaled": float(item.tone_scaled),
                "event_weight": float(item.event_weight),
                "NumSources": float(item.NumSources),
                "is_material": bool(item.is_material),
                "is_conflict": bool(item.is_conflict),
                "is_retrospective": bool(item.is_retrospective),
                "is_third_country_action": bool(item.is_third_country_action),
                "is_direct_trade": bool(item.is_direct_trade),
                "is_trade_context": bool(item.is_trade_context),
                "trade_relevance": float(item.trade_relevance),
                "week_from_month_end": int(item.week_from_month_end),
            })
    return pd.DataFrame(rows)


def aggregate_scope(exploded: pd.DataFrame, months: list[int], scope: str) -> pd.DataFrame:
    if scope == "direct_trade":
        selected = exploded.loc[exploded["is_direct_trade"]].copy()
        selected["scope_weight"] = 1.0
    elif scope == "trade_context":
        selected = exploded.loc[exploded["is_trade_context"]].copy()
        selected["scope_weight"] = selected["trade_relevance"]
    else:
        selected = exploded.copy()
        selected["scope_weight"] = 1.0
    selected["mass"] = selected["domain_weight"] * selected["event_weight"] * selected["scope_weight"]
    selected["contribution"] = selected["mass"] * selected["signed_intensity"]
    selected["cooperation"] = selected["contribution"].clip(lower=0)
    selected["conflict"] = (-selected["contribution"]).clip(lower=0)
    selected["tone_mass"] = selected["mass"] * selected["tone_scaled"]
    selected["material_mass"] = selected["mass"] * selected["is_material"].astype(float)
    selected["third_geo_mass"] = selected["mass"] * selected["is_third_country_action"].astype(float)
    selected["retrospective_mass"] = selected["mass"] * selected["is_retrospective"].astype(float)
    selected["source_mass"] = selected["domain_weight"] * selected["scope_weight"] * np.log1p(selected["NumSources"].clip(0, 10))

    grouped = selected.groupby(["month", "target_country", "domain"], sort=True).agg(
        event_count=("domain_weight", "size"),
        effective_event_count=("domain_weight", "sum"),
        weighted_mass=("mass", "sum"),
        cooperation_mass=("cooperation", "sum"),
        conflict_mass=("conflict", "sum"),
        tone_mass=("tone_mass", "sum"),
        material_mass=("material_mass", "sum"),
        third_geo_mass=("third_geo_mass", "sum"),
        retrospective_mass=("retrospective_mass", "sum"),
        source_diversity=("source_mass", "sum"),
    ).reset_index()
    panel = pd.MultiIndex.from_product(
        [months, COUNTRY_LABELS.values(), DOMAINS], names=["month", "target_country", "domain"]
    ).to_frame(index=False)
    result = panel.merge(grouped, how="left", on=["month", "target_country", "domain"])
    count_columns = ["event_count", "effective_event_count", "weighted_mass", "cooperation_mass", "conflict_mass", "tone_mass", "material_mass", "third_geo_mass", "retrospective_mass", "source_diversity"]
    result[count_columns] = result[count_columns].fillna(0.0)
    denominator = result["weighted_mass"].replace(0, np.nan)
    result["mean_signed_intensity"] = (
        (result["cooperation_mass"] - result["conflict_mass"]) / denominator
    ).fillna(0.0)
    result["volume_adjusted_shock"] = result["mean_signed_intensity"] * np.log1p(result["effective_event_count"])
    result["mean_tone"] = (result["tone_mass"] / denominator).fillna(0.0)
    result["material_share"] = (result["material_mass"] / denominator).fillna(0.0)
    result["third_geo_share"] = (result["third_geo_mass"] / denominator).fillna(0.0)
    result["retrospective_share"] = (result["retrospective_mass"] / denominator).fillna(0.0)
    result["scope"] = scope
    result["data_status"] = "observed"
    weekly = selected.groupby(
        ["month", "target_country", "domain", "week_from_month_end"], sort=True
    ).agg(
        effective_event_count=("domain_weight", "sum"),
        weighted_mass=("mass", "sum"),
        signed_total=("contribution", "sum"),
    ).reset_index()
    weekly["weekly_shock"] = (
        weekly["signed_total"] / weekly["weighted_mass"].replace(0, np.nan)
    ).fillna(0.0) * np.log1p(weekly["effective_event_count"])
    weekly_panel = pd.MultiIndex.from_product(
        [months, COUNTRY_LABELS.values(), DOMAINS, range(5)],
        names=["month", "target_country", "domain", "week_from_month_end"],
    ).to_frame(index=False)
    weekly = weekly_panel.merge(
        weekly[["month", "target_country", "domain", "week_from_month_end", "weekly_shock"]],
        how="left",
        on=["month", "target_country", "domain", "week_from_month_end"],
    ).fillna({"weekly_shock": 0.0})
    weekly_wide = weekly.pivot(
        index=["month", "target_country", "domain"],
        columns="week_from_month_end",
        values="weekly_shock",
    ).reindex(columns=range(5), fill_value=0.0)
    chronological = weekly_wide[[4, 3, 2, 1, 0]].to_numpy(dtype=float)
    weekly_features = weekly_wide.reset_index()[["month", "target_country", "domain"]]
    weekly_features["weekly_variation"] = np.abs(np.diff(chronological, axis=1)).sum(axis=1)
    weekly_features["weekly_peak"] = np.abs(chronological).max(axis=1)
    weekly_features["weekly_recent"] = chronological[:, -1]
    result = result.merge(
        weekly_features, how="left", on=["month", "target_country", "domain"]
    )
    return result.drop(columns=["tone_mass", "material_mass", "third_geo_mass", "retrospective_mass"])


def add_temporal_features(frame: pd.DataFrame, decay: float) -> pd.DataFrame:
    result = frame.sort_values(["scope", "target_country", "domain", "month"]).copy()
    result["shock_z_past"] = 0.0
    result["net_surprise"] = 0.0
    result["cooperation_stock"] = 0.0
    result["conflict_stock"] = 0.0
    for _, indices in result.groupby(["scope", "target_country", "domain"], sort=False).groups.items():
        idx = list(indices)
        values = result.loc[idx, "volume_adjusted_shock"].to_numpy(float)
        z = np.zeros(len(values))
        surprise = np.zeros(len(values))
        coop_stock = np.zeros(len(values))
        conflict_stock = np.zeros(len(values))
        previous_level = 0.0
        for position, value in enumerate(values):
            history = values[max(0, position - 12):position]
            if len(history) >= 3:
                median = float(np.median(history))
                mad = float(np.median(np.abs(history - median)))
                scale = max(1.4826 * mad, 0.10)
                z[position] = float(np.clip((value - median) / scale, -6, 6))
            surprise[position] = value - previous_level
            previous_level = decay * previous_level + (1 - decay) * value
            if position:
                coop_stock[position] = decay * coop_stock[position - 1]
                conflict_stock[position] = decay * conflict_stock[position - 1]
            coop_stock[position] += result.loc[idx[position], "cooperation_mass"]
            conflict_stock[position] += result.loc[idx[position], "conflict_mass"]
        result.loc[idx, "shock_z_past"] = z
        result.loc[idx, "net_surprise"] = surprise
        result.loc[idx, "cooperation_stock"] = coop_stock
        result.loc[idx, "conflict_stock"] = conflict_stock
    return result.sort_values(["month", "scope", "target_country", "domain"]).reset_index(drop=True)


def mapping_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    codes = frame[["base_code", "root_code"]].drop_duplicates().sort_values(["root_code", "base_code"])
    for item in codes.itertuples(index=False):
        weights, level = domain_weights(item.base_code, int(item.root_code))
        for domain in DOMAINS:
            rows.append({
                "EventBaseCode": item.base_code,
                "EventRootCode": int(item.root_code),
                "mapping_level": level,
                "is_direct_trade": item.base_code in DIRECT_TRADE_CODES,
                "domain": domain,
                "weight": weights.get(domain, 0.0),
            })
    return pd.DataFrame(rows)


def export_demo_impacts(result: pd.DataFrame, path: Path, scope: str) -> None:
    demo = result.loc[result["scope"].eq(scope), [
        "month", "target_country", "domain", "data_status", "volume_adjusted_shock",
        "weekly_variation", "weekly_peak", "weekly_recent",
    ]].rename(columns={"target_country": "country", "volume_adjusted_shock": "preference_update_impact"})
    if demo["preference_update_impact"].isna().any() or len(demo) == 0:
        raise ValueError("Demo impact export is empty or contains missing values.")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    demo.sort_values(["month", "country", "domain"]).to_csv(path, index=False, encoding="utf-8-sig")


def write_report(output_dir: Path, events: pd.DataFrame, monthly: pd.DataFrame, qc: pd.DataFrame, demo_scope: str) -> None:
    direct = int(events["is_direct_trade"].sum())
    context = int(events["is_trade_context"].sum())
    retrospective = float(events["is_retrospective"].mean())
    lines = [
        "# China-US GDELT event-data quality report", "",
        "## Dataset", "",
        f"- Files: {len(qc)} monthly files, {int(qc['month'].min())}-{int(qc['month'].max())}.",
        f"- Valid bilateral root events: {len(events):,}.",
        f"- Direct-trade events: {direct:,} ({direct / max(len(events), 1):.1%}).",
        f"- Broader trade-context events: {context:,} ({context / max(len(events), 1):.1%}).",
        f"- Reports referring to events older than 31 days or with invalid dates: {retrospective:.1%}.",
        "- No source month is missing; zeros in sparse domain cells are observed zeros.", "",
        "## Validity decisions", "",
        "- Month is the SourceFileTimestamp/file month: it represents when information became available and prevents backdating reports.",
        "- SQLDATE is retained for timeliness weighting; retrospective references receive less weight.",
        "- Only CHN-to-USA and USA-to-CHN root events are retained; direction is preserved through target_country.",
        "- Media reach uses capped mentions, articles, and independent sources, normalized within month and direction.",
        "- Cooperation and conflict are retained separately; their net value is not the only model input.",
        "- Direct trade uses explicit CAMEO economic/coercive codes. Generic consultations are excluded because article text/GKG themes are unavailable.",
        "- Trade context adds bilateral economic actors and selected legal/regulatory actions with lower relevance weights.",
        "- shock_z_past uses only the previous 12 months; full-sample z-scores are intentionally not produced.", "",
        "## Recommended model inputs", "",
        f"- Demo-compatible preference input uses scope `{demo_scope}`, `volume_adjusted_shock`, and weekly variation/peak/recent features.",
        "- For richer models use cooperation_mass, conflict_mass, mean_tone, material_share, source_diversity, net_surprise, cooperation_stock, and conflict_stock.",
        "- Treat direct_trade as a high-precision sensitivity analysis and all_bilateral as the strategic-environment robustness analysis.", "",
        "## Limitation", "",
        "The supplied GDELT Events extract has no article URL, text, GKG themes, or Mentions table. Therefore generic diplomatic consultations cannot be verified as trade-related. The direct-trade subset prioritizes validity over recall; adding GKG themes would improve recall.",
    ]
    (output_dir / "data_quality_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "valid_bilateral_root_events": len(events),
        "direct_trade_events": direct,
        "trade_context_events": context,
        "retrospective_or_invalid_date_share": retrospective,
        "monthly_feature_rows": len(monthly),
        "demo_scope": demo_scope,
    }
    (output_dir / "processing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = sorted(args.input_dir.resolve().glob("gdelt_events_*_bilateral.csv"))
    if not paths:
        raise FileNotFoundError(f"No bilateral event files found in {args.input_dir.resolve()}")
    raw, qc = load_events(paths)
    events = prepare_events(raw)
    exploded = explode_domains(events)
    months = sorted(qc["month"].astype(int).tolist())
    monthly = pd.concat(
        [aggregate_scope(exploded, months, scope) for scope in SCOPES], ignore_index=True
    )
    monthly = add_temporal_features(monthly, float(np.clip(args.stock_decay, 0, 0.999)))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_audit(events).to_csv(output_dir / "bilateral_event_audit.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output_dir / "monthly_trade_conflict_features.csv", index=False, encoding="utf-8-sig")
    compatibility = monthly.loc[monthly["scope"].eq(args.demo_scope)].rename(columns={"target_country": "country"})
    compatibility.to_csv(output_dir / "monthly_country_domain_shocks.csv", index=False, encoding="utf-8-sig")
    wide_values = compatibility.pivot(index="month", columns=["country", "domain"], values="volume_adjusted_shock")
    wide_values.columns = [f"{country}_{domain}_shock" for country, domain in wide_values.columns]
    wide_values.reset_index().to_csv(output_dir / "monthly_country_domain_shocks_wide.csv", index=False, encoding="utf-8-sig")
    mapping_table(events).to_csv(output_dir / "cameo_base_domain_mapping.csv", index=False, encoding="utf-8-sig")
    qc.sort_values("month").to_csv(output_dir / "aggregation_qc.csv", index=False, encoding="utf-8-sig")
    export_demo_impacts(monthly, args.demo_input, args.demo_scope)
    write_report(output_dir, events, monthly, qc, args.demo_scope)
    print(
        f"Processed {len(events):,} bilateral events; "
        f"trade_context={int(events['is_trade_context'].sum()):,}, "
        f"direct_trade={int(events['is_direct_trade'].sum()):,}; output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
