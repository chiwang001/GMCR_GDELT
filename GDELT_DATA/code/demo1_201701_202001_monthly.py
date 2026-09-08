#!/usr/bin/env python3
"""Download and filter monthly GDELT 2.0 events for the China-US trade conflict.

The script streams 15-minute GDELT Event exports, immediately deletes each raw
zip, and appends selected events to monthly CSV files. The default selection is
the broad, auditable ``trade_context`` scope. Use ``--scope direct_trade`` for
a high-precision CAMEO-only subset or ``--scope all_bilateral`` for a broad
strategic-environment robustness dataset.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

import pandas as pd

from aggregate_monthly_domain_shocks import (
    DIRECT_TRADE_CODES,
    ECONOMIC_ACTOR_TYPES,
    TRADE_CONTEXT_ROOTS,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "input"
DEFAULT_RAW_DIR = PROJECT_DIR / "results" / "raw_tmp"
DEFAULT_SUMMARY = PROJECT_DIR / "results" / "gdelt_trade_download_summary.csv"
GDELT_V2_BASE_URL = "https://data.gdeltproject.org/gdeltv2"
COUNTRIES = {"USA", "CHN"}

# Actions that provide relevant regulatory/legal context even when GDELT did
# not identify a business actor. They are lower-confidence than direct codes.
LEGAL_REGULATORY_CODES = {"090", "091", "115", "166"}
URL_CONTEXT_ROOTS = {1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 16, 17}
TRADE_URL_TERMS = (
    "trade war", "trade-war", "trade dispute", "trade talks", "trade deal",
    "tariff", "customs", "import", "export", "commerce", "economic sanction",
    "embargo", "boycott", "wto", "world trade organization", "section 301",
    "intellectual property", "technology transfer", "market access",
    "phase one", "phase-one", "huawei", "zte", "soybean", "rare earth",
)
TRADE_URL_PATTERN = re.compile(
    "|".join(re.escape(term).replace(r"\\ ", "[-_+%20 ]+") for term in TRADE_URL_TERMS),
    flags=re.IGNORECASE,
)

GDELT_EVENT_COLUMNS = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat",
    "Actor1Geo_Long", "Actor1Geo_FeatureID", "Actor2Geo_Type",
    "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code",
    "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long",
    "Actor2Geo_FeatureID", "ActionGeo_Type", "ActionGeo_FullName",
    "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_ADM2Code",
    "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID", "DATEADDED",
    "SOURCEURL",
]

OUTPUT_COLUMNS = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear",
    *GDELT_EVENT_COLUMNS[5:58], "DATEADDED", "SOURCEURL",
]
READ_COLUMNS = list(dict.fromkeys(OUTPUT_COLUMNS))

INTEGER_COLUMNS = {
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "IsRootEvent", "QuadClass",
    "NumMentions", "NumSources", "NumArticles", "Actor1Geo_Type",
    "Actor2Geo_Type", "ActionGeo_Type", "DATEADDED",
}
FLOAT_COLUMNS = {
    "GoldsteinScale", "AvgTone", "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor2Geo_Lat", "Actor2Geo_Long", "ActionGeo_Lat", "ActionGeo_Long",
}

CAMEO_ROOT_MAP = {
    "01": "Make public statement", "02": "Appeal",
    "03": "Express intent to cooperate", "04": "Consult",
    "05": "Engage in diplomatic cooperation",
    "06": "Engage in material cooperation", "07": "Provide aid",
    "08": "Yield", "09": "Investigate", "10": "Demand",
    "11": "Disapprove", "12": "Reject", "13": "Threaten",
    "14": "Protest", "15": "Exhibit force posture",
    "16": "Reduce relations", "17": "Coerce", "18": "Assault",
    "19": "Fight", "20": "Use unconventional mass violence",
}
QUAD_CLASS_MAP = {
    1: "Verbal cooperation", 2: "Material cooperation",
    3: "Verbal conflict", 4: "Material conflict",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=int, default=20170101)
    parser.add_argument("--end-date", type=int, default=20200131)
    parser.add_argument(
        "--scope", choices=("trade_context", "direct_trade", "all_bilateral"),
        default="trade_context",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--overwrite-existing-months", action="store_true",
        help="Process months whose output CSV already contains events.",
    )
    return parser.parse_args()


def parse_date(value: int) -> datetime:
    return datetime.strptime(str(value), "%Y%m%d")


def iter_timestamps(start_date: int, end_date: int):
    current = parse_date(start_date)
    end = parse_date(end_date) + timedelta(days=1)
    while current < end:
        yield current.strftime("%Y%m%d%H%M%S")
        current += timedelta(minutes=15)


def iter_months(start_date: int, end_date: int):
    current = parse_date(start_date).replace(day=1)
    end = parse_date(end_date).replace(day=1)
    while current <= end:
        yield current.strftime("%Y%m")
        current = (
            current.replace(year=current.year + 1, month=1)
            if current.month == 12
            else current.replace(month=current.month + 1)
        )


def code3(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.astype("string").str.zfill(3)


def apply_types(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in INTEGER_COLUMNS & set(result.columns):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    for column in FLOAT_COLUMNS & set(result.columns):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Float64")
    for column in set(result.columns) - INTEGER_COLUMNS - FLOAT_COLUMNS:
        result[column] = result[column].astype("string")
    return result


def read_events(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if not names:
            return pd.DataFrame(columns=READ_COLUMNS)
        with archive.open(names[0]) as source:
            frame = pd.read_csv(
                source, sep="\t", header=None, names=GDELT_EVENT_COLUMNS,
                usecols=READ_COLUMNS, dtype="string", low_memory=False,
            )
    return apply_types(frame)


def classify_trade_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Add auditable trade relevance flags without using future information."""
    result = frame.copy()
    base = code3(result["EventBaseCode"])
    root = pd.to_numeric(result["EventRootCode"], errors="coerce").astype("Int64")
    actor_economic = (
        result["Actor1Type1Code"].isin(ECONOMIC_ACTOR_TYPES)
        | result["Actor2Type1Code"].isin(ECONOMIC_ACTOR_TYPES)
    )
    url_keyword = result["SOURCEURL"].fillna("").str.contains(TRADE_URL_PATTERN, na=False)
    direct = base.isin(DIRECT_TRADE_CODES)
    economic_actor_context = actor_economic & root.isin(TRADE_CONTEXT_ROOTS)
    url_context = url_keyword & root.isin(URL_CONTEXT_ROOTS)
    regulatory_context = base.isin(LEGAL_REGULATORY_CODES)
    context = direct | economic_actor_context | url_context | regulatory_context

    result["UrlTradeKeyword"] = url_keyword
    result["IsEconomicActorEvent"] = actor_economic
    result["IsDirectTrade"] = direct
    result["IsTradeContext"] = context
    result["TradeRelevance"] = 0.0
    result.loc[regulatory_context, "TradeRelevance"] = 0.45
    result.loc[url_context, "TradeRelevance"] = 0.60
    result.loc[economic_actor_context, "TradeRelevance"] = 0.70
    result.loc[direct, "TradeRelevance"] = 1.00
    reasons = pd.Series("", index=result.index, dtype="string")
    reason_flags = (
        (direct, "direct_cameo"),
        (economic_actor_context, "economic_actor"),
        (url_context, "url_keyword"),
        (regulatory_context, "legal_regulatory"),
    )
    for flag, label in reason_flags:
        reasons = reasons.mask(flag & reasons.eq(""), label)
        reasons = reasons.mask(flag & reasons.ne("") & ~reasons.str.contains(label), reasons + "+" + label)
    result["TradeSelectionReason"] = reasons
    return result


def filter_events(frame: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, dict[str, int]]:
    if frame.empty:
        return frame, {"root_bilateral_rows": 0, "direct_trade_rows": 0, "trade_context_rows": 0}
    bilateral = (
        frame["IsRootEvent"].eq(1).fillna(False)
        & frame["Actor1CountryCode"].isin(COUNTRIES)
        & frame["Actor2CountryCode"].isin(COUNTRIES)
        & frame["Actor1CountryCode"].ne(frame["Actor2CountryCode"])
    )
    candidates = classify_trade_events(frame.loc[bilateral].copy())
    counts = {
        "root_bilateral_rows": len(candidates),
        "direct_trade_rows": int(candidates["IsDirectTrade"].sum()),
        "trade_context_rows": int(candidates["IsTradeContext"].sum()),
    }
    if scope == "direct_trade":
        candidates = candidates.loc[candidates["IsDirectTrade"]]
    elif scope == "trade_context":
        candidates = candidates.loc[candidates["IsTradeContext"]]
    return candidates.copy(), counts


def build_output(frame: pd.DataFrame, timestamp: str) -> pd.DataFrame:
    columns = [column for column in OUTPUT_COLUMNS if column in frame.columns]
    result = frame.loc[:, columns].copy()
    for derived in (
        "UrlTradeKeyword", "IsEconomicActorEvent", "IsDirectTrade",
        "IsTradeContext", "TradeRelevance", "TradeSelectionReason",
    ):
        result[derived] = frame[derived].values
    result["SourceFileTimestamp"] = int(timestamp)
    result["EventRootName"] = code3(result["EventRootCode"]).str[:2].map(CAMEO_ROOT_MAP)
    result["QuadClassName"] = result["QuadClass"].map(QUAD_CLASS_MAP)
    return result


def download(timestamp: str, raw_dir: Path, timeout: int) -> Path | None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{timestamp}.export.CSV.zip"
    request = Request(
        f"{GDELT_V2_BASE_URL}/{timestamp}.export.CSV.zip",
        headers={"User-Agent": "china-us-trade-gdelt-filter/2.0"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            path.write_bytes(response.read())
        return path
    except HTTPError as exc:
        if exc.code != 404:
            print(f"HTTP error for {timestamp}: {exc}")
    except (URLError, TimeoutError) as exc:
        print(f"Network error for {timestamp}: {exc}")
    return None


def append_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path, mode="a", header=not path.exists() or path.stat().st_size == 0,
        index=False, encoding="utf-8-sig",
    )


def monthly_output(output_dir: Path, timestamp: str) -> Path:
    # Keep the historical filename expected by the downstream aggregator.
    return output_dir / f"gdelt_events_{timestamp[:6]}_bilateral.csv"


def reset_outputs(args: argparse.Namespace) -> None:
    if not args.reset:
        return
    for month in iter_months(args.start_date, args.end_date):
        path = args.output_dir / f"gdelt_events_{month}_bilateral.csv"
        if path.exists():
            path.unlink()
    if args.summary.exists():
        args.summary.unlink()


def completed_months(args: argparse.Namespace) -> set[str]:
    if args.overwrite_existing_months:
        return set()
    completed = set()
    for month in iter_months(args.start_date, args.end_date):
        path = args.output_dir / f"gdelt_events_{month}_bilateral.csv"
        if path.exists() and path.stat().st_size > 100:
            completed.add(month)
    return completed


def process_timestamp(timestamp: str, args: argparse.Namespace) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp": timestamp, "scope": args.scope, "raw_rows": 0,
        "root_bilateral_rows": 0, "direct_trade_rows": 0,
        "trade_context_rows": 0, "matched_rows": 0, "status": "download_failed",
    }
    raw_path = download(timestamp, args.raw_dir, args.timeout)
    if raw_path is None:
        append_csv(pd.DataFrame([row]), args.summary)
        return row
    try:
        events = read_events(raw_path)
        selected, counts = filter_events(events, args.scope)
        output = build_output(selected, timestamp) if not selected.empty else selected
        append_csv(output, monthly_output(args.output_dir, timestamp))
        row.update(counts)
        row["raw_rows"] = len(events)
        row["matched_rows"] = len(selected)
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = f"error:{type(exc).__name__}:{exc}"
    finally:
        raw_path.unlink(missing_ok=True)
    append_csv(pd.DataFrame([row]), args.summary)
    return row


def main() -> int:
    args = parse_args()
    if parse_date(args.end_date) < parse_date(args.start_date):
        raise ValueError("--end-date must not precede --start-date")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    reset_outputs(args)
    skip_months = completed_months(args)
    timestamps = (
        timestamp for timestamp in iter_timestamps(args.start_date, args.end_date)
        if timestamp[:6] not in skip_months
    )
    total = 0
    processed = 0
    for timestamp in timestamps:
        if args.max_files is not None and processed >= args.max_files:
            break
        processed += 1
        row = process_timestamp(timestamp, args)
        total += int(row["matched_rows"])
        if processed == 1 or processed % 96 == 0:
            print(
                f"processed={processed:,} timestamp={timestamp} "
                f"matched={int(row['matched_rows']):,} cumulative={total:,} "
                f"status={row['status']}"
            )
    print(
        f"Completed scope={args.scope}; files={processed:,}; matched={total:,}; "
        f"output={args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
