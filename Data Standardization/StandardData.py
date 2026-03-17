from pathlib import Path
import re
import unicodedata
import warnings
from zipfile import BadZipFile
from zipfile import ZipFile

import pandas as pd


warnings.filterwarnings("ignore")

# Build absolute paths from script location so cwd does not matter.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_DIR = PROJECT_ROOT / "BaseData"
DOE_SPLIT_DIR = BASE_DIR / "DoE_split_by_location"
AOD_FILE = BASE_DIR / "AOD-14-21-daywise.csv"
OUTPUT_DIR = PROJECT_ROOT / "cleaned data"
OUTPUT_FILE = OUTPUT_DIR / "matched_AOD_PM25_2014_2021.csv"

START_DATE = pd.Timestamp("2014-01-01")
END_DATE = pd.Timestamp("2021-12-31")


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_inputs() -> None:
    missing = [
        str(path)
        for path in [AOD_FILE, DOE_SPLIT_DIR]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required input path(s):\n" + "\n".join(missing))


def get_aod_station_columns(aod_df: pd.DataFrame) -> list[str]:
    excluded = {"time", "date", "month", "year"}
    return [col for col in aod_df.columns if normalize_text(col) not in excluded]


def load_aod_long() -> pd.DataFrame:
    print("Loading AOD dataset...")
    aod_df = pd.read_csv(AOD_FILE)

    date_col = next((col for col in aod_df.columns if normalize_text(col) in {"time", "date"}), None)
    month_col = next((col for col in aod_df.columns if normalize_text(col) == "month"), None)
    year_col = next((col for col in aod_df.columns if normalize_text(col) == "year"), None)

    if date_col is None or month_col is None or year_col is None:
        raise ValueError("AOD CSV must contain Time/Date, month, and year columns.")

    station_cols = get_aod_station_columns(aod_df)
    if not station_cols:
        raise ValueError("No station columns found in AOD CSV.")

    long_df = aod_df.melt(
        id_vars=[date_col, month_col, year_col],
        value_vars=station_cols,
        var_name="Monitoring_Station",
        value_name="AOD",
    )

    long_df["Date"] = pd.to_datetime(long_df[date_col], format="%d-%m-%Y", errors="coerce")
    long_df["Month"] = pd.to_numeric(long_df[month_col], errors="coerce").astype("Int64")
    long_df["Year"] = pd.to_numeric(long_df[year_col], errors="coerce").astype("Int64")
    long_df["AOD"] = pd.to_numeric(long_df["AOD"], errors="coerce")

    long_df = long_df[
        long_df["Date"].between(START_DATE, END_DATE)
        & long_df["AOD"].notna()
    ].copy()
    return long_df[["Monitoring_Station", "Date", "Month", "Year", "AOD"]]


def detect_header_row(file_path: Path, scan_rows: int = 10) -> int:
    preview = pd.read_excel(file_path, header=None, nrows=scan_rows)
    for idx in range(len(preview.index)):
        row_values = [normalize_text(value) for value in preview.iloc[idx].tolist()]
        has_date = any(value == "date" for value in row_values)
        has_pm25 = any("pm25" in value.replace(" ", "") for value in row_values)
        if has_date and has_pm25:
            return idx
    return 0


def find_column(columns: pd.Index, accepted_keys: set[str], contains: tuple[str, ...] = ()) -> str | None:
    for col in columns:
        key = normalize_text(col)
        if key in accepted_keys:
            return str(col)
    for col in columns:
        key = normalize_text(col).replace(" ", "")
        if any(token in key for token in contains):
            return str(col)
    return None


def is_readable_xlsx(file_path: Path) -> tuple[bool, str]:
    if not file_path.exists():
        return False, "file not found"
    if file_path.stat().st_size == 0:
        return False, "empty file"

    try:
        with ZipFile(file_path, "r") as zip_file:
            bad_member = zip_file.testzip()
            if bad_member is not None:
                return False, f"corrupted zip member: {bad_member}"
    except BadZipFile as exc:
        return False, str(exc)
    except OSError as exc:
        return False, str(exc)

    return True, ""


def load_daily_pm25(file_path: Path, station_name: str) -> pd.DataFrame:
    try:
        header_row = detect_header_row(file_path)
        df = pd.read_excel(file_path, header=header_row)
    except Exception as exc:
        raise RuntimeError(f"Failed to read station file '{file_path.name}': {exc}") from exc

    df = df.dropna(axis=1, how="all")

    date_col = find_column(df.columns, {"date"})
    pm25_col = find_column(df.columns, set(), contains=("pm25", "pm2.5", "pm2_5"))

    if date_col is None or pm25_col is None:
        raise ValueError(f"Date/PM2.5 columns not found in {file_path.name}")

    work = df[[date_col, pm25_col]].copy()
    work.columns = ["Date", "PM2.5"]

    work["Date"] = pd.to_datetime(work["Date"], errors="coerce", format="mixed")
    work["PM2.5"] = pd.to_numeric(work["PM2.5"], errors="coerce")
    work = work[
        work["Date"].between(START_DATE, END_DATE)
        & work["PM2.5"].notna()
    ].copy()

    work["Date"] = work["Date"].dt.normalize()
    work = work.groupby("Date", as_index=False)["PM2.5"].mean()
    work.insert(0, "Monitoring_Station", station_name)
    return work


def build_station_file_map() -> dict[str, tuple[str, Path]]:
    station_map: dict[str, tuple[str, Path]] = {}
    for file_path in sorted(DOE_SPLIT_DIR.glob("*.xls*")):
        readable, reason = is_readable_xlsx(file_path)
        if not readable:
            print(f"  [!] Ignoring unreadable split file '{file_path.name}': {reason}")
            continue

        station_name = file_path.stem.strip()
        station_key = normalize_text(station_name)
        if station_key:
            station_map[station_key] = (station_name, file_path)
    return station_map


def merge_aod_pm25() -> pd.DataFrame:
    aod_long = load_aod_long()
    station_file_map = build_station_file_map()

    aod_station_lookup = {
        normalize_text(station): station
        for station in sorted(aod_long["Monitoring_Station"].dropna().unique())
    }
    matched_keys = sorted(set(aod_station_lookup) & set(station_file_map))

    if not matched_keys:
        raise ValueError("No overlapping stations found between AOD CSV and DoE split files.")

    print(f"Matched stations: {len(matched_keys)}")
    for key in matched_keys:
        print(f"  -> {aod_station_lookup[key]}")

    pm25_frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for station_key in matched_keys:
        aod_station_name = aod_station_lookup[station_key]
        _, file_path = station_file_map[station_key]
        try:
            pm25_frames.append(load_daily_pm25(file_path, aod_station_name))
        except RuntimeError as exc:
            skipped.append(aod_station_name)
            print(f"  [!] Skipping {aod_station_name}: {exc}")

    if not pm25_frames:
        raise ValueError("All matched station files failed to read. Please regenerate DoE_split_by_location files.")

    if skipped:
        print(f"Skipped stations due to unreadable files: {', '.join(skipped)}")

    pm25_long = pd.concat(pm25_frames, ignore_index=True)
    merged = aod_long.merge(pm25_long, on=["Monitoring_Station", "Date"], how="inner")
    merged = merged[["Monitoring_Station", "Date", "Month", "Year", "AOD", "PM2.5"]].copy()
    merged = merged.dropna(subset=["AOD", "PM2.5"])
    merged = merged.sort_values(["Monitoring_Station", "Date"], ignore_index=True)
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
    return merged


def main() -> None:
    validate_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Merging AOD and PM2.5 datasets...")
    final_df = merge_aod_pm25()
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 50)
    print(f"SUCCESS! Final dataset saved to: {OUTPUT_FILE}")
    print(f"Total rows generated: {len(final_df)}")
    print(f"Total stations used: {final_df['Monitoring_Station'].nunique()}")
    print("=" * 50 + "\n")
    print(final_df.head())


if __name__ == "__main__":
    main()