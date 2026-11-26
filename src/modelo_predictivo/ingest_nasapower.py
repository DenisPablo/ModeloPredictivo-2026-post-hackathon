"""Automated NASA POWER regional daily weather downloader with live progress.

Downloads daily meteorological variables for Argentina (2018-2026) from the
NASA POWER `/api/temporal/daily/regional` API over 6 bounding-box tiles
(native 0.5 deg lat x 0.625 deg lon grid), converts -999 fill values to null,
and persists resumable per-year Parquet files (`float32`, ZSTD).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

REGIONAL_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/regional"

# NASA POWER daily parameters (1 parameter per /regional request limit)
NASA_PARAMETERS = (
    "T2M_MAX",            # Max temperature at 2m (C)
    "RH2M",               # Relative humidity at 2m (%)
    "WS10M_MAX",          # Max wind speed at 10m (m/s -> converted to km/h)
    "PRECTOTCORR",        # Bias-corrected precipitation (mm/day)
    "ALLSKY_SFC_SW_DWN",  # All-sky surface shortwave downward irradiance (MJ/m2/day)
)

# 12 tiles covering continental Argentina (lat -55.5..-21.5, lon -74.0..-53.0),
# each 8.5 deg lat x 7.0 deg lon to stay within NASA POWER's 10x10 deg limit.
ARGENTINA_TILES: tuple[tuple[str, float, float, float, float], ...] = (
    ("norte_oeste", -30.0, -21.5, -74.0, -67.0),
    ("norte_centro", -30.0, -21.5, -67.0, -60.0),
    ("norte_este", -30.0, -21.5, -60.0, -53.0),
    ("centro_norte_oeste", -38.5, -30.0, -74.0, -67.0),
    ("centro_norte_centro", -38.5, -30.0, -67.0, -60.0),
    ("centro_norte_este", -38.5, -30.0, -60.0, -53.0),
    ("centro_sur_oeste", -47.0, -38.5, -74.0, -67.0),
    ("centro_sur_centro", -47.0, -38.5, -67.0, -60.0),
    ("centro_sur_este", -47.0, -38.5, -60.0, -53.0),
    ("sur_oeste", -55.5, -47.0, -74.0, -67.0),
    ("sur_centro", -55.5, -47.0, -67.0, -60.0),
    ("sur_este", -55.5, -47.0, -60.0, -53.0),
)

FILL_VALUE = -999.0

NASA_SCHEMA = pa.schema(
    [
        ("lat_clima", pa.float32()),
        ("lon_clima", pa.float32()),
        ("elevacion_m", pa.float32()),
        ("fecha", pa.date32()),
        ("temperature_2m_max", pa.float32()),
        ("relative_humidity_2m", pa.float32()),
        ("wind_speed_10m_max", pa.float32()),
        ("precipitation_sum", pa.float32()),
        ("shortwave_radiation_sum", pa.float32()),
    ]
)

PARAM_TO_COL = {
    "T2M_MAX": "temperature_2m_max",
    "RH2M": "relative_humidity_2m",
    "WS10M_MAX": "wind_speed_10m_max",
    "PRECTOTCORR": "precipitation_sum",
    "ALLSKY_SFC_SW_DWN": "shortwave_radiation_sum",
}


def format_duration(seconds: float) -> str:
    """Format seconds into compact HH:MM:SS string."""
    secs = max(0, int(seconds))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_progress(
    done_steps: int,
    total_steps: int,
    label: str,
    start_ts: float,
    rows_total: int,
) -> None:
    """Print an in-place terminal progress bar with percentage and ETA."""
    pct = (done_steps / max(total_steps, 1)) * 100.0
    bar_len = 28
    filled = int(bar_len * done_steps / max(total_steps, 1))
    bar = "#" * filled + "-" * (bar_len - filled)

    elapsed = time.perf_counter() - start_ts
    if done_steps > 0:
        eta_s = (elapsed / done_steps) * (total_steps - done_steps)
        eta_str = format_duration(eta_s)
    else:
        eta_str = "--:--:--"

    line = (
        f"\r[{bar}] {done_steps}/{total_steps} ({pct:5.1f}%) | "
        f"{label} | rows={rows_total:,} | elapsed={format_duration(elapsed)} | ETA={eta_str}"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def build_regional_url(
    parameter: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start_ymd: str,
    end_ymd: str,
) -> str:
    """Build NASA POWER regional daily API URL for one parameter and tile."""
    params = {
        "parameters": parameter,
        "community": "AG",
        "longitude-min": f"{lon_min}",
        "longitude-max": f"{lon_max}",
        "latitude-min": f"{lat_min}",
        "latitude-max": f"{lat_max}",
        "start": start_ymd,
        "end": end_ymd,
        "format": "JSON",
    }
    return f"{REGIONAL_ENDPOINT}?{urllib.parse.urlencode(params)}"


def fetch_nasa_json(url: str, max_retries: int = 5, timeout_s: int = 90) -> dict[str, Any]:
    """Fetch JSON from NASA POWER with exponential backoff on transient errors."""
    delay = 3.0
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ml_nasaspaceapp-nasapower-ingest/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from NASA POWER: {body}") from exc
        except urllib.error.URLError:
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise


def merge_tile_payload(
    grid_store: dict[tuple[float, float, str], dict[str, Any]],
    payload: dict[str, Any],
    parameter: str,
) -> None:
    """Merge one NASA POWER parameter payload into (lat, lon, date_str) dict."""
    col_name = PARAM_TO_COL[parameter]
    for feat in payload.get("features", []):
        coords = feat.get("geometry", {}).get("coordinates", [])
        if len(coords) < 2:
            continue
        lon_val = float(coords[0])
        lat_val = float(coords[1])
        elev_val = float(coords[2]) if len(coords) >= 3 and coords[2] is not None else None

        series = feat.get("properties", {}).get("parameter", {}).get(parameter, {})
        for ymd, raw_val in series.items():
            val = None if raw_val is None or float(raw_val) <= FILL_VALUE else float(raw_val)
            # Convert WS10M_MAX from m/s to km/h to match Open-Meteo units
            if val is not None and parameter == "WS10M_MAX":
                val *= 3.6

            key = (lat_val, lon_val, ymd)
            rec = grid_store.get(key)
            if rec is None:
                rec = {
                    "lat_clima": lat_val,
                    "lon_clima": lon_val,
                    "elevacion_m": elev_val,
                    "fecha": ymd,
                }
                grid_store[key] = rec
            rec[col_name] = val


def write_year_parquet(
    grid_store: dict[tuple[float, float, str], dict[str, Any]],
    output_path: Path,
) -> int:
    """Write merged year records to ZSTD Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list[Any]] = {name: [] for name in NASA_SCHEMA.names}

    for rec in grid_store.values():
        ymd = rec["fecha"]
        columns["lat_clima"].append(rec["lat_clima"])
        columns["lon_clima"].append(rec["lon_clima"])
        columns["elevacion_m"].append(rec.get("elevacion_m"))
        columns["fecha"].append(datetime.strptime(ymd, "%Y%m%d").date())
        for col in PARAM_TO_COL.values():
            columns[col].append(rec.get(col))

    table = pa.table(columns, schema=NASA_SCHEMA)
    pq.write_table(table, output_path, compression="zstd")
    return len(grid_store)


def show_status(output_dir: Path, start_year: int, end_year: int) -> None:
    """Display current download progress in output_dir."""
    years = list(range(start_year, end_year + 1))
    done_years = []
    total_bytes = 0
    total_rows = 0

    print(f"\n=== NASA POWER Archive Status ({output_dir}) ===")
    for y in years:
        p = output_dir / f"nasapower_{y}.parquet"
        if p.exists():
            size_b = p.stat().st_size
            meta = pq.read_metadata(p)
            total_bytes += size_b
            total_rows += meta.num_rows
            done_years.append(y)
            print(f"  [OK] {y}: {meta.num_rows:>9,} rows | {size_b / 1e6:6.2f} MB")
        else:
            print(f"  [  ] {y}: pending")

    pct = (len(done_years) / max(len(years), 1)) * 100.0
    print(
        f"\nCompleted: {len(done_years)}/{len(years)} years ({pct:.1f}%) | "
        f"Total: {total_rows:,} rows | {total_bytes / 1e6:.2f} MB\n"
    )


def run_download(
    output_dir: Path,
    start_year: int,
    end_year: int,
    workers: int = 4,
) -> None:
    """Download NASA POWER 2018-2026 archive with live terminal progress bar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, workers)
    # NASA POWER typically lags ~3-4 days behind real time
    latest_safe = date.today() - timedelta(days=4)

    pending_years = [
        y for y in range(start_year, end_year + 1)
        if not (output_dir / f"nasapower_{y}.parquet").exists()
    ]
    if not pending_years:
        print("All requested years are already downloaded.")
        show_status(output_dir, start_year, end_year)
        return

    total_steps = len(pending_years) * len(ARGENTINA_TILES) * len(NASA_PARAMETERS)
    done_steps = 0
    rows_total = 0
    start_ts = time.perf_counter()

    print(
        f"Starting NASA POWER download: years={pending_years[0]}..{pending_years[-1]} "
        f"({len(pending_years)} pending) | {len(ARGENTINA_TILES)} tiles x "
        f"{len(NASA_PARAMETERS)} params = {total_steps} requests | workers={workers}"
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for year in pending_years:
            year_file = output_dir / f"nasapower_{year}.parquet"
            start_ymd = f"{year}0101"
            end_dt = min(date(year, 12, 31), latest_safe)
            end_ymd = end_dt.strftime("%Y%m%d")

            grid_store: dict[tuple[float, float, str], dict[str, Any]] = {}
            future_to_meta = {}
            for tile_name, lat_min, lat_max, lon_min, lon_max in ARGENTINA_TILES:
                for param in NASA_PARAMETERS:
                    url = build_regional_url(
                        parameter=param,
                        lat_min=lat_min,
                        lat_max=lat_max,
                        lon_min=lon_min,
                        lon_max=lon_max,
                        start_ymd=start_ymd,
                        end_ymd=end_ymd,
                    )
                    fut = executor.submit(fetch_nasa_json, url)
                    future_to_meta[fut] = (tile_name, param)

            render_progress(done_steps, total_steps, f"{year} fetching...", start_ts, rows_total)
            try:
                for fut in as_completed(future_to_meta):
                    tile_name, param = future_to_meta[fut]
                    payload = fut.result()
                    merge_tile_payload(grid_store, payload, param)
                    done_steps += 1
                    label = f"{year} {tile_name} [{param}]"
                    render_progress(done_steps, total_steps, label, start_ts, rows_total)
            except BaseException:
                for fut in future_to_meta:
                    fut.cancel()
                raise

            written = write_year_parquet(grid_store, year_file)
            rows_total += written
            size_mb = year_file.stat().st_size / 1e6
            sys.stdout.write(
                f"\n[Saved] {year_file.name}: {written:,} rows ({size_mb:.2f} MB)\n"
            )
            sys.stdout.flush()

    print("\nDownload completed.")
    show_status(output_dir, start_year, end_year)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated NASA POWER daily weather downloader with live progress."
    )
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/weather/nasapower"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current download progress without downloading.",
    )
    args = parser.parse_args()

    if args.status:
        show_status(args.output_dir, args.start_year, args.end_year)
    else:
        run_download(args.output_dir, args.start_year, args.end_year, workers=args.workers)


if __name__ == "__main__":
    main()
