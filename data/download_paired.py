"""Download paired VH+VL antibody sequences from the Observed Antibody Space (OAS).

OAS paired data-units are .csv.gz files where:
  - Line 1: JSON metadata (species, study info, etc.)
  - Remaining lines: CSV with paired heavy and light chain sequences

Key columns for paired data include:
  sequence_alignment_aa_heavy, sequence_alignment_aa_light,
  cdr{1,2,3}_aa_heavy, cdr{1,2,3}_aa_light,
  v_call_heavy, j_call_heavy, v_call_light, j_call_light
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

OAS_PAIRED_SEARCH_URL = "https://opig.stats.ox.ac.uk/webapps/oas/oas_paired/"


def search_oas_paired(
    species: str = "human",
    max_results: int = 200,
) -> list[str]:
    """Query the OAS paired search endpoint and return download URLs.

    Falls back to a curated list of verified human paired data-unit URLs
    if the search endpoint is unavailable.
    """
    # Verified URLs from the largest OAS paired studies (as of 2026-04).
    # Jaffe_2022 (~1.4M seqs, 94 units) and Wang_2023 (~192K, 26 units)
    # together easily cover 500K+ sequences.
    fallback_urls = [
        # Jaffe_2022 — largest paired study (~15K seqs per unit)
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279049_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279050_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279051_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279052_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279053_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279054_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279055_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279056_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279057_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279058_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279059_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279060_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279061_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279062_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279063_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279064_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279065_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279066_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279067_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279068_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279069_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279070_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279071_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279072_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279073_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279074_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279075_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279076_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279077_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279078_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279079_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279080_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279081_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279082_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279083_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279084_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279085_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279086_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279087_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1279088_1_Paired_All.csv.gz",
        # Wang_2023 — large paired study (~7.4K seqs per unit)
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Wang_2023/csv_paired/SRR22279283_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Wang_2023/csv_paired/SRR22279284_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Wang_2023/csv_paired/SRR22279285_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Wang_2023/csv_paired/SRR22279286_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Wang_2023/csv_paired/SRR22279287_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Wang_2023/csv_paired/SRR22279288_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Wang_2023/csv_paired/SRR22279289_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Wang_2023/csv_paired/SRR22279290_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Wang_2023/csv_paired/SRR22279291_1_Paired_All.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Wang_2023/csv_paired/SRR22279292_1_Paired_All.csv.gz",
    ]

    try:
        response = requests.post(
            OAS_PAIRED_SEARCH_URL,
            data={"Species": species, "Age": "*", "BSource": "*", "BType": "*",
                  "Vaccine": "*", "Disease": "*", "Subject": "*", "Longitudinal": "*"},
            timeout=30,
        )
        response.raise_for_status()

        urls = []
        for line in response.text.splitlines():
            if "ngsdb/paired" in line and ".csv.gz" in line:
                start = line.find("https://")
                if start != -1:
                    end = line.find(".csv.gz", start) + len(".csv.gz")
                    urls.append(line[start:end])

        if urls:
            return urls[:max_results]
    except Exception as e:
        logger.warning("OAS paired search failed (%s), using fallback URLs", e)

    return fallback_urls[:max_results]


def download_data_unit(url: str, output_dir: Path) -> Path:
    """Download a single OAS data-unit .csv.gz file.

    Returns the path to the downloaded file. Skips download if file exists.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    output_path = output_dir / filename

    if output_path.exists():
        logger.info("Already downloaded: %s", filename)
        return output_path

    logger.info("Downloading: %s", url)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with output_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info("Saved: %s", output_path)
    return output_path


def parse_paired_data_unit(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Parse a paired OAS .csv.gz file into metadata and sequence records.

    Returns:
        metadata: dict with study info (species, author, etc.)
        records: list of dicts with paired heavy+light chain fields
    """
    with gzip.open(path, "rt") as f:
        first_line = f.readline().strip()
        metadata = json.loads(first_line.replace('""', '"').strip('"'))

        reader = csv.DictReader(f)
        records = []
        for row in reader:
            seq_heavy = row.get("sequence_alignment_aa_heavy", "").replace("-", "")
            seq_light = row.get("sequence_alignment_aa_light", "").replace("-", "")
            if not seq_heavy or not seq_light:
                continue

            records.append({
                "sequence_heavy": seq_heavy,
                "sequence_light": seq_light,
                "cdr1_aa_heavy": row.get("cdr1_aa_heavy", ""),
                "cdr2_aa_heavy": row.get("cdr2_aa_heavy", ""),
                "cdr3_aa_heavy": row.get("cdr3_aa_heavy", ""),
                "cdr1_aa_light": row.get("cdr1_aa_light", ""),
                "cdr2_aa_light": row.get("cdr2_aa_light", ""),
                "cdr3_aa_light": row.get("cdr3_aa_light", ""),
                "v_call_heavy": row.get("v_call_heavy", ""),
                "j_call_heavy": row.get("j_call_heavy", ""),
                "v_call_light": row.get("v_call_light", ""),
                "j_call_light": row.get("j_call_light", ""),
            })

    return metadata, records


def download_paired_subset(
    output_dir: str | Path,
    species: str = "human",
    max_files: int = 100,
    max_sequences: int = 500_000,
) -> list[dict[str, str]]:
    """Download and parse a subset of OAS paired data.

    Downloads up to `max_files` data-units and collects up to
    `max_sequences` total records.

    Returns:
        List of paired sequence records with fields:
        sequence_heavy, sequence_light, cdr{1,2,3}_aa_{heavy,light},
        v_call_{heavy,light}, j_call_{heavy,light}
    """
    output_dir = Path(output_dir)
    urls = search_oas_paired(species=species, max_results=max_files)

    all_records: list[dict[str, str]] = []
    for url in urls:
        if len(all_records) >= max_sequences:
            break
        try:
            path = download_data_unit(url, output_dir)
            _metadata, records = parse_paired_data_unit(path)
            all_records.extend(records)
            logger.info(
                "Parsed %d paired sequences from %s (total: %d)",
                len(records),
                path.name,
                len(all_records),
            )
        except Exception as e:
            logger.warning("Failed to process %s: %s", url, e)
            continue

    return all_records[:max_sequences]
