"""Download antibody sequences from the Observed Antibody Space (OAS).

OAS data-units are .csv.gz files where:
  - Line 1: JSON metadata (species, chain type, study info, etc.)
  - Remaining lines: CSV with annotated antibody sequences

The key column for amino acid sequences is 'sequence_alignment_aa'.
CDR regions are also pre-annotated (cdr1_aa, cdr2_aa, cdr3_aa).
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

OAS_SEARCH_URL = "https://opig.stats.ox.ac.uk/webapps/oas/oas_unpaired/"


def search_oas(
    species: str = "human",
    chain: str = "Heavy",
    max_results: int = 5,
) -> list[str]:
    """Query the OAS search endpoint and return download URLs.

    Falls back to a curated list of known human heavy chain data-unit URLs
    if the search endpoint is unavailable.
    """
    fallback_urls = [
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Setliff_2019/csv/SRR8365927_Heavy_IGHA.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Setliff_2019/csv/SRR8365927_Heavy_IGHD.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Setliff_2019/csv/SRR8365927_Heavy_IGHG.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Setliff_2019/csv/SRR8365927_Heavy_IGHM.csv.gz",
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Briney_2019/csv/SRR5110244_Heavy_IGHM.csv.gz",
    ]

    try:
        response = requests.post(
            OAS_SEARCH_URL,
            data={"species": species, "chain": chain},
            timeout=30,
        )
        response.raise_for_status()

        urls = []
        for line in response.text.splitlines():
            if "ngsdb/unpaired" in line and ".csv.gz" in line:
                start = line.find("https://")
                if start != -1:
                    end = line.find(".csv.gz", start) + len(".csv.gz")
                    urls.append(line[start:end])

        if urls:
            return urls[:max_results]
    except Exception as e:
        logger.warning("OAS search failed (%s), using fallback URLs", e)

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


def parse_data_unit(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Parse an OAS .csv.gz file into metadata and sequence records.

    Returns:
        metadata: dict with study info (species, chain, author, etc.)
        records: list of dicts, each with at minimum 'sequence_alignment_aa'
    """
    with gzip.open(path, "rt") as f:
        first_line = f.readline().strip()
        metadata = json.loads(first_line.replace('""', '"').strip('"'))

        reader = csv.DictReader(f)
        records = []
        for row in reader:
            seq = row.get("sequence_alignment_aa", "").replace("-", "")
            if seq:
                records.append({
                    "sequence": seq,
                    "cdr1_aa": row.get("cdr1_aa", ""),
                    "cdr2_aa": row.get("cdr2_aa", ""),
                    "cdr3_aa": row.get("cdr3_aa", ""),
                    "v_call": row.get("v_call", ""),
                    "j_call": row.get("j_call", ""),
                })

    return metadata, records


def download_oas_subset(
    output_dir: str | Path,
    species: str = "human",
    chain: str = "Heavy",
    max_files: int = 5,
    max_sequences: int = 50_000,
) -> list[dict[str, str]]:
    """Download and parse a subset of OAS data.

    Downloads up to `max_files` data-units and collects up to
    `max_sequences` total records.

    Returns:
        List of sequence records with fields:
        sequence, cdr1_aa, cdr2_aa, cdr3_aa, v_call, j_call
    """
    output_dir = Path(output_dir)
    urls = search_oas(species=species, chain=chain, max_results=max_files)

    all_records: list[dict[str, str]] = []
    for url in urls:
        if len(all_records) >= max_sequences:
            break
        try:
            path = download_data_unit(url, output_dir)
            _metadata, records = parse_data_unit(path)
            all_records.extend(records)
            logger.info(
                "Parsed %d sequences from %s (total: %d)",
                len(records),
                path.name,
                len(all_records),
            )
        except Exception as e:
            logger.warning("Failed to process %s: %s", url, e)
            continue

    return all_records[:max_sequences]
