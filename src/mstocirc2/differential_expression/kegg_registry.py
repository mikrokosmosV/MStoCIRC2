from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

KEGG_ORGANISM_SOURCE = "https://www.genome.jp/kegg/tables/br08606.html"


@lru_cache(maxsize=1)
def load_kegg_organism_records() -> list[dict[str, object]]:
    data_path = files("mstocirc2.differential_expression").joinpath("kegg_organisms.json")
    return json.loads(data_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_kegg_organism_map() -> dict[str, dict[str, object]]:
    return {
        str(record["abbr"]).strip().lower(): dict(record)
        for record in load_kegg_organism_records()
    }


def normalize_kegg_organism(value: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError(
            "Missing KEGG organism abbreviation. Use a standard abbreviation such as "
            "'hsa', 'mmu', or 'ath'."
        )
    if normalized not in load_kegg_organism_map():
        raise ValueError(
            f"Unsupported KEGG organism abbreviation '{value}'. "
            f"See {KEGG_ORGANISM_SOURCE} for the official registry."
        )
    return normalized


def describe_kegg_organism(value: str) -> str:
    record = load_kegg_organism_map()[normalize_kegg_organism(value)]
    return f"{record['abbr']} ({record['name']})"

