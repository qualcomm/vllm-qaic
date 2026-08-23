# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#!/usr/bin/env python3
"""
Scrape vLLM's supported_models.md for a given git ref, select one
representative HF model per architecture variant (smallest by param count,
or first listed if no size info), and write:
  llm_models_<ref>.json
  vlm_models_<ref>.json

Usage:
    python scrape_models.py --ref v0.9.1 --output-dir ./model_lists
    python scrape_models.py --ref main --output-dir .
    python scrape_models.py --ref main --type llm
"""

import regex as re
import sys
import json
import argparse
import urllib.request
from pathlib import Path

RAW_URL = "https://raw.githubusercontent.com/vllm-project/vllm/{ref}/docs/models/supported_models.md"


def ref_suffix(ref: str) -> str:
    """Normalise a git ref into a filename suffix.

    A ref may contain '/' (e.g. 'release/1.2'), which would otherwise be read as
    a path separator. Consumers of these files (ci_fallback_ops.sh,
    parse_logs_to_excel.py) apply the same substitution, so keep them in step.
    """
    return ref.replace("/", "_")


def fetch_markdown(ref: str) -> str:
    url = RAW_URL.format(ref=ref)
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def parse_param_count(model_id: str) -> float | None:
    """Return param count in billions, or None if not parseable."""
    name = model_id.split("/")[-1]
    # Patterns like 7B, 72B, 0.5B, 1.5B, 405B, 2.6B
    m = re.search(r"(\d+\.?\d*)[Bb](?:[^a-zA-Z]|$)", name)
    if m:
        return float(m.group(1))
    # Patterns like 350M, 700M
    m = re.search(r"(\d+\.?\d*)[Mm](?:[^a-zA-Z]|$)", name)
    if m:
        return float(m.group(1)) / 1000
    return None


def normalize(s: str) -> str:
    """Normalize string for fuzzy matching: lowercase, remove separators."""
    return re.sub(r"[\s\-_\.]", "", s).lower()


def extract_family(variant: str) -> str:
    """
    Derive a base family name from a variant string for priority matching.

    Rules (applied per token, splitting on spaces/hyphens/underscores):
    - Accumulate tokens that have 2+ leading alpha chars
    - If a token contains a digit it is a version marker: extract its alpha
      prefix and stop (e.g. 'Qwen2.5' -> 'qwen', then stop)
    - Single-char tokens (e.g. 'V' in 'V3', 'R' in 'Command-R') break the loop

    Examples:
      'Llama 3.1'   -> 'llama'
      'DeepSeek-V3' -> 'deepseek'
      'InternVL 3.5'-> 'internvl'
      'GPT-OSS'     -> 'gptoss'
      'GPT-2'       -> 'gpt'
      'Qwen2.5-VL'  -> 'qwen'
      'Command-R'   -> 'command'
    """
    tokens = re.split(r"[\s\-_]", variant.strip())
    parts = []
    for t in tokens:
        m = re.match(r"^([a-zA-Z]{2,})", t)
        if not m:
            break
        parts.append(m.group(1).lower())
        if re.search(r"\d", t):
            break
    return "".join(parts) if parts else re.sub(r"[^a-z]", "", variant.lower())


def extract_hf_models(cell: str) -> list:
    """Extract HF model IDs (org/repo) from a markdown table cell."""
    models = []
    seen = set()

    # Backtick format: `org/repo`
    for m in re.finditer(r"`([^`\s]+/[^`\s]+)`", cell):
        mid = m.group(1).strip()
        if mid not in seen and not mid.startswith("http"):
            seen.add(mid)
            models.append(mid)

    # Link format: [org/repo](url)
    for m in re.finditer(r"\[([^\]]+)\]\([^)]*\)", cell):
        text = m.group(1).strip()
        if "/" in text and text not in seen and not text.startswith("http"):
            seen.add(text)
            models.append(text)

    return models


def extract_architectures(cell: str) -> list:
    """Extract architecture class names from a table cell."""
    archs = re.findall(r"`([A-Za-z][A-Za-z0-9]+)`", cell)
    if not archs:
        raw = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cell).strip()
        archs = [a.strip() for a in raw.split(",") if a.strip()]
    return archs


def split_variants(cell: str) -> list:
    """Split the Models column into individual variant names."""
    cell = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cell)
    cell = re.sub(r"`([^`]+)`", r"\1", cell)
    return [v.strip() for v in cell.split(",") if v.strip()]


def select_model(candidates: list) -> str:
    """Pick the smallest-param model, or first if no size info available."""
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    sized = [(m, parse_param_count(m)) for m in candidates]
    with_size = [(m, s) for m, s in sized if s is not None]
    if with_size:
        return min(with_size, key=lambda x: x[1])[0]
    return candidates[0]


def match_variant_to_models(variant: str, all_models: list) -> list:
    """Find models whose repo name contains the normalized variant string.
    Falls back to stripping trailing zeros from the version (e.g. '3.0' -> '3')
    to handle cases like InternVL 3.0 matching InternVL3-9B.
    """
    norm_v = normalize(variant)
    matched = [m for m in all_models if norm_v in normalize(m.split("/")[-1])]
    if not matched:
        stripped = re.sub(r"0+$", "", norm_v)
        if stripped != norm_v:
            matched = [m for m in all_models if stripped in normalize(m.split("/")[-1])]
    return matched


def split_table_cells(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_separator_row(cells: list) -> bool:
    return all(re.match(r"^[-: ]+$", c) for c in cells if c)


def parse_tables(lines: list) -> list:
    """
    Walk through lines, find markdown tables with an 'Architecture' header,
    and parse each row. Returns list of dicts: architectures, variants, hf_models.
    """
    rows = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if "|" not in line or "Architecture" not in line:
            i += 1
            continue

        header_cells = split_table_cells(line)
        try:
            arch_idx = next(
                j for j, c in enumerate(header_cells) if "Architecture" in c
            )
            model_idx = next(
                j for j, c in enumerate(header_cells) if c.strip() == "Models"
            )
            hf_idx = next(
                j for j, c in enumerate(header_cells) if "HF" in c or "Example" in c
            )
        except StopIteration:
            i += 1
            continue

        i += 1  # skip header
        if i < len(lines) and "|" in lines[i]:
            i += 1  # skip separator row

        while i < len(lines) and "|" in lines[i]:
            cells = split_table_cells(lines[i].rstrip())
            i += 1

            if is_separator_row(cells):
                continue
            if max(arch_idx, model_idx, hf_idx) >= len(cells):
                continue

            archs = extract_architectures(cells[arch_idx])
            variants = split_variants(cells[model_idx])
            hf_models = [m for m in extract_hf_models(cells[hf_idx]) if "/" in m]

            if archs and variants and hf_models:
                rows.append(
                    {
                        "architectures": archs,
                        "variants": variants,
                        "hf_models": hf_models,
                    }
                )

    return rows


def get_section_lines(markdown: str, is_vlm: bool) -> list:
    """
    Return the lines belonging to the LLM or VLM top-level section.
    Falls back to all lines if detection fails.
    """
    lines = markdown.split("\n")

    llm_markers = ["text-only", "text generation", "text language"]
    vlm_markers = ["multimodal", "vision language", "vision model"]
    target_markers = vlm_markers if is_vlm else llm_markers

    # Find all # / ## header positions
    header_positions = [
        (i, len(re.match(r"^(#+)", lines[i]).group(1)), lines[i])
        for i in range(len(lines))
        if re.match(r"^#+\s", lines[i])
    ]

    target_start = None
    target_end = len(lines)

    for pos_idx, (line_i, level, header_text) in enumerate(header_positions):
        if any(m in header_text.lower() for m in target_markers):
            target_start = line_i
            # End at next header of same or higher level
            for next_i, next_level, _ in header_positions[pos_idx + 1 :]:
                if next_level <= level:
                    target_end = next_i
                    break
            break

    if target_start is None:
        return lines

    return lines[target_start:target_end]


def process(markdown: str, is_vlm: bool) -> list:
    """Parse section, select one model per variant, return list of entries."""
    section_lines = get_section_lines(markdown, is_vlm)
    rows = parse_tables(section_lines)

    selected = []
    seen_models = set()

    for row in rows:
        variants = row["variants"]
        hf_models = row["hf_models"]
        arch_str = ", ".join(row["architectures"])

        for variant in variants:
            matched = match_variant_to_models(variant, hf_models)
            if not matched:
                # Single-variant row: use all candidates
                if len(variants) == 1:
                    matched = hf_models
                else:
                    continue

            model = select_model(matched)
            if model and model not in seen_models:
                seen_models.add(model)
                selected.append(
                    {
                        "model": model,
                        "variant": variant,
                        "family": extract_family(variant),
                        "arch": arch_str,
                    }
                )

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM/VLM model lists from vLLM supported_models.md"
    )
    parser.add_argument(
        "--ref",
        default="main",
        help="vLLM git ref (tag, branch, commit). E.g. v0.9.1, main (default: main)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write output files (default: current dir)",
    )
    parser.add_argument(
        "--type",
        choices=["llm", "vlm", "both"],
        default="both",
        help="Which model type to generate (default: both)",
    )
    args = parser.parse_args()

    ref = args.ref
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching supported_models.md @ {ref} ...")
    try:
        markdown = fetch_markdown(ref)
    except Exception as e:
        print(f"Error fetching markdown: {e}", file=sys.stderr)
        sys.exit(1)

    targets = []
    if args.type in ("llm", "both"):
        targets.append((False, "llm"))
    if args.type in ("vlm", "both"):
        targets.append((True, "vlm"))

    for is_vlm, label in targets:
        print(f"Processing {label.upper()} models...")
        entries = process(markdown, is_vlm)

        json_path = out_dir / f"{label}_models_{ref_suffix(ref)}.json"

        with open(json_path, "w") as f:
            json.dump(entries, f, indent=2)

        print(f"  {len(entries)} models -> {json_path}")


if __name__ == "__main__":
    main()
