#!/usr/bin/env python3
"""
Label Telegram posts using labels.txt as the authoritative taxonomy.

The label file still lists piracy labels under PRIMARY / SECONDARY / TERTIARY sections, but this
script treats them as one pool: for piracy posts, assign up to three exact names in label_1 …
label_3 (most salient first). Fewer than three is normal. Benign posts get exactly one benign label.

Writes labeled_posts.sqlite (default path) with id, channel_name, post_link, post_text, labels, status.

Use --label-batch-size N (default 5) to send N posts per Ollama call as one JSON array.
Use 1 for one post per call.

Per-post failures do not stop the run.

If the output SQLite already exists, rows with labeling_status=ok are skipped on the next run
(checkpoint / resume). Use --no-resume to relabel every sampled row. Each batch is committed
before the next, so a crash loses at most the in-flight batch.

Dependencies:
  pip install ollama
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sqlite3
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

# Piracy: up to this many distinct taxonomy names, ordered label_1 (most salient) … label_N.
MAX_PIRACY_LABELS = 3
PIRACY_LABEL_KEYS: tuple[str, ...] = tuple(f"label_{i}" for i in range(1, MAX_PIRACY_LABELS + 1))

LABELED_TABLE = "labeled_posts"


def _empty_piracy_slot_dict() -> dict[str, str]:
    return {k: "" for k in PIRACY_LABEL_KEYS}


def _strip_json_fence(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", s, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s


def _piracy_tier_from_heading(heading: str) -> str | None:
    ul = heading.strip().upper()
    if ul.startswith("PRIMARY LABELS"):
        return "primary"
    if ul.startswith("SECONDARY LABELS"):
        return "secondary"
    if ul.startswith("TERTIARY LABELS"):
        return "tertiary"
    return None


def parse_labels_txt(path: Path) -> dict[str, dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cur_cat: str | None = None
    piracy_tier: str | None = None
    cur_label: str | None = None
    cur_desc_lines: list[str] = []
    out: dict[str, dict[str, str]] = {
        "piracy_primary": {},
        "piracy_secondary": {},
        "piracy_tertiary": {},
        "benign": {},
    }

    def flush() -> None:
        nonlocal cur_label, cur_desc_lines
        if not cur_label:
            cur_desc_lines = []
            return
        desc = "\n".join(l.rstrip() for l in cur_desc_lines).strip()
        if not desc:
            cur_label = None
            cur_desc_lines = []
            return
        if cur_cat == "piracy":
            if piracy_tier is None:
                raise ValueError(
                    f"Malformed labels.txt: label {cur_label!r} appears before a "
                    "PRIMARY/SECONDARY/TERTIARY LABELS section under PIRACY."
                )
            bucket = f"piracy_{piracy_tier}"
            out[bucket][cur_label] = desc
        elif cur_cat == "benign":
            out["benign"][cur_label] = desc
        cur_label = None
        cur_desc_lines = []

    for line in lines:
        if line.startswith("## "):
            flush()
            hdr = line[3:].strip().lower()
            if hdr.startswith("piracy"):
                cur_cat = "piracy"
                piracy_tier = None
            elif hdr.startswith("benign"):
                cur_cat = "benign"
                piracy_tier = None
            else:
                cur_cat = None
                piracy_tier = None
            continue
        if line.startswith("### "):
            flush()
            name = line[4:].strip()
            if cur_cat == "piracy":
                th = _piracy_tier_from_heading(name)
                if th:
                    piracy_tier = th
                    cur_label = None
                    continue
                cur_label = name
            elif cur_cat == "benign":
                cur_label = name
            else:
                cur_label = None
            continue
        if cur_label is not None:
            cur_desc_lines.append(line)

    flush()

    for key in ("piracy_primary", "piracy_secondary", "piracy_tertiary", "benign"):
        out[key] = {k: v for k, v in out[key].items() if k and v}

    if not any(out[k] for k in out):
        raise ValueError(
            "no labels parsed; expected ## PIRACY LABELS / ## BENIGN LABELS "
            "with ### tier headings and ### label names"
        )
    if not out["piracy_primary"] or not out["benign"]:
        raise ValueError(
            "labels.txt must define at least one piracy primary label and one benign label"
        )
    return out


def piracy_allowed_set(tax: dict[str, dict[str, str]]) -> set[str]:
    s: set[str] = set()
    s.update(tax["piracy_primary"])
    s.update(tax["piracy_secondary"])
    s.update(tax["piracy_tertiary"])
    return s


def piracy_merged_descriptions(tax: dict[str, dict[str, str]]) -> dict[str, str]:
    """Single map name -> description for prompt (primary, then secondary, then tertiary)."""
    merged: dict[str, str] = {}
    merged.update(tax["piracy_primary"])
    merged.update(tax["piracy_secondary"])
    merged.update(tax["piracy_tertiary"])
    return merged


def _dedupe_cap(items: list[str], cap: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
        if len(out) >= cap:
            break
    return out


def _normalize_piracy_slots(slot_values: list[str], allowed: set[str]) -> list[str]:
    """Keep order, drop invalid, dedupe, cap at MAX_PIRACY_LABELS; pad with empty strings."""
    ordered = [x.strip() for x in slot_values if isinstance(x, str) and x.strip()]
    filt = [x for x in ordered if x in allowed]
    uniq = _dedupe_cap(filt, MAX_PIRACY_LABELS)
    while len(uniq) < MAX_PIRACY_LABELS:
        uniq.append("")
    return uniq


def _string_slot(data: dict[str, Any], key: str) -> str:
    v = data.get(key)
    if v is None:
        return ""
    if not isinstance(v, str):
        raise ValueError(f"{key} must be a string")
    return v.strip()


def parse_decision_dict(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    category = (data.get("category") or "").strip().lower()
    if category not in {"piracy", "benign"}:
        raise ValueError("category must be 'piracy' or 'benign'")

    if category == "benign":
        label = data.get("benign_label")
        if label is None:
            label = data.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("benign responses require string field benign_label")
        out = {"category": "benign", "benign_label": label.strip()}
        out.update(_empty_piracy_slot_dict())
        return out

    slots = {k: _string_slot(data, k) for k in PIRACY_LABEL_KEYS}
    out = {"category": "piracy", "benign_label": None, **slots}
    return out


def parse_decision_json(raw: str) -> dict[str, Any]:
    cleaned = _strip_json_fence(raw)
    data = json.loads(cleaned)
    return parse_decision_dict(data)


def _extract_balanced_json_array(s: str) -> str | None:
    start = s.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        c = s[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None


def _remove_trailing_commas_outside_strings(s: str) -> str:
    out: list[str] = []
    in_string = False
    escape = False
    i = 0
    while i < len(s):
        c = s[i]
        if in_string:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == "," and i + 1 < len(s):
            j = i + 1
            while j < len(s) and s[j] in " \t\n\r":
                j += 1
            if j < len(s) and s[j] in "]}":
                i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _parse_json_array_loose(raw: str) -> list[Any] | None:
    cleaned = _strip_json_fence(raw)
    for candidate in (
        cleaned,
        _extract_balanced_json_array(cleaned) or "",
        _remove_trailing_commas_outside_strings(cleaned),
        _remove_trailing_commas_outside_strings(_extract_balanced_json_array(cleaned) or ""),
    ):
        if not candidate.strip():
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and "decisions" in data:
                data = data["decisions"]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            continue
    return None


def validate_against_taxonomy(
    parsed: dict[str, Any],
    tax: dict[str, dict[str, str]],
) -> dict[str, Any]:
    allowed = piracy_allowed_set(tax)
    ben = tax["benign"]

    if parsed["category"] == "benign":
        bl = parsed["benign_label"]
        if bl not in ben:
            raise ValueError(f"benign_label not in taxonomy: {bl!r}")
        out = {"category": "benign", "benign_label": bl}
        out.update(_empty_piracy_slot_dict())
        return out

    slot_vals = [parsed[k] for k in PIRACY_LABEL_KEYS]
    for k in PIRACY_LABEL_KEYS:
        x = parsed[k]
        if x and x not in allowed:
            raise ValueError(f"{k} not in piracy taxonomy: {x!r}")
    if not any(slot_vals):
        raise ValueError(
            f"piracy requires at least one non-empty slot among {', '.join(PIRACY_LABEL_KEYS)}"
        )
    normalized = _normalize_piracy_slots(slot_vals, allowed)
    out = {"category": "piracy", "benign_label": ""}
    for i, k in enumerate(PIRACY_LABEL_KEYS):
        out[k] = normalized[i]
    return out


def empty_label_row() -> dict[str, Any]:
    row: dict[str, Any] = {"category": "", "benign_label": ""}
    row.update(_empty_piracy_slot_dict())
    return row


def try_parse_json_object(raw: str) -> dict[str, Any] | None:
    try:
        cleaned = _strip_json_fence(raw)
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def salvage_labels(parsed: dict[str, Any] | None, tax: dict[str, dict[str, str]]) -> dict[str, Any]:
    row = empty_label_row()
    if not parsed:
        return row
    allowed = piracy_allowed_set(tax)
    cat = (parsed.get("category") or "").strip().lower()
    if cat not in {"piracy", "benign"}:
        return row
    row["category"] = cat
    if cat == "benign":
        bl = parsed.get("benign_label")
        if bl is None:
            bl = parsed.get("label")
        if isinstance(bl, str) and bl.strip() in tax["benign"]:
            row["benign_label"] = bl.strip()
        return row

    candidates: list[str] = []
    for k in PIRACY_LABEL_KEYS:
        v = parsed.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())
    if isinstance(parsed.get("primary"), str) and parsed["primary"].strip():
        candidates.append(parsed["primary"].strip())
    for key in ("secondary", "tertiary"):
        arr = parsed.get(key)
        if isinstance(arr, list):
            for x in arr:
                if isinstance(x, str) and x.strip():
                    candidates.append(x.strip())
    filt = [x for x in candidates if x in allowed]
    uniq = _dedupe_cap(filt, MAX_PIRACY_LABELS)
    while len(uniq) < MAX_PIRACY_LABELS:
        uniq.append("")
    for i, k in enumerate(PIRACY_LABEL_KEYS):
        row[k] = uniq[i]
    return row


def clip_error_message(msg: str, max_len: int = 2000) -> str:
    one_line = " ".join(msg.split())
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 1] + "…"


def _label_one_post(
    prompt: str,
    tax: dict[str, dict[str, str]],
    model: str,
    host: str | None,
) -> tuple[dict[str, Any], str, str]:
    try:
        raw = call_ollama_chat(model, prompt, host)
    except Exception as e:
        return salvage_labels(None, tax), "api_error", clip_error_message(f"{type(e).__name__}: {e}")

    try:
        parsed = parse_decision_json(raw)
    except Exception as e:
        loose = try_parse_json_object(raw)
        base = salvage_labels(loose, tax) if loose else empty_label_row()
        return base, "json_error", clip_error_message(f"{type(e).__name__}: {e}")

    try:
        validated = validate_against_taxonomy(parsed, tax)
    except ValueError as e:
        return salvage_labels(parsed, tax), "taxonomy_error", clip_error_message(str(e))

    return validated, "ok", ""


def _format_label_block(title: str, labels: dict[str, str]) -> str:
    parts = [f"### {name}\n{desc}" for name, desc in sorted(labels.items(), key=lambda kv: kv[0].lower())]
    return f"{title}\n" + "\n\n".join(parts)


def _labeling_prompt_core(tax: dict[str, dict[str, str]]) -> str:
    piracy_all = piracy_merged_descriptions(tax)
    n = MAX_PIRACY_LABELS
    piracy_block = _format_label_block(
        f"PIRACY — you may use at most {n} of these exact names total, in order of importance, "
        f"as label_1 through label_{n} (label_1 = most salient). Leave unused slots as \"\". "
        "Using only one or two labels is normal and preferred when the post does not justify more. "
        "Any name below is allowed in any slot.",
        piracy_all,
    )
    benign_block = _format_label_block(
        "BENIGN — assign exactly ONE of these names (flat):",
        tax["benign"],
    )
    piracy_json_example = (
        '{"category":"piracy","label_1":"<exact name or empty>","label_2":"","label_3":""}'
    )
    return textwrap.dedent(
        f"""\
        You label Telegram posts using ONLY the taxonomy below. The names are exact strings.

        CRITICAL: Never invent label names, paraphrases, or abbreviations. Every string you output for
        piracy slots or benign_label MUST be copied character-for-character from the lists below.
        If unsure, use fewer labels (even a single label_1 only is correct).

        TOP-LEVEL: Decide "piracy" or "benign" (mutually exclusive).

        If PIRACY:
        - At most {n} labels, in fields label_1 … label_{n} (most important first).
        - It is expected and good to output fewer than {n} labels; leave remaining slots as "".
        - Use ONLY names from the PIRACY list below.
        - Do not output benign_label.

        If BENIGN:
        - Exactly ONE benign label in benign_label (must be from the BENIGN list).
        - Set label_1 … label_{n} to "" (or omit them).

        Prefer observable evidence.

        --- TAXONOMY (exact names) ---

        {piracy_block}

        {benign_block}

        Each decision object must be exactly one of:

        {piracy_json_example}

        {{"category":"benign","benign_label":"<exact benign name>"}}

        Strings must match the taxonomy EXACTLY (including spaces, /, and &).
        """
    )


def build_label_prompt(post_text: str, tax: dict[str, dict[str, str]]) -> str:
    return (
        _labeling_prompt_core(tax)
        + "\nReturn ONLY valid JSON (no markdown fences), a single object for this post.\n\nPost:\n"
        + post_text
    )


def build_batch_label_prompt(clipped_posts: list[str], tax: dict[str, dict[str, str]]) -> str:
    n = len(clipped_posts)
    posts_body = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(clipped_posts))
    return (
        _labeling_prompt_core(tax)
        + f"\nReturn ONLY valid JSON (no markdown fences): a JSON array of exactly {n} objects.\n"
        + "Order must match posts: first array element = post [1], second = post [2], etc.\n"
        + "Do not use post_index fields.\n\nPosts:\n"
        + posts_body
    )


def _process_decision_item(
    item: Any,
    tax: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], str, str]:
    if not isinstance(item, dict):
        return (
            empty_label_row(),
            "json_error",
            clip_error_message(f"expected object, got {type(item).__name__}"),
        )
    try:
        parsed = parse_decision_dict(item)
    except Exception as e:
        return salvage_labels(item, tax), "json_error", clip_error_message(str(e))
    try:
        validated = validate_against_taxonomy(parsed, tax)
    except ValueError as e:
        return salvage_labels(parsed, tax), "taxonomy_error", clip_error_message(str(e))
    return validated, "ok", ""


def _label_batch_posts(
    clipped_posts: list[str],
    tax: dict[str, dict[str, str]],
    model: str,
    host: str | None,
) -> list[tuple[dict[str, Any], str, str]]:
    n = len(clipped_posts)
    if n == 0:
        return []
    if n == 1:
        prompt = build_label_prompt(clipped_posts[0], tax)
        return [_label_one_post(prompt, tax, model, host)]

    prompt = build_batch_label_prompt(clipped_posts, tax)
    try:
        raw = call_ollama_chat(model, prompt, host)
    except Exception as e:
        err = clip_error_message(f"{type(e).__name__}: {e}")
        empty = salvage_labels(None, tax)
        return [(empty, "api_error", err)] * n

    items = _parse_json_array_loose(raw)
    if items is None:
        err = clip_error_message("batch: could not parse JSON array from model output")
        empty = salvage_labels(None, tax)
        return [(empty, "json_error", err)] * n

    if len(items) > n:
        items = items[:n]

    results: list[tuple[dict[str, Any], str, str]] = []
    for j in range(n):
        if j < len(items):
            results.append(_process_decision_item(items[j], tax))
        else:
            results.append(
                (
                    empty_label_row(),
                    "json_error",
                    clip_error_message(
                        f"batch: missing array element for post [{j + 1}] (got {len(items)}/{n} objects)"
                    ),
                )
            )
    return results


def _maybe_int_id(val: Any) -> int | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _stable_post_link_key(row: dict[str, Any]) -> str:
    """Primary key for output DB: prefer post_link, else synthetic from source id or post hash."""
    pl = (row.get("post_link") or "").strip()
    if pl:
        return pl
    sid = row.get("id")
    if sid is not None and str(sid).strip() != "":
        return f"__source_id:{sid}"
    text = (row.get("post_text") or "").encode("utf-8", errors="replace")
    h = hashlib.sha256(text).hexdigest()[:32]
    return f"__hash:{h}"


def sample_posts_sqlite(db: Path, table: str, column: str, n: int, seed: int) -> list[dict[str, Any]]:
    """If n == 0, return every row with non-empty text column (random order)."""
    random.seed(seed)
    conn = sqlite3.connect(str(db))
    try:
        base = (
            f'SELECT id, channel_name, post_link, "{column}" AS post_text FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL AND trim("{column}") != \'\' '
            f"ORDER BY RANDOM()"
        )
        if n > 0:
            cur = conn.execute(base + " LIMIT ?", (n,))
        else:
            cur = conn.execute(base)
        rows: list[dict[str, Any]] = []
        for r in cur.fetchall():
            rows.append(
                {
                    "id": r[0],
                    "channel_name": r[1] if r[1] is not None else "",
                    "post_link": r[2] if r[2] is not None else "",
                    "post_text": r[3],
                }
            )
    finally:
        conn.close()
    return rows


def sample_posts_csv(path: Path, column: str, n: int, seed: int) -> list[dict[str, Any]]:
    """If n <= 0, return every row with non-empty text column (shuffled with seed)."""
    rng = random.Random(seed)
    res: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise SystemExit(f"CSV missing column {column!r}; have {reader.fieldnames!r}")
        for req in ("id", "channel_name", "post_link"):
            if req not in reader.fieldnames:
                print(
                    f"warning: CSV missing column {req!r}; stored values may be empty",
                    file=sys.stderr,
                )
        if n <= 0:
            for row in reader:
                t = (row.get(column) or "").strip()
                if not t:
                    continue
                res.append(
                    {
                        "id": _maybe_int_id(row.get("id")),
                        "channel_name": (row.get("channel_name") or "").strip(),
                        "post_link": (row.get("post_link") or "").strip(),
                        "post_text": t,
                    }
                )
            rng.shuffle(res)
            return res

        k = n
        seen = 0
        for row in reader:
            t = (row.get(column) or "").strip()
            if not t:
                continue
            item = {
                "id": _maybe_int_id(row.get("id")),
                "channel_name": (row.get("channel_name") or "").strip(),
                "post_link": (row.get("post_link") or "").strip(),
                "post_text": t,
            }
            seen += 1
            if len(res) < k:
                res.append(item)
            else:
                j = rng.randint(1, seen)
                if j <= k:
                    res[j - 1] = item
    return res


def init_labeled_sqlite(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    cols = ", ".join(
        [
            "post_link TEXT PRIMARY KEY",
            "id INTEGER",
            "channel_name TEXT",
            "post_text TEXT NOT NULL",
            "category TEXT",
            *[f'"{k}" TEXT' for k in PIRACY_LABEL_KEYS],
            "benign_label TEXT",
            "labeling_status TEXT",
            "labeling_error TEXT",
            "labeled_at TEXT NOT NULL DEFAULT (datetime('now'))",
        ]
    )
    conn.execute(f"CREATE TABLE IF NOT EXISTS {LABELED_TABLE} ({cols})")
    conn.commit()
    return conn


def insert_labeled_row(
    conn: sqlite3.Connection,
    *,
    post_link_key: str,
    id_val: int | None,
    channel_name: str,
    post_text: str,
    validated: dict[str, Any],
    label_status: str,
    label_error: str,
) -> None:
    cols = [
        "post_link",
        "id",
        "channel_name",
        "post_text",
        "category",
        *PIRACY_LABEL_KEYS,
        "benign_label",
        "labeling_status",
        "labeling_error",
    ]
    placeholders = ", ".join(["?"] * len(cols))
    col_sql = ", ".join(cols)
    vals: list[Any] = [
        post_link_key,
        id_val,
        channel_name,
        post_text,
        validated["category"],
        *[validated[k] for k in PIRACY_LABEL_KEYS],
        validated["benign_label"],
        label_status,
        label_error,
    ]
    conn.execute(
        f"INSERT OR REPLACE INTO {LABELED_TABLE} ({col_sql}) VALUES ({placeholders})",
        vals,
    )


def load_checkpoint_ok_keys(output_db: Path) -> set[str]:
    """Primary keys for posts already labeled successfully (resume after crash or stop)."""
    if not output_db.is_file():
        return set()
    try:
        uri = output_db.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    except (OSError, sqlite3.Error):
        return set()
    try:
        cur = conn.execute(
            f'SELECT post_link FROM "{LABELED_TABLE}" WHERE labeling_status = ?',
            ("ok",),
        )
        return {str(row[0]) for row in cur if row[0] is not None and str(row[0]).strip() != ""}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


def truncate_post(text: str, max_chars: int) -> str:
    t = text.replace("\r\n", "\n").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def call_ollama_chat(model: str, prompt: str, host: str | None) -> str:
    import ollama

    client = ollama.Client(host=host) if host else ollama.Client()
    r = client.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"]


def _format_eta(seconds: float) -> str:
    if seconds < 0 or seconds > 365 * 24 * 3600:
        return "…"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m:
        return f"{m:d}m {s:02d}s"
    return f"{s}s"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Label posts using labels.txt (piracy: up to 3 slots label_1–3; benign: one label) via Ollama."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--db",
        type=Path,
        help="SQLite database (ORDER BY RANDOM(); LIMIT from --sample-size unless 0 = all eligible rows)",
    )
    src.add_argument("--csv", type=Path, help="Posts CSV (reservoir sample if --sample-size > 0; else all rows)")
    p.add_argument("--table", default="posts", help="SQLite table name (default: posts)")
    p.add_argument("--column", default="post_text_en", help="Text column (default: post_text_en)")
    p.add_argument("--labels", type=Path, default=Path("labels.txt"), help="labels.txt path")
    p.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of posts to label; 0 = all eligible rows (default: 0)",
    )
    p.add_argument("--max-post-chars", type=int, default=1200, help="Truncate each post in prompt (default: 1200)")
    p.add_argument("--model", default="gemma3:27b", help="Ollama model name (default: gemma3:27b)")
    p.add_argument("--host", default=None, help="Ollama base URL (default: client default)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    p.add_argument(
        "--label-batch-size",
        type=int,
        default=5,
        help="Posts per Ollama call as one JSON array (default: 5). Use 1 for one post per call.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("labeled_posts.sqlite"),
        help="Output SQLite path (default: labeled_posts.sqlite)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip posts already labeled ok in -o (relabel whole sample; INSERT OR REPLACE)",
    )
    args = p.parse_args()

    if not args.labels.is_file():
        print(f"error: labels file not found: {args.labels}", file=sys.stderr)
        return 1

    try:
        tax = parse_labels_txt(args.labels)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    n_pir = len(piracy_allowed_set(tax))
    print(
        f"loaded taxonomy: piracy_names={n_pir} (pooled), benign={len(tax['benign'])}",
        flush=True,
    )

    n = args.sample_size
    if n < 0:
        print("error: --sample-size must be >= 0 (use 0 for full dataset)", file=sys.stderr)
        return 1

    batch_size = max(1, args.label_batch_size)

    t_load = time.perf_counter()
    if args.db:
        if not args.db.is_file():
            print(f"error: database not found: {args.db}", file=sys.stderr)
            return 1
        posts = sample_posts_sqlite(args.db, args.table, args.column, n, args.seed)
    else:
        if not args.csv.is_file():
            print(f"error: CSV not found: {args.csv}", file=sys.stderr)
            return 1
        posts = sample_posts_csv(args.csv, args.column, n, args.seed)
    load_dt = time.perf_counter() - t_load
    print(
        f"loaded {len(posts)} posts in {_format_eta(load_dt)}; "
        f"{batch_size} post(s) per Ollama call",
        flush=True,
    )

    if not posts:
        print("warning: no posts to label", file=sys.stderr)
        return 0

    sampled_n = len(posts)
    if args.no_resume:
        print("checkpoint: --no-resume (not skipping rows already ok in output)", flush=True)
    else:
        ok_keys = load_checkpoint_ok_keys(args.output)
        if ok_keys:
            posts = [p for p in posts if _stable_post_link_key(p) not in ok_keys]
            skipped = sampled_n - len(posts)
            if skipped:
                print(
                    f"checkpoint: skipping {skipped} post(s) already ok in {args.output.name}; "
                    f"{len(posts)} remaining",
                    flush=True,
                )

    if not posts:
        print(
            f"nothing to label (all {sampled_n} sampled post(s) already have labeling_status=ok).",
            flush=True,
        )
        return 0

    out_conn = init_labeled_sqlite(args.output)

    t_all = time.perf_counter()
    status_counts: dict[str, int] = {"ok": 0, "api_error": 0, "json_error": 0, "taxonomy_error": 0}
    done_posts = 0
    batch_idx = 0

    for batch_start in range(0, len(posts), batch_size):
        batch_idx += 1
        chunk = posts[batch_start : batch_start + batch_size]
        clipped_chunk = [truncate_post(r["post_text"], args.max_post_chars) for r in chunk]
        t0 = time.perf_counter()
        results = _label_batch_posts(clipped_chunk, tax, args.model, args.host)
        dt_batch = time.perf_counter() - t0

        st0, er0 = results[0][1], results[0][2]
        batch_wide_warn = len(chunk) > 1 and all(
            r[1] == st0 and r[2] == er0 for r in results
        ) and (
            st0 == "api_error"
            or (
                st0 == "json_error"
                and er0.startswith("batch: could not parse JSON array")
            )
        )

        for j, row in enumerate(chunk):
            global_i = batch_start + j + 1
            validated, label_status, label_error = results[j]
            status_counts[label_status] = status_counts.get(label_status, 0) + 1

            if label_status != "ok" and not batch_wide_warn:
                hint = label_error[:240] + ("…" if len(label_error) > 240 else "")
                print(
                    f"\nwarning [{label_status}] post {global_i}/{len(posts)}: {hint}",
                    file=sys.stderr,
                    flush=True,
                )

            id_val = row.get("id")
            if id_val is not None and not isinstance(id_val, int):
                id_val = _maybe_int_id(id_val)

            insert_labeled_row(
                out_conn,
                post_link_key=_stable_post_link_key(row),
                id_val=id_val,
                channel_name=str(row.get("channel_name") or ""),
                post_text=str(row.get("post_text") or ""),
                validated=validated,
                label_status=label_status,
                label_error=label_error,
            )
        out_conn.commit()

        if batch_wide_warn:
            hint = er0[:280] + ("…" if len(er0) > 280 else "")
            a, b = batch_start + 1, batch_start + len(chunk)
            print(
                f"\nwarning [{st0}] batch {batch_idx} posts {a}-{b}/{len(posts)}: {hint}",
                file=sys.stderr,
                flush=True,
            )

        done_posts += len(chunk)
        elapsed = time.perf_counter() - t_all
        avg_per_post = elapsed / done_posts if done_posts else 0.0
        eta = _format_eta(avg_per_post * max(0, len(posts) - done_posts))
        total_batches = (len(posts) + batch_size - 1) // batch_size
        fail_n = done_posts - status_counts["ok"]
        fail_suffix = f" | issues {fail_n}" if fail_n else ""
        print(
            f"[label] posts {done_posts}/{len(posts)} ({(100.0 * done_posts / len(posts)):.1f}%) | "
            f"batch {batch_idx}/{total_batches} last {_format_eta(dt_batch)} | "
            f"elapsed {_format_eta(elapsed)} | ETA {eta}{fail_suffix}",
            flush=True,
        )

    out_conn.close()
    print(
        f"done -> {args.output.resolve()} | table {LABELED_TABLE} | "
        f"ok={status_counts['ok']} api_error={status_counts['api_error']} "
        f"json_error={status_counts['json_error']} taxonomy_error={status_counts['taxonomy_error']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
