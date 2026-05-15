#!/usr/bin/env python3
"""
One-shot classifier test: loads fine-tuned LoRA and runs labeled inference.

By default uses a built-in sample Telegram-style post (`WITCH WATCH`). Override:

  python single_test_sample.py --text "your own post..."
  python single_test_sample.py --text "$(cat snippet.txt)"

Default `--adapter` search: beside this script, cwd, then parent repo.

Requires GPU + deps: pip install -r requirements.txt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

# Default corpus sample when --text is not given.
DEFAULT_POST_TEXT = """🔰 WITCH WATCH 🔰
( Witch Watch Season 1 )

🎭 Genre : Comedy, Drama, Fantasy, Romance, Slice of Life, Supernatural
🔊 Language : English Sub
📡 Status : FINISHED
🗓 Episode : 25
💾 Quality : 720p, 1080p
🔞 Rating : 13+
⭐️ Score : 7.2/10"""


def _glob_any(here: Path, pattern: str) -> bool:
    return next(here.glob(pattern), None) is not None


def validate_adapter_bundle(here: Path) -> tuple[list[str], list[str]]:
    fatal: list[str] = []
    adv: list[str] = []

    if not (here / "adapter_config.json").is_file():
        fatal.append("adapter_config.json (LoRA metadata)")

    weights_ok = (
        (here / "adapter_model.safetensors").is_file()
        or (here / "adapter_model.bin").is_file()
        or (here / "pytorch_model.bin").is_file()
        or _glob_any(here, "adapter_model*.safetensors")
    )
    if not weights_ok:
        fatal.append("adapter_model.safetensors (or compatible LoRA *.safetensors / .bin weights)")

    if not (here / "tokenizer_config.json").is_file():
        fatal.append("tokenizer_config.json")
    if not (here / "tokenizer.json").is_file() and not (here / "tokenizer.model").is_file():
        fatal.append("tokenizer.json (or tokenizer.model)")

    if not (here / "chat_template.jinja").is_file():
        adv.append("chat_template.jinja absent (fine if tokenizer_config embeds chat_template)")
    if not (here / "special_tokens_map.json").is_file():
        adv.append("special_tokens_map.json absent (recommended with tokenizer)")
    return fatal, adv


def adapter_search_paths() -> list[Path]:
    raw = (
        _SCRIPT_DIR / "unsloth_telegram_labels",
        Path.cwd() / "unsloth_telegram_labels",
        _SCRIPT_DIR.parent / "unsloth_telegram_labels",
    )
    ordered: list[Path] = []
    seen: set[str] = set()
    for q in raw:
        try:
            rp = q.resolve(strict=False)
        except (OSError, RuntimeError):
            rp = q
        key = str(rp)
        if key not in seen:
            seen.add(key)
            ordered.append(rp)
    return ordered


_USER_INSTRUCTION_INTRO = (
    "You label Telegram posts for a piracy/abuse taxonomy. "
    "Reply with ONLY a single JSON object, no markdown fences, no extra text."
)


def build_user_message(post_text: str) -> str:
    t = post_text.replace("\r\n", "\n").strip()
    return f"{_USER_INSTRUCTION_INTRO}\n\n--- post ---\n{t}\n"


def strip_json_fence(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", s, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s


def try_parse_output(raw: str) -> tuple[dict | None, str]:
    cleaned = strip_json_fence(raw)
    try:
        obj = json.loads(cleaned)
        return (obj if isinstance(obj, dict) else None, cleaned)
    except json.JSONDecodeError:
        return None, cleaned


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-sample LoRA classify test.")
    p.add_argument(
        "--base-model",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct",
        help="HF Hub id OR local snapshot directory.",
    )
    p.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Checkpoint dir (omit = search beside script, cwd, parent/).",
    )
    p.add_argument(
        "--text",
        type=str,
        default=None,
        help="Post body to classify. Omit = built-in Witch Watch sample.",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit base load.")
    p.add_argument("--max-seq-length", type=int, default=4096)
    return p.parse_args()


def resolve_adapter(explicit: Path | None) -> tuple[Path, list[Path]]:
    search = adapter_search_paths()
    if explicit is not None:
        path = explicit.expanduser().resolve(strict=False)
        return path, search
    for cand in search:
        if cand.is_dir():
            return cand.resolve(strict=False), search
    return search[0], search


def main() -> None:
    args = parse_args()
    post = DEFAULT_POST_TEXT if args.text is None else args.text
    if not post.strip():
        raise SystemExit("Empty post text (pass --text or fix default).")

    try:
        import torch
        from peft import PeftModel

        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise SystemExit(
            "Missing deps. Install categorization_model/requirements.txt "
            f"(GPU + CUDA): {e}"
        )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — load the model on GPU.")

    adapter_path, searched = resolve_adapter(args.adapter)
    if args.adapter is not None and not adapter_path.is_dir():
        raise SystemExit(f"--adapter is not a directory: {adapter_path}")
    if args.adapter is None and not adapter_path.is_dir():
        tries = "\n  ".join(str(p) for p in searched)
        raise SystemExit(
            "Adapter directory not found.\n"
            f"Tried (in order):\n  {tries}\n\n"
            "Copy unsloth_telegram_labels here, train with finetune.py, "
            "or pass:  --adapter /path/to/unsloth_telegram_labels"
        )

    bad, adv = validate_adapter_bundle(adapter_path)
    if bad:
        msg = (
            f"{adapter_path} missing expected checkpoint files.\n"
            "Prefer the checkpoint root with adapter_model.safetensors + tokenizer files.\n"
            "Missing:\n  - "
        )
        raise SystemExit(msg + "\n  - ".join(bad))
    for line in adv:
        print(line, file=sys.stderr)

    source = "built-in Witch Watch sample" if args.text is None else "--text override"
    print(f"Using {source}; adapter={adapter_path}", file=sys.stderr)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_4bit = not args.no_4bit

    print(f"Loading base {args.base_model!r}…", file=sys.stderr)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None if load_4bit else dtype,
        load_in_4bit=load_4bit,
        token=None,
    )
    print(f"Loading LoRA from {adapter_path}…", file=sys.stderr)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    tok_cfg = adapter_path / "tokenizer_config.json"
    if tok_cfg.is_file():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=True)
    FastLanguageModel.for_inference(model)

    user_msg = build_user_message(post)
    messages = [{"role": "user", "content": user_msg}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = eos_id

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

    prompt_len = int(input_ids.shape[1])
    gen_ids = out[0][prompt_len:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    parsed, cleaned = try_parse_output(raw_text)

    print("--- input post ---")
    print(post.strip())
    print("--- raw_model_output ---")
    print(raw_text)
    print("--- stripped_for_json ---")
    print(cleaned)
    if parsed is not None:
        print("--- parsed ---")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    else:
        print("--- parsed ---")
        print("(not valid JSON; check raw output)")


if __name__ == "__main__":
    main()
