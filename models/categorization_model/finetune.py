#!/usr/bin/env python3
"""

Binge Bot Repeat

Categorization model: predict Taxonomy labels from post_text.

Default base model aligns with `llama3.2:1b` (~Meta Llama 3.2 1B Instruct):
  HF / Unsloth: unsloth/Llama-3.2-1B-Instruct (4-bit preload via Unsloth)

All paths are anchored to this directory (standalone if you move the folder):
  - Training data: labeled_posts.sqlite next to these scripts or in cwd (--db)
  - Checkpoints:   ./unsloth_telegram_labels by default (--output-dir)

Target JSON matches label_posts.parse_decision_dict shapes:
  - benign: {\"category\":\"benign\",\"benign_label\":\"...\"}
  - piracy: {\"category\":\"piracy\",\"label_1\":\"\",\"label_2\":\"\",\"label_3\":\"\"}
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent

LABELED_TABLE = "labeled_posts"


@dataclass(frozen=True)
class LabeledExample:
    post_text: str
    category: str
    benign_label: str
    label_1: str
    label_2: str
    label_3: str


def default_sqlite_candidates() -> list[Path]:
    """Standalone: this folder first, then current working directory."""
    return [_SCRIPT_DIR / "labeled_posts.sqlite", Path.cwd() / "labeled_posts.sqlite"]


def resolve_database_path(cli: Path | None) -> Path:
    if cli is not None:
        return cli.expanduser().resolve()
    for p in default_sqlite_candidates():
        if p.is_file():
            return p.resolve()
    return (_SCRIPT_DIR / "labeled_posts.sqlite").resolve()


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def load_examples(db: Path, *, max_posts: int | None, truncate_chars: int) -> list[LabeledExample]:
    conn = _connect_ro(db)
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT post_text, category, benign_label, label_1, label_2, label_3
        FROM "{LABELED_TABLE}"
        WHERE labeling_status = 'ok'
          AND LENGTH(TRIM(COALESCE(post_text, ''))) > 0
        """
    )
    rows: list[LabeledExample] = []
    for post_text, category, benign_label, l1, l2, l3 in cur.fetchall():
        text = str(post_text or "").replace("\r\n", "\n").strip()
        if not text:
            continue
        if truncate_chars > 0 and len(text) > truncate_chars:
            text = text[: truncate_chars - 1] + "…"
        ex = LabeledExample(
            post_text=text,
            category=(category or "").strip().lower(),
            benign_label=str(benign_label or "").strip(),
            label_1=str(l1 or "").strip(),
            label_2=str(l2 or "").strip(),
            label_3=str(l3 or "").strip(),
        )
        if ex.category == "benign":
            if not ex.benign_label:
                continue
        elif not (ex.label_1 or ex.label_2 or ex.label_3):
            continue
        rows.append(ex)
    conn.close()
    if max_posts is not None and len(rows) > max_posts:
        rng = random.Random(42)
        rows = rng.sample(rows, max_posts)
    return rows


def example_to_instruction(ex: LabeledExample) -> str:
    """User-side prompt: task + post."""
    intro = (
        "You label Telegram posts for a piracy/abuse taxonomy. "
        "Reply with ONLY a single JSON object, no markdown fences, no extra text."
    )
    return f"{intro}\n\n--- post ---\n{ex.post_text}\n"


def example_to_labels_json(ex: LabeledExample) -> str:
    """Assistant target: mirrors label_posts.parse_decision_dict."""
    if ex.category == "benign":
        obj = {"category": "benign", "benign_label": ex.benign_label}
    else:
        obj = {
            "category": "piracy",
            "label_1": ex.label_1,
            "label_2": ex.label_2,
            "label_3": ex.label_3,
        }
    return json.dumps(obj, ensure_ascii=False)


def split_train_eval(
    examples: list[LabeledExample],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[list[LabeledExample], list[LabeledExample]]:
    rng = random.Random(seed)
    order = examples[:]
    rng.shuffle(order)
    n_eval = max(1, min(len(order) - 1, int(round(len(order) * eval_fraction))))
    if len(order) <= 2:
        n_eval = 1 if len(order) == 2 else 0
    eval_set = order[:n_eval]
    train_set = order[n_eval:]
    if not train_set:
        train_set, eval_set = eval_set, eval_set[:1]
    return train_set, eval_set


def examples_to_hf_dataset(
    tokenizer: Any,
    train: list[LabeledExample],
    eval_: list[LabeledExample],
):
    """Build HuggingFace datasets with serialized chat `text`."""
    from datasets import Dataset

    def serialize_batch(batch_insts: list[str], batch_outs: list[str]) -> dict[str, list[str]]:
        texts: list[str] = []
        for inst, out in zip(batch_insts, batch_outs):
            msgs = [{"role": "user", "content": inst}, {"role": "assistant", "content": out}]
            texts.append(
                tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    tr_insts = [example_to_instruction(x) for x in train]
    tr_outs = [example_to_labels_json(x) for x in train]
    ev_insts = [example_to_instruction(x) for x in eval_]
    ev_outs = [example_to_labels_json(x) for x in eval_]

    train_ds = Dataset.from_dict(serialize_batch(tr_insts, tr_outs))
    eval_ds = Dataset.from_dict(serialize_batch(ev_insts, ev_outs))
    return train_ds, eval_ds


def build_training_arguments(TrainingArguments: Any, **kwargs: Any) -> Any:
    """Map TrainingArguments keywords when Transformers deprecates naming (eval_strategy ↔ evaluation_strategy)."""
    import inspect

    kw = dict(kwargs)
    params = inspect.signature(TrainingArguments.__init__).parameters
    if (
        "eval_strategy" in kw
        and "eval_strategy" not in params
        and "evaluation_strategy" in params
    ):
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    return TrainingArguments(**kw)


def instantiate_sft_trainer(
    SFTTrainer: Any,
    *,
    tokenizer: Any,
    tokenizer_kw_candidates: tuple[str, ...],
    kwargs: dict[str, Any],
) -> Any:
    import inspect

    sig = inspect.signature(SFTTrainer.__init__)
    param_names = set(sig.parameters.keys())
    base = {k: v for k, v in kwargs.items() if k in param_names}
    errs: list[BaseException] = []
    for key in tokenizer_kw_candidates:
        if key not in param_names:
            continue
        attempt = dict(base)
        attempt[key] = tokenizer
        try:
            return SFTTrainer(**attempt)
        except (TypeError, ValueError) as e:
            errs.append(e)

    stripped = dict(base)
    stripped.pop("max_seq_length", None)
    stripped.pop("packing", None)
    for key in tokenizer_kw_candidates:
        if key not in param_names:
            continue
        attempt = dict(stripped)
        attempt[key] = tokenizer
        try:
            return SFTTrainer(**attempt)
        except (TypeError, ValueError) as e:
            errs.append(e)

    hint = repr(errs[-1]) if errs else "(no tokenizer param matched SFTTrainer.__init__)"
    raise RuntimeError(f"Could not construct SFTTrainer (TRL API mismatch): {hint}")


def parse_args() -> argparse.Namespace:
    db_help = (
        "labeled_posts.sqlite (defaults: beside finetune.py, then cwd; use explicit path "
        "if you keep the DB elsewhere)."
    )
    p = argparse.ArgumentParser(description="Unsloth LoRA SFT on labeled_posts.sqlite.")
    p.add_argument(
        "--model",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct",
        help="HF id or local snapshot directory matching what you trained (offline: point at saved base).",
    )
    p.add_argument("--db", type=Path, default=None, help=db_help)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR / "unsloth_telegram_labels",
        help="Checkpoint directory (default: ./unsloth_telegram_labels in categorization_model).",
    )
    p.add_argument("--max-posts", type=int, default=None, help="Subsample rows (debug).")
    p.add_argument("--truncate-chars", type=int, default=12000)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--eval-fraction", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--no-4bit", action="store_true", help="Load full precision / default dtype.")

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.db = resolve_database_path(args.db)
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.db.is_file():
        searched = ", ".join(str(p) for p in default_sqlite_candidates())
        raise SystemExit(f"SQLite DB not found at {args.db} (defaults try: {searched}).")

    try:
        import torch
        from transformers import TrainingArguments

        import unsloth  # noqa: F401 — registers patches
        from trl import SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise SystemExit(
            "Missing deps. GPU + CUDA: pip install unsloth && "
            "pip install transformers trl datasets accelerate peft bitsandbytes "
            f"({e})"
        )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — Unsloth training expects a NVIDIA GPU.")

    print(f"Loading examples from {args.db}...")
    examples = load_examples(args.db, max_posts=args.max_posts, truncate_chars=args.truncate_chars)
    if len(examples) < 10:
        raise SystemExit(
            f"Too few labeled rows after filters: {len(examples)} "
            "(need labeling_status='ok', non-empty post_text, valid benign/piracy fields)."
        )

    train_ex, eval_ex = split_train_eval(examples, eval_fraction=args.eval_fraction, seed=args.seed)
    print(f"Train examples: {len(train_ex)}, eval examples: {len(eval_ex)}")

    print(f"Loading model {args.model!r} (Unsloth FastLanguageModel)...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_4bit = not args.no_4bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None if load_4bit else dtype,
        load_in_4bit=load_4bit,
        token=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    train_ds, eval_ds = examples_to_hf_dataset(tokenizer, train_ex, eval_ex)

    save_strategy = "steps"
    eval_strategy = "steps" if len(eval_ds) >= 10 else "no"

    training_args_kwargs: dict[str, Any] = dict(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=args.seed,
        report_to=[],
        max_steps=-1,
    )
    if eval_strategy != "no":
        training_args_kwargs["eval_strategy"] = eval_strategy
        training_args_kwargs["eval_steps"] = max(args.save_steps, args.logging_steps)
        training_args_kwargs["per_device_eval_batch_size"] = max(1, args.batch_size // 2)

    training_args = build_training_arguments(TrainingArguments, **training_args_kwargs)

    sft_kw: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if eval_strategy != "no" else None,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    trainer = instantiate_sft_trainer(
        SFTTrainer,
        tokenizer=tokenizer,
        tokenizer_kw_candidates=("tokenizer", "processing_class"),
        kwargs=sft_kw,
    )

    print("Starting training...")
    trainer.train()
    print(f"Saving LoRA adapters to {args.output_dir}...")
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("Done.")


if __name__ == "__main__":
    main()
