import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Avoid tokenizers fork warning when DataLoader workers are used.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_UNSLOTH_AVAILABLE = True
_UNSLOTH_IMPORT_ERROR: BaseException | None = None

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import bitsandbytes as bnb
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from huggingface_hub import login
import config


# ---------------------------------------------------------------------------
# Prompt & dataset formatting (CSV -> text, one post per row)
# ---------------------------------------------------------------------------

def format_post_and_label(post_text: str, label: str) -> str:
    """One post + label suffix for causal-LM label prediction."""
    return f"{post_text}\n\n{label}"


# ---------------------------------------------------------------------------
# Dataset loading (single-post CSV)
# ---------------------------------------------------------------------------

def _resolve_data_path(data_path: str) -> Path:
    p = Path(data_path)
    if p.is_file():
        return p.resolve()
    root = Path(__file__).resolve().parent
    alt = root / data_path
    if alt.is_file():
        return alt.resolve()
    raise FileNotFoundError(f"Dataset not found: {data_path} (tried {p.resolve()} and {alt})")


def load_and_preprocess_dataset(data_path: str, test_size: float = 0.2):
    """
    Load CSV with columns ``text``, ``label`` and convert each row to:
      ``<post_text>\\n\\n<label>`` (same as ``finetune_channels_unsloth_llama.py``).
    """
    path = _resolve_data_path(data_path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file; got {path}")
    print(f"Loading dataset from: {path}")

    formatted_texts: list[str] = []
    labels_for_strat: list[str] = []
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV must have 'text' and 'label' columns; got {list(df.columns)}")

    n_raw = len(df)
    for _, row in df.iterrows():
        out = str(row["label"]).strip().lower()
        if out not in ("pirated", "benign"):
            continue
        post_text = str(row["text"]).strip()
        if not post_text:
            continue
        formatted_texts.append(format_post_and_label(post_text, out))
        labels_for_strat.append(out)

    print(f"Loaded {n_raw:,} CSV rows -> {len(formatted_texts):,} valid")

    if not formatted_texts:
        raise ValueError("No valid data found after preprocessing.")

    idx = np.arange(len(formatted_texts))
    strat = labels_for_strat if len(set(labels_for_strat)) > 1 else None
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=42, stratify=strat
    )
    train_dataset = Dataset.from_dict({"text": [formatted_texts[i] for i in train_idx]})
    test_dataset = Dataset.from_dict({"text": [formatted_texts[i] for i in test_idx]})

    print(f"Train samples: {len(train_dataset):,} (for k-fold CV)")
    print(f"Test samples:  {len(test_dataset):,} (held-out)")
    return train_dataset, test_dataset


# ---------------------------------------------------------------------------
# Tokenization — label-masking approach from finetune.py
# ---------------------------------------------------------------------------

def tokenize_dataset(dataset, tokenizer, max_length: int = 256):
    # Preserve supervision by always keeping the label suffix in sequence.
    pirated_tokens = tokenizer("pirated", add_special_tokens=False, return_tensors=None)["input_ids"]
    benign_tokens = tokenizer("benign", add_special_tokens=False, return_tensors=None)["input_ids"]

    def tokenize_fn(examples):
        out_input_ids, out_attention_mask, out_labels = [], [], []
        for full_text in examples["text"]:
            text = str(full_text)
            if "\n\n" in text:
                post_text, label_text = text.rsplit("\n\n", 1)
            else:
                post_text, label_text = text, "benign"
            label_text = label_text.strip().lower()
            if label_text not in ("pirated", "benign"):
                label_text = "benign"

            label_ids = pirated_tokens if label_text == "pirated" else benign_tokens
            sep_ids = tokenizer("\n\n", add_special_tokens=False, return_tensors=None)["input_ids"]
            suffix_ids = sep_ids + label_ids
            prefix_budget = max(1, max_length - len(suffix_ids))

            prefix_ids = tokenizer(
                post_text,
                add_special_tokens=False,
                truncation=True,
                max_length=prefix_budget,
                return_tensors=None,
            )["input_ids"]

            input_ids = prefix_ids + suffix_ids
            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]

            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(input_ids)
            label_start = len(input_ids) - len(label_ids)
            for i in range(label_start, len(input_ids)):
                tid = input_ids[i]
                if tid not in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
                    labels[i] = tid

            out_input_ids.append(input_ids)
            out_attention_mask.append(attention_mask)
            out_labels.append(labels)

        return {"input_ids": out_input_ids, "attention_mask": out_attention_mask, "labels": out_labels}

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def find_all_linear_names(model):
    import torch.nn as nn
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, (bnb.nn.Linear4bit, nn.Linear)):
            parts = name.split(".")
            names.add(parts[0] if len(parts) == 1 else parts[-1])
    names.discard("lm_head")
    return list(names)


def setup_model_and_tokenizer(
    model_path: str | None = None,
    use_4bit: bool = True,
    max_seq_length: int = 256,
):
    if model_path is None:
        model_path = config.MODEL_PATH

    hf_token = (os.environ.get("HF_TOKEN") or "").strip() or None
    if hf_token is None:
        cfg_t = getattr(config, "HF_TOKEN", None)
        if cfg_t and str(cfg_t).strip():
            hf_token = str(cfg_t).strip()

    use_unsloth = (
        getattr(config, "USE_UNSLOTH", True)
        and _UNSLOTH_AVAILABLE
        and torch.cuda.is_available()
    )
    if model_path.startswith("unsloth/") and not use_unsloth:
        raise ImportError(
            "config.MODEL_PATH is an Unsloth repo ID. Install Unsloth: "
            "`pip install unsloth unsloth_zoo` (and use a CUDA GPU), or set USE_UNSLOTH=False "
            "and use a non-unsloth model id."
        )

    print(f"Loading model: {model_path}")
    if use_unsloth:
        print("Using Unsloth FastLanguageModel (QLoRA)")

    if use_unsloth:
        from unsloth import FastLanguageModel

        dtype = getattr(config, "UNSLOTH_DTYPE", None)
        load_kw: dict = dict(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            token=hf_token,
            trust_remote_code=True,
        )
        if use_4bit:
            load_kw["load_in_4bit"] = True
        else:
            load_kw["load_in_4bit"] = False
            load_kw["load_in_16bit"] = True

        model, tokenizer = FastLanguageModel.from_pretrained(**load_kw)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        target_modules = getattr(
            config,
            "LORA_TARGET_MODULES",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=getattr(config, "LORA_R", 8),
            lora_alpha=getattr(config, "LORA_ALPHA", 16),
            target_modules=target_modules,
            lora_dropout=getattr(config, "LORA_DROPOUT", 0.1),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        model.print_trainable_parameters()
        return model, tokenizer, None

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    if use_4bit and torch.cuda.is_available():
        if hasattr(model, "config"):
            model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)

    modules = find_all_linear_names(model)
    print(f"LoRA target modules ({len(modules)}): {modules}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    # Explicit non-reentrant checkpointing avoids upcoming PyTorch default warning.
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            # Fallback for older transformers versions.
            model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model, tokenizer, lora_config


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

_PIRATED_TOKEN_IDS: set[int] | None = None
_BENIGN_TOKEN_IDS: set[int] | None = None


def init_label_token_ids(tokenizer):
    """Pre-compute token IDs for 'pirated' and 'benign' — call once before training."""
    global _PIRATED_TOKEN_IDS, _BENIGN_TOKEN_IDS
    _PIRATED_TOKEN_IDS = set(tokenizer("pirated", add_special_tokens=False)["input_ids"])
    _BENIGN_TOKEN_IDS = set(tokenizer("benign", add_special_tokens=False)["input_ids"])


def extract_class_from_label_tokens(tokenizer, label_tokens):
    """Map predicted/true label tokens to binary class: 0=pirated, 1=benign.
    Uses pre-computed token ID sets for speed (no decoding)."""
    if not label_tokens:
        return 1
    token_set = set(label_tokens)
    if _PIRATED_TOKEN_IDS and token_set & _PIRATED_TOKEN_IDS:
        return 0
    if _BENIGN_TOKEN_IDS and token_set & _BENIGN_TOKEN_IDS:
        return 1
    return 1


# ---------------------------------------------------------------------------
# Training loop (single fold)
# ---------------------------------------------------------------------------

def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    output_dir: str = "./llama-finetune",
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 2,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_length: int = 256,
    early_stopping_patience: int = 3,
    label_smoothing: float = 0.0,
    max_grad_norm: float = 1.0,
    dataloader_workers: int = 0,
    log_every_steps: int = 500,
    step_log_jsonl: bool = False,
    log_val_misclassified: bool = True,
    max_val_misclassified: int = 25,
):
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        max_len = min(max(len(s) for s in input_ids), max_length)

        padded_ids, padded_labels, masks = [], [], []
        for seq, lab in zip(input_ids, labels):
            seq, lab = seq[:max_len], lab[:max_len]
            pad = max_len - len(seq)
            padded_ids.append(seq + [tokenizer.pad_token_id] * pad)
            padded_labels.append(lab + [-100] * pad)
            masks.append([1] * len(seq) + [0] * pad)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    train_loader_kw = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=dataloader_workers,
    )
    eval_loader_kw = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=dataloader_workers,
    )
    if torch.cuda.is_available():
        train_loader_kw["pin_memory"] = True
        eval_loader_kw["pin_memory"] = True
    train_loader = DataLoader(train_dataset, shuffle=True, **train_loader_kw)
    eval_loader = (
        DataLoader(eval_dataset, shuffle=False, **eval_loader_kw)
        if eval_dataset
        else None
    )

    if torch.cuda.is_available():
        try:
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            print("Using 8-bit AdamW")
        except ImportError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=learning_rate * 0.1),
        ],
        milestones=[warmup_steps],
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
    os.makedirs(output_dir, exist_ok=True)

    step_log_path = (
        os.path.join(output_dir, "training_steps.jsonl") if step_log_jsonl and log_every_steps > 0 else None
    )
    if step_log_path:
        print(f"Step logs (JSONL): {step_log_path} every {log_every_steps} optimizer step(s)")

    train_prec_m = BinaryPrecision().to(device)
    train_rec_m = BinaryRecall().to(device)
    train_f1_m = BinaryF1Score().to(device)
    val_prec_m = BinaryPrecision().to(device)
    val_rec_m = BinaryRecall().to(device)
    val_f1_m = BinaryF1Score().to(device)

    training_history = []
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = None
    best_checkpoint = None
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        train_prec_m.reset(); train_rec_m.reset(); train_f1_m.reset()
        val_prec_m.reset(); val_rec_m.reset(); val_f1_m.reset()

        # --- train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_pred_cls, train_true_cls = [], []
        optimizer.zero_grad()

        # Window for step-level logging (micro-batches + grad norms per optimizer step)
        win_loss_sum, win_loss_batches = 0.0, 0
        win_tok_correct, win_tok_total = 0, 0
        win_label_correct, win_label_total = 0, 0
        win_grad_norms: list[float] = []

        def _flush_step_log(*, force: bool = False) -> None:
            nonlocal win_loss_sum, win_loss_batches, win_tok_correct, win_tok_total, win_label_correct, win_label_total, win_grad_norms
            if log_every_steps <= 0 and not force:
                return
            if win_loss_batches == 0 and not win_grad_norms:
                return
            avg_loss = win_loss_sum / max(win_loss_batches, 1)
            tok_acc = win_tok_correct / win_tok_total if win_tok_total else 0.0
            label_acc = win_label_correct / win_label_total if win_label_total else 0.0
            avg_gn = sum(win_grad_norms) / len(win_grad_norms) if win_grad_norms else 0.0
            last_gn = win_grad_norms[-1] if win_grad_norms else 0.0
            lr_now = optimizer.param_groups[0]["lr"]
            tag = "[step]" if not force else "[step_end]"
            line = (
                f"{tag} global_step={global_step} epoch={epoch + 1}/{num_epochs} "
                f"loss_avg={avg_loss:.4f} label_acc={label_acc * 100:.2f}% "
                f"(tok_acc={tok_acc * 100:.2f}%) "
                f"grad_norm_avg={avg_gn:.4f} grad_norm_last={last_gn:.4f} lr={lr_now:.2e}"
            )
            print(line)
            if step_log_path:
                rec = {
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    "loss_avg_microbatch": avg_loss,
                    "label_acc_window": label_acc,
                    "token_acc_window": tok_acc,
                    "grad_norm_avg": avg_gn,
                    "grad_norm_last": last_gn,
                    "learning_rate": lr_now,
                    "partial_epoch_flush": force,
                }
                with open(step_log_path, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps(rec) + "\n")
            win_loss_sum = 0.0
            win_loss_batches = 0
            win_tok_correct = 0
            win_tok_total = 0
            win_label_correct = 0
            win_label_total = 0
            win_grad_norms = []

        for batch_idx, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            labs = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)

            logits = model(input_ids=ids, attention_mask=attn).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labs[..., 1:].contiguous()

            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            preds = shift_logits.argmax(dim=-1)
            mask = shift_labels != -100
            train_correct += ((preds == shift_labels) & mask).sum().item()
            train_total += mask.sum().item()
            if log_every_steps > 0:
                win_loss_sum += loss.item()
                win_loss_batches += 1
                win_tok_correct += ((preds == shift_labels) & mask).sum().item()
                win_tok_total += mask.sum().item()
                with torch.no_grad():
                    for si in range(ids.size(0)):
                        lm = shift_labels[si] != -100
                        if lm.sum() == 0:
                            continue
                        pc = extract_class_from_label_tokens(
                            tokenizer, preds[si][lm].cpu().tolist()
                        )
                        tc = extract_class_from_label_tokens(
                            tokenizer, shift_labels[si][lm].cpu().tolist()
                        )
                        win_label_total += 1
                        if pc == tc:
                            win_label_correct += 1

            (loss / gradient_accumulation_steps).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                grad_norm = float(grad_norm_t.detach().cpu()) if torch.is_tensor(grad_norm_t) else float(grad_norm_t)
                if log_every_steps > 0:
                    win_grad_norms.append(grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if log_every_steps > 0 and global_step % log_every_steps == 0:
                    _flush_step_log(force=False)

            train_loss += loss.item()

            with torch.no_grad():
                for si in range(ids.size(0)):
                    lm = shift_labels[si] != -100
                    if lm.sum() == 0:
                        continue
                    train_pred_cls.append(extract_class_from_label_tokens(tokenizer, preds[si][lm].cpu().tolist()))
                    train_true_cls.append(extract_class_from_label_tokens(tokenizer, shift_labels[si][lm].cpu().tolist()))

            del logits, shift_logits, shift_labels, preds, mask

        if log_every_steps > 0 and (win_loss_batches > 0 or win_grad_norms):
            _flush_step_log(force=True)

        avg_train_loss = train_loss / len(train_loader)
        train_token_acc = train_correct / train_total if train_total else 0.0
        train_label_acc = 0.0

        train_precision = train_recall = train_f1 = None
        if train_pred_cls:
            pt = torch.tensor(train_pred_cls, dtype=torch.long, device=device)
            tt = torch.tensor(train_true_cls, dtype=torch.long, device=device)
            train_prec_m.update(pt, tt)
            train_rec_m.update(pt, tt)
            train_f1_m.update(pt, tt)
            train_precision = train_prec_m.compute().item()
            train_recall = train_rec_m.compute().item()
            train_f1 = train_f1_m.compute().item()
            # Label-level accuracy (matches how P/R/F1 is computed)
            correct_labels = sum(1 for p, t in zip(train_pred_cls, train_true_cls) if p == t)
            train_label_acc = correct_labels / len(train_true_cls) if train_true_cls else 0.0
        else:
            train_precision = train_recall = train_f1 = None

        # --- eval ---
        avg_val_loss = None
        val_token_acc = None
        val_label_acc = 0.0
        val_precision = val_recall = val_f1 = None
        if eval_loader:
            model.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0
            val_pred_cls, val_true_cls = [], []
            val_misclassified_records: list[dict] = []
            val_mis_count = 0
            val_seen = 0
            with torch.no_grad():
                for batch in eval_loader:
                    ids = batch["input_ids"].to(device)
                    labs = batch["labels"].to(device)
                    attn = batch["attention_mask"].to(device)

                    logits = model(input_ids=ids, attention_mask=attn).logits
                    sl = logits[..., :-1, :].contiguous()
                    slabs = labs[..., 1:].contiguous()

                    loss = criterion(sl.view(-1, sl.size(-1)), slabs.view(-1))
                    preds = sl.argmax(dim=-1)
                    m = slabs != -100
                    val_correct += ((preds == slabs) & m).sum().item()
                    val_total += m.sum().item()
                    val_loss_sum += loss.item()

                    for si in range(ids.size(0)):
                        lm = slabs[si] != -100
                        if lm.sum() == 0:
                            val_seen += 1
                            continue
                        pred_cls = extract_class_from_label_tokens(tokenizer, preds[si][lm].cpu().tolist())
                        true_cls = extract_class_from_label_tokens(tokenizer, slabs[si][lm].cpu().tolist())
                        val_pred_cls.append(pred_cls)
                        val_true_cls.append(true_cls)
                        # Misclassification logging (for validation debugging)
                        if log_val_misclassified and pred_cls != true_cls and val_mis_count < max_val_misclassified:
                            decoded = tokenizer.decode(ids[si].cpu().tolist(), skip_special_tokens=True).strip()
                            # Strip label suffix so logs show only the post text
                            msg = decoded
                            if "\n\n" in decoded:
                                parts = decoded.rsplit("\n\n", 1)
                                if len(parts) == 2 and parts[1].strip().lower() in ("pirated", "benign"):
                                    msg = parts[0].strip()
                            val_misclassified_records.append({
                                "index": val_seen,
                                "text": msg,
                                "true_label": "pirated" if true_cls == 0 else "benign",
                                "predicted_label": "pirated" if pred_cls == 0 else "benign",
                            })
                            val_mis_count += 1
                        val_seen += 1

                    del logits, sl, slabs, preds, m, loss

            avg_val_loss = val_loss_sum / len(eval_loader)
            val_token_acc = val_correct / val_total if val_total else 0.0

            if val_pred_cls:
                pt = torch.tensor(val_pred_cls, dtype=torch.long, device=device)
                tt = torch.tensor(val_true_cls, dtype=torch.long, device=device)
                val_prec_m.update(pt, tt); val_rec_m.update(pt, tt); val_f1_m.update(pt, tt)
                val_precision = val_prec_m.compute().item()
                val_recall = val_rec_m.compute().item()
                val_f1 = val_f1_m.compute().item()

                # Label-level accuracy (this matches P/R/F1 computation)
                correct_labels = sum(1 for p, t in zip(val_pred_cls, val_true_cls) if p == t)
                val_label_acc = correct_labels / len(val_true_cls) if val_true_cls else 0.0
            else:
                val_label_acc = 0.0

            # Write misclassified logs (one file per epoch, inside fold output_dir)
            if log_val_misclassified and val_misclassified_records:
                mis_path = os.path.join(output_dir, f"val_misclassified_epoch_{epoch + 1}.jsonl")
                with open(mis_path, "w", encoding="utf-8") as f:
                    for rec in val_misclassified_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"Validation misclassifications saved: {mis_path} ({len(val_misclassified_records)} examples)")

        cur_lr = optimizer.param_groups[0]["lr"]
        epoch_hist = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            # Align "acc" with precision/recall/F1 (label-level)
            "train_acc": train_label_acc,
            "train_token_acc": train_token_acc,
            "val_loss": avg_val_loss,
            # Align "acc" with precision/recall/F1 (label-level)
            "val_acc": val_label_acc,
            "val_token_acc": val_token_acc,
            "val_label_acc": val_label_acc,
            "learning_rate": cur_lr,
        }
        if train_precision is not None:
            epoch_hist["train_precision"] = train_precision
            epoch_hist["train_recall"] = train_recall
            epoch_hist["train_f1"] = train_f1
        if val_precision is not None:
            epoch_hist["val_precision"] = val_precision
            epoch_hist["val_recall"] = val_recall
            epoch_hist["val_f1"] = val_f1
        training_history.append(epoch_hist)

        # "Acc" should correspond to label-level accuracy
        line = (
            f"Train Loss: {avg_train_loss:.4f} | "
            f"TokenAcc: {train_token_acc * 100:.2f}% | "
            f"LabelAcc: {train_label_acc * 100:.2f}%"
        )
        if train_precision is not None:
            line += f" | P: {train_precision:.4f} R: {train_recall:.4f} F1: {train_f1:.4f}"
        print(line)

        if avg_val_loss is not None:
            line = (
                f"Val   Loss: {avg_val_loss:.4f} | "
                f"TokenAcc: {val_token_acc * 100.0:.2f}% | "
                f"LabelAcc: {val_label_acc * 100.0:.2f}%"
            )
            if val_precision is not None:
                line += f" | P: {val_precision:.4f} R: {val_recall:.4f} F1: {val_f1:.4f}"
            print(line)
        print(f"LR: {cur_lr:.2e}")

        # Checkpoint
        ckpt = os.path.join(output_dir, f"checkpoint-{epoch + 1}")
        os.makedirs(ckpt, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

        if avg_val_loss is not None:
            current_f1 = val_f1 if val_f1 is not None else 0.0
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_checkpoint = ckpt
                epochs_without_improvement = 0
                print(f"** New best (epoch {best_epoch}, val_f1={best_val_f1:.4f}, val_loss={best_val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s) (best F1: {best_val_f1:.4f})")
                if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}.")
                    break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _keep_best_checkpoint(output_dir, best_checkpoint, best_epoch, tokenizer)

    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    return model, training_history


def _keep_best_checkpoint(output_dir, best_checkpoint, best_epoch, tokenizer):
    ckpt_dirs = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]

    if best_checkpoint and os.path.exists(best_checkpoint):
        with tempfile.TemporaryDirectory() as tmp:
            for item in os.listdir(best_checkpoint):
                src = os.path.join(best_checkpoint, item)
                dst = os.path.join(tmp, item)
                (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)

            for d in ckpt_dirs:
                shutil.rmtree(d, ignore_errors=True)

            for item in os.listdir(tmp):
                src = os.path.join(tmp, item)
                dst = os.path.join(output_dir, item)
                (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)

        tokenizer.save_pretrained(output_dir)
        print(f"Best checkpoint (epoch {best_epoch}) saved to {output_dir}")
    else:
        for d in ckpt_dirs:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3-fold cross-validation
# ---------------------------------------------------------------------------

def perform_kfold_cv(
    dataset,
    tokenizer,
    base_output_dir: str,
    n_folds: int = 3,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 2,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    use_4bit: bool = True,
    early_stopping_patience: int = 3,
    label_smoothing: float = 0.0,
    max_grad_norm: float = 1.0,
    max_length: int = 256,
    dataloader_workers: int = 0,
    log_every_steps: int = 500,
    step_log_jsonl: bool = False,
):
    indices = np.arange(len(dataset))
    if n_folds == 1:
        # Single run: use the full dataset for both train and val (no real CV)
        splits = [(indices, indices)]
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(kfold.split(indices))

    fold_histories = []
    fold_val_losses = []
    fold_val_accs = []
    fold_val_f1s = []
    fold_times = []
    cv_start = time.time()

    print(f"\nStarting {n_folds}-fold cross-validation ({len(dataset):,} samples)")

    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n{'=' * 80}")
        print(f"Fold {fold_idx}/{n_folds}  (train: {len(train_idx):,}, val: {len(val_idx):,})")
        print(f"{'=' * 80}")

        train_fold = dataset.select(train_idx.tolist())
        val_fold = dataset.select(val_idx.tolist())

        model, tok, _ = setup_model_and_tokenizer(use_4bit=use_4bit, max_seq_length=max_length)
        init_label_token_ids(tok)
        train_tok = tokenize_dataset(train_fold, tok, max_length=max_length)
        val_tok = tokenize_dataset(val_fold, tok, max_length=max_length)

        fold_dir = os.path.join(base_output_dir, f"fold_{fold_idx}")
        fold_start = time.time()

        model, history = train_model(
            model=model,
            tokenizer=tok,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            output_dir=fold_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            max_length=max_length,
            early_stopping_patience=early_stopping_patience,
            label_smoothing=label_smoothing,
            max_grad_norm=max_grad_norm,
            dataloader_workers=dataloader_workers,
            log_every_steps=log_every_steps,
            step_log_jsonl=step_log_jsonl,
        )

        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        fold_histories.append(history)

        if history:
            best_ep = max(history, key=lambda h: h.get("val_f1", -1)) if any(h.get("val_f1") for h in history) else history[-1]
            vl = best_ep.get("val_loss", float("inf"))
            va = best_ep.get("val_acc")
            vp = best_ep.get("val_precision")
            vr = best_ep.get("val_recall")
            vf = best_ep.get("val_f1")
            fold_val_losses.append((fold_idx, vl, fold_dir))
            if va is not None:
                fold_val_accs.append((fold_idx, va))
            if vf is not None:
                fold_val_f1s.append((fold_idx, vf, fold_dir))
            h, m, s = int(fold_time // 3600), int((fold_time % 3600) // 60), int(fold_time % 60)
            line = f"\nFold {fold_idx} done in {h}h {m}m {s}s  |  best_val_f1={vf:.4f}  val_loss={vl:.4f}"
            if va is not None:
                line += f"  acc={va * 100:.2f}%"
            if vp is not None:
                line += f"  P={vp:.4f} R={vr:.4f} F1={vf:.4f}"
            print(line)

        del model, tok
        torch.cuda.empty_cache()

    # Summary
    cv_total = time.time() - cv_start
    h, m, s = int(cv_total // 3600), int((cv_total % 3600) // 60), int(cv_total % 60)
    print(f"\n{'=' * 80}")
    print(f"Cross-Validation Summary  (total time: {h}h {m}m {s}s)")
    print(f"{'=' * 80}")

    for fi, vl, _ in fold_val_losses:
        va = next((a for i, a in fold_val_accs if i == fi), None)
        vf = next((f for i, f, _ in fold_val_f1s if i == fi), None)
        line = f"  Fold {fi}: loss={vl:.4f}"
        if va is not None:
            line += f"  acc={va * 100:.2f}%"
        if vf is not None:
            line += f"  F1={vf:.4f}"
        print(line)

    if fold_val_f1s:
        best_fold_idx, best_f1, best_dir = max(fold_val_f1s, key=lambda x: x[1])
        best_loss = next((l for i, l, _ in fold_val_losses if i == best_fold_idx), None)
    else:
        best_fold_idx, best_loss, best_dir = min(fold_val_losses, key=lambda x: x[1])
        best_f1 = None
    best_model_dir = os.path.join(base_output_dir, "best_model")
    if os.path.exists(best_model_dir):
        shutil.rmtree(best_model_dir)
    shutil.copytree(best_dir, best_model_dir)
    sel_msg = f"val_f1={best_f1:.4f}" if best_f1 is not None else f"val_loss={best_loss:.4f}"
    print(f"\nBest model: fold {best_fold_idx} ({sel_msg})")
    print(f"Saved to:   {best_model_dir}")

    visualize_training(fold_histories, base_output_dir)

    return {
        "best_fold": best_fold_idx,
        "best_f1": best_f1,
        "best_loss": best_loss,
        "fold_val_losses": {f"fold_{i}": l for i, l, _ in fold_val_losses},
        "fold_val_accs": {f"fold_{i}": a for i, a in fold_val_accs},
        "fold_val_f1s": {f"fold_{i}": f for i, f, _ in fold_val_f1s},
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_training(fold_histories, output_dir):
    if not fold_histories:
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 15))

    for fold_idx, hist in enumerate(fold_histories, 1):
        if not hist:
            continue
        epochs = [h["epoch"] for h in hist]

        axes[0, 0].plot(epochs, [h["train_loss"] for h in hist], label=f"Fold {fold_idx}", linewidth=2, alpha=0.8)

        vl = [(h["epoch"], h["val_loss"]) for h in hist if h.get("val_loss") is not None]
        if vl:
            e, l = zip(*vl)
            axes[0, 1].plot(e, l, label=f"Fold {fold_idx}", linewidth=2, alpha=0.8)

        vp = [(h["epoch"], h["val_precision"]) for h in hist if h.get("val_precision") is not None]
        if vp:
            e, v = zip(*vp)
            axes[1, 0].plot(e, v, label=f"Fold {fold_idx}", linewidth=2, alpha=0.8)

        vr = [(h["epoch"], h["val_recall"]) for h in hist if h.get("val_recall") is not None]
        if vr:
            e, v = zip(*vr)
            axes[1, 1].plot(e, v, label=f"Fold {fold_idx}", linewidth=2, alpha=0.8)

        va = [(h["epoch"], h["val_acc"]) for h in hist if h.get("val_acc") is not None]
        if va:
            e, a = zip(*va)
            axes[2, 0].plot(e, a, label=f"Fold {fold_idx}", linewidth=2, alpha=0.8)

        vf = [(h["epoch"], h["val_f1"]) for h in hist if h.get("val_f1") is not None]
        if vf:
            e, v = zip(*vf)
            axes[2, 1].plot(e, v, label=f"Fold {fold_idx}", linewidth=2, alpha=0.8)

    titles = [
        "Training Loss", "Validation Loss",
        "Validation Precision", "Validation Recall",
        "Validation Accuracy", "Validation F1 Score",
    ]
    ylabels = [
        "Loss", "Loss",
        "Precision", "Recall",
        "Accuracy", "F1",
    ]
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training metrics plot saved to: {plot_path}")

    json_path = os.path.join(output_dir, "training_histories.json")
    with open(json_path, "w") as f:
        json.dump(fold_histories, f, indent=2)
    print(f"Training histories saved to: {json_path}")

    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    if fold_histories and fold_histories[0]:
        epochs = [h["epoch"] for h in fold_histories[0]]
        lrs = [h["learning_rate"] for h in fold_histories[0]]
        ax.plot(epochs, lrs, "g-", label="Learning Rate", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    lr_path = os.path.join(output_dir, "learning_rate.png")
    plt.savefig(lr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Learning rate plot saved to: {lr_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    hf_token = (os.environ.get("HF_TOKEN") or "").strip() or None
    if hf_token is None and getattr(config, "HF_TOKEN", None):
        t = str(config.HF_TOKEN).strip()
        hf_token = t or None
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
            print("Authenticated with HuggingFace Hub")
            os.environ.setdefault("HF_TOKEN", hf_token)
        except Exception as e:
            print(f"HuggingFace login failed: {e}")
    else:
        print("No HF_TOKEN set (export HF_TOKEN=... for gated models)")

    DATASET_PATH = "path_to_dataset"
    OUTPUT_DIR = "./llama-finetune"
    NUM_EPOCHS = 10
    BATCH_SIZE = 8
    GRADIENT_ACCUM = 4
    LEARNING_RATE = 1e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    TEST_SPLIT = 0.2
    EARLY_STOPPING = 3
    MAX_LENGTH = 256  # single-post rows; match finetune_channels_unsloth_llama.py
    n_folds = 3
    DATALOADER_WORKERS = 0  # single-process loading (no worker forking)
    # Fewer lines on huge datasets (each step = one optimizer update after grad accum).
    LOG_EVERY_STEPS = 500  # set lower (e.g. 50–200) for more detail; 0 = off
    STEP_LOG_JSONL = False  # print only; no training_steps.jsonl output

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected -- training will be slow on CPU")
    print("\n" + "=" * 80)
    print("Fine-tuning Configuration")
    print("=" * 80)
    print(f"  Dataset:          {DATASET_PATH}")
    print(f"  Output:           {OUTPUT_DIR}")
    print(f"  Strategy:         {n_folds}-fold cross-validation")
    print(f"  Epochs:           {NUM_EPOCHS}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Effective batch:  {BATCH_SIZE * GRADIENT_ACCUM}")
    print(f"  Learning rate:    {LEARNING_RATE}")
    print(f"  Warmup ratio:     {WARMUP_RATIO}")
    print(f"  Weight decay:     {WEIGHT_DECAY}")
    print(f"  Max seq length:   {MAX_LENGTH}")
    print(f"  DataLoader workers: {DATALOADER_WORKERS}")
    print(f"  Log every steps:  {LOG_EVERY_STEPS} (0 = off)")
    print(f"  Step log JSONL:   {STEP_LOG_JSONL} (disabled)")
    print(f"  Early stopping:   {EARLY_STOPPING} epochs patience")
    print(f"  Test split:       {TEST_SPLIT}")
    use_u = getattr(config, "USE_UNSLOTH", True) and _UNSLOTH_AVAILABLE and torch.cuda.is_available()
    print(
        f"  Unsloth:          {'yes (FastLanguageModel)' if use_u else 'no (CUDA required for Unsloth path; falls back to HF+PEFT)'}"
    )
    print("=" * 80 + "\n")

    train_dataset, test_dataset = load_and_preprocess_dataset(DATASET_PATH, test_size=TEST_SPLIT)
    print(f"\nTrain: {len(train_dataset):,}  |  Test (held-out): {len(test_dataset):,}")
    if n_folds > 1:
        print(f"Each fold: ~{int(len(train_dataset) * (1 - 1/n_folds)):,} train, ~{int(len(train_dataset) / n_folds):,} val\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_path = os.path.join(OUTPUT_DIR, "test_dataset.json")
    test_dataset.to_json(test_path)
    print(f"Test dataset saved to: {test_path}")

    _, tokenizer, _ = setup_model_and_tokenizer(
        use_4bit=torch.cuda.is_available(),
        max_seq_length=MAX_LENGTH,
    )

    cv_results = perform_kfold_cv(
        dataset=train_dataset,
        tokenizer=tokenizer,
        base_output_dir=OUTPUT_DIR,
        n_folds=n_folds,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=GRADIENT_ACCUM,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        use_4bit=torch.cuda.is_available(),
        early_stopping_patience=EARLY_STOPPING,
        max_grad_norm=1.0,
        max_length=MAX_LENGTH,
        dataloader_workers=DATALOADER_WORKERS,
        log_every_steps=LOG_EVERY_STEPS,
        step_log_jsonl=STEP_LOG_JSONL,
    )

    cv_path = os.path.join(OUTPUT_DIR, "cross_validation_results.json")
    with open(cv_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"CV results saved to: {cv_path}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"  Train set:    {len(train_dataset):,} samples")
    print(f"  Test set:     {len(test_dataset):,} samples (held-out)")
    if cv_results.get("best_fold"):
        bf1 = cv_results.get("best_f1")
        bl = cv_results.get("best_loss")
        sel = f"val_f1={bf1:.4f}" if bf1 is not None else f"val_loss={bl:.4f}"
        print(f"  Best model:   fold {cv_results['best_fold']} ({sel})")
    print(f"  Output dir:   {OUTPUT_DIR}")
    print(f"  Best model:   {OUTPUT_DIR}/best_model")
    print(f"  Test dataset: {test_path}")
    print(f"  Plots:        {OUTPUT_DIR}/training_metrics.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
