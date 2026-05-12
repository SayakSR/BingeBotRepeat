"""
Evaluation script for fine-tuned causal-LM channel classifier (pirated vs benign).

At inference, the script passes ``<single post text>\\n\\n`` and lets the model
generate ``pirated`` or ``benign``.

Computes: accuracy, precision, recall, F1 (per-class, macro, weighted), confusion matrix.
"""

import os
import json
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.llama import LlamaTokenizer
from peft import PeftModel
import config


def format_test_prompt(message: str) -> str:
    """
    Format a single-post prompt for inference.
    
    Args:
        message (str): Input post text without the label suffix.
    
    Returns:
        str: Prompt ending with "\\n\\n" so the model completes with class label.
    """
    return f"{message}\n\n"


def load_finetuned_model(
    base_model_path: str = None,
    finetuned_path: str = "path to best model",
    use_4bit: bool = None,
):
    """
    Load the fine-tuned model with LoRA adapters.
    
    Args:
        base_model_path (str): Path to the base model. If None, uses config.MODEL_PATH.
        finetuned_path (str): Path to the fine-tuned model directory.
        use_4bit (bool): Whether to use 4-bit quantization. If None, auto-detects.
    
    Returns:
        tuple: (model, tokenizer) loaded and ready for inference.
    """
    if not os.path.exists(finetuned_path):
        raise FileNotFoundError(
            f"Fine-tuned model directory not found: {finetuned_path}\n"
            f"Please run finetune.py first to create the fine-tuned model."
        )
    
    if base_model_path is None:
        # Prefer adapter metadata when available so eval stays aligned with the checkpoint.
        adapter_cfg = os.path.join(finetuned_path, "adapter_config.json")
        if os.path.exists(adapter_cfg):
            try:
                with open(adapter_cfg, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                base_model_path = cfg.get("base_model_name_or_path") or config.MODEL_PATH
            except Exception:
                base_model_path = config.MODEL_PATH
        else:
            base_model_path = config.MODEL_PATH
    
    if use_4bit is None:
        use_4bit = torch.cuda.is_available()
    
    hf_token = config.HF_TOKEN if config.HF_TOKEN and config.HF_TOKEN.strip() else None
    
    # Robust tokenizer loading (some environments/models can return unexpected objects).
    tokenizer = None
    tok_errors = []
    tok_candidates = [
        (finetuned_path, {"use_fast": True}),
        (finetuned_path, {"use_fast": False}),
        (base_model_path, {"use_fast": True}),
        (base_model_path, {"use_fast": False}),
    ]
    for source, kw in tok_candidates:
        try:
            cand = AutoTokenizer.from_pretrained(
                source,
                trust_remote_code=True,
                token=hf_token,
                **kw,
            )
            if hasattr(cand, "pad_token") and hasattr(cand, "eos_token_id"):
                tokenizer = cand
                break
            tok_errors.append(f"{source} ({kw}) -> invalid tokenizer type: {type(cand)}")
        except Exception as e:
            tok_errors.append(f"{source} ({kw}) -> {e!r}")

    if tokenizer is None:
        # Final fallback for legacy llama tokenizer edge cases.
        try:
            tokenizer = LlamaTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                token=hf_token,
            )
        except Exception as e:
            tok_errors.append(f"LlamaTokenizer fallback -> {e!r}")
            raise RuntimeError(
                "Failed to load a valid tokenizer.\n" + "\n".join(tok_errors)
            ) from e
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    model = PeftModel.from_pretrained(base_model, finetuned_path)
    model.eval()
    
    return model, tokenizer


def predict_label(model, tokenizer, message: str, max_new_tokens: int = 20) -> str:
    """
    Predict the label for a message.
    
    Args:
        model: The fine-tuned model.
        tokenizer: The tokenizer.
        message (str): The message to classify.
        max_new_tokens (int): Maximum tokens to generate.
    
    Returns:
        str: The predicted label ('pirated' or 'benign').
    """
    prompt = format_test_prompt(message)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Prompt is "<post text>\\n\\n"; model generates "pirated" or "benign".
    if prompt in decoded:
        response = decoded.replace(prompt, "", 1).strip()
        response_lower = response.lower().split()[0] if response.split() else response.lower()
        if "pirated" in response_lower:
            return "pirated"
        if "benign" in response_lower:
            return "benign"
    response_lower = decoded.lower()
    if "pirated" in response_lower:
        return "pirated"
    if "benign" in response_lower:
        return "benign"
    return "benign"


def _parse_item(obj):
    """
    From one record, get (message, label) or (None, None).
    Expected:
      - {"text": "<single post text>\\n\\nlabel"}
    """
    if not isinstance(obj, dict):
        return None, None
    if "text" in obj:
        text = obj["text"]
        if not text or not isinstance(text, str):
            return None, None
        last_double = text.rfind("\n\n")
        if last_double == -1:
            return None, None
        message = text[:last_double].strip()
        label = text[last_double + 2 :].strip().lower()
        if not message or label not in ("pirated", "benign"):
            return None, None
        return message, label
    return None, None


def _read_jsonl(file_handle):
    """Read JSONL: one JSON object per line. Yields (message, label) from _parse_item."""
    for line in file_handle:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            msg, lab = _parse_item(obj)
            if msg is not None and lab is not None:
                yield msg, lab
        except json.JSONDecodeError:
            continue


def load_test_data(test_dataset_path: str, max_samples: int = None):
    """
    Load test data from JSON/JSONL produced by finetune_channels_unsloth.py.
    Expected records:
      - {"text": "<single post text>\\n\\nlabel"}
    Also supports single JSON object {"text": [...]}.
    
    Returns:
        tuple: (messages, labels) where message is post text (label removed).
    """
    if not os.path.exists(test_dataset_path):
        raise FileNotFoundError(
            f"Test dataset not found at: {test_dataset_path}\n"
            f"Run finetune_channels.py first to create the test dataset."
        )
    print(f"Loading test dataset from: {test_dataset_path}")
    messages = []
    labels = []
    with open(test_dataset_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict) and "text" in data and isinstance(data["text"], list):
                for text in data["text"]:
                    msg, lab = _parse_item({"text": text})
                    if msg is not None and lab is not None:
                        messages.append(msg)
                        labels.append(lab)
            elif isinstance(data, list):
                for row in data:
                    msg, lab = _parse_item(row)
                    if msg is not None and lab is not None:
                        messages.append(msg)
                        labels.append(lab)
            else:
                raise ValueError("Single JSON must be {\"text\": [...]} or a list of {\"text\": \"...\"} objects.")
        except json.JSONDecodeError:
            f.seek(0)
            for msg, lab in _read_jsonl(f):
                messages.append(msg)
                labels.append(lab)
    if max_samples is not None and len(messages) > max_samples:
        messages = messages[:max_samples]
        labels = labels[:max_samples]
    print(f"Loaded {len(messages)} test samples\n")
    return messages, labels


def evaluate_model(model, tokenizer, messages, true_labels):
    """
    Evaluate the model on test data and compute classification metrics.
    
    Args:
        model: The fine-tuned model.
        tokenizer: The tokenizer.
        messages (list): List of test messages.
        true_labels (list): List of true labels.
    
    Returns:
        dict: Dictionary containing evaluation metrics.
        list[str]: Predicted labels aligned with messages/true_labels.
    """
    print("Running predictions on test set...")
    predicted_labels = []
    
    for i, message in enumerate(messages):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(messages)}")
        
        try:
            pred = predict_label(model, tokenizer, message)
            predicted_labels.append(pred)
        except Exception as e:
            predicted_labels.append("benign")
    
    print(f"Completed {len(predicted_labels)} predictions\n")
    
    # Compute classification metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels, labels=['pirated', 'benign'])
    
    # Calculate precision, recall, F1-score (per class and averages)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, 
        predicted_labels, 
        labels=['pirated', 'benign'],
        average=None,  # Get per-class metrics
        zero_division=0
    )
    
    # Calculate macro-averaged metrics (unweighted mean of per-class metrics)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=['pirated', 'benign'],
        average='macro',
        zero_division=0
    )
    
    # Calculate weighted-averaged metrics (weighted by support)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=['pirated', 'benign'],
        average='weighted',
        zero_division=0
    )
    
    # Calculate micro-averaged metrics (treats each sample equally, single overall metric)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=['pirated', 'benign'],
        average='micro',
        zero_division=0
    )
    
    # Calculate misclassifications
    total_samples = len(true_labels)
    correct_predictions = int(accuracy * total_samples)
    misclassifications = total_samples - correct_predictions
    misclassification_rate = 1.0 - accuracy
    
    # Per-class metrics
    metrics = {
        'accuracy': float(accuracy),
        'total_samples': int(total_samples),
        'correct_predictions': int(correct_predictions),
        'misclassifications': int(misclassifications),
        'misclassification_rate': float(misclassification_rate),
        'confusion_matrix': cm.tolist(),
        # Whole-model metrics (micro-averaged - treats each sample equally)
        'overall': {
            'precision': float(precision_micro),
            'recall': float(recall_micro),
            'f1_score': float(f1_micro)
        },
        # Per-class metrics
        'per_class': {
            'pirated': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1_score': float(f1[0]),
                'support': int(support[0])
            },
            'benign': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1_score': float(f1[1]),
                'support': int(support[1])
            }
        },
        # Macro-averaged metrics (unweighted mean)
        'macro_avg': {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1_score': float(f1_macro)
        },
        # Weighted-averaged metrics (weighted by support)
        'weighted_avg': {
            'precision': float(precision_weighted),
            'recall': float(recall_weighted),
            'f1_score': float(f1_weighted)
        }
    }
    
    return metrics, predicted_labels


def print_metrics(metrics: dict, model_name: str = "Model"):
    """
    Print evaluation metrics.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics.
        model_name (str): Name of the model being evaluated.
    """
    print("="*80)
    print(f"Test Set Evaluation: {model_name}")
    print("="*80)
    
    # Overall metrics (whole model)
    print(f"\nOverall Model Metrics (Whole Model):")
    overall = metrics['overall']
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {overall['precision']:.4f} ({overall['precision']*100:.2f}%)")
    print(f"  Recall: {overall['recall']:.4f} ({overall['recall']*100:.2f}%)")
    print(f"  F1-Score: {overall['f1_score']:.4f} ({overall['f1_score']*100:.2f}%)")
    print(f"  Total Samples: {metrics['total_samples']:,}")
    print(f"  Correct Predictions: {metrics['correct_predictions']:,}")
    print(f"  Misclassifications: {metrics['misclassifications']:,} ({metrics['misclassification_rate']*100:.2f}%)")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-" * 60)
    for label in ['pirated', 'benign']:
        pc = metrics['per_class'][label]
        print(f"{label.capitalize():<12} {pc['precision']:<12.4f} {pc['recall']:<12.4f} {pc['f1_score']:<12.4f} {pc['support']:<12}")
    
    # Macro-averaged metrics
    print(f"\nMacro-Averaged (Unweighted Mean):")
    macro = metrics['macro_avg']
    print(f"  Precision: {macro['precision']:.4f}")
    print(f"  Recall: {macro['recall']:.4f}")
    print(f"  F1-Score: {macro['f1_score']:.4f}")
    
    # Weighted-averaged metrics
    print(f"\nWeighted-Averaged (Weighted by Support):")
    weighted = metrics['weighted_avg']
    print(f"  Precision: {weighted['precision']:.4f}")
    print(f"  Recall: {weighted['recall']:.4f}")
    print(f"  F1-Score: {weighted['f1_score']:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"                Predicted")
    print(f"                Pirated  Benign")
    print(f"Actual Pirated  {cm[0,0]:<8} {cm[0,1]:<8}")
    print(f"       Benign   {cm[1,0]:<8} {cm[1,1]:<8}")
    
    # Interpretation
    print("\nInterpretation:")
    print(f"  - True Positives (Pirated): {cm[0,0]}")
    print(f"  - False Positives (Pirated): {cm[1,0]} (Benign misclassified as Pirated)")
    print(f"  - False Negatives (Pirated): {cm[0,1]} (Pirated misclassified as Benign)")
    print(f"  - True Negatives (Benign): {cm[1,1]}")
    
    print("="*80 + "\n")


def main():
    """
    Evaluate the channel model (pirated vs benign) on the held-out test set
    from finetune_channels_unsloth.py.
    """
    base_dir = "./path_to_model"
    model_dir = os.path.join(base_dir, "best_model")
    model_name = "Llama 3.2 Channel (merged translated posts: pirated vs benign)"
    test_dataset_path = os.path.join(base_dir, "test_dataset.json")
    output_file = os.path.join(base_dir, "evaluation_results_unsloth.json")
    
    print("="*80)
    print(f"Evaluating: {model_name}")
    print("="*80)
    
    # Load model
    try:
        model, tokenizer = load_finetuned_model(finetuned_path=model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load test data
    try:
        messages, true_labels = load_test_data(
            test_dataset_path=test_dataset_path
        )
    except Exception as e:
        print(f"Error loading test data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not messages:
        print("No test data loaded. Exiting.")
        return
    
    # Evaluate on test set
    try:
        metrics, predicted_labels = evaluate_model(model, tokenizer, messages, true_labels)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print results
    print_metrics(metrics, model_name)
    
    # Save results
    results = {
        'model_name': model_name,
        'model_directory': model_dir,
        'test_dataset_path': test_dataset_path,
        'num_samples': len(messages),
        'metrics': metrics,
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save misclassified examples for inspection
    misclassified_path = os.path.join(base_dir, "misclassified_examples.jsonl")
    mis_count = 0
    with open(misclassified_path, "w", encoding="utf-8") as f:
        for idx, (msg, true, pred) in enumerate(zip(messages, true_labels, predicted_labels)):
            if true != pred:
                rec = {
                    "index": idx,
                    "text": msg,
                    "true_label": true,
                    "predicted_label": pred,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mis_count += 1
    
    print(f" Results saved to: {output_file}")
    print(f" Misclassified examples: {mis_count} (written to {misclassified_path})")


if __name__ == "__main__":
    main()
