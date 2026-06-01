"""BERT/SST-2 quantitative analysis for sink-aware and gradient-weighted RFEM.

This script is self-contained. It does not import from the existing
sst2_quantitative_analysis folder or from the qualitative RFEM scripts.

Methods:
    vanilla   last-layer mean attention, CLS row
    rollout   standard attention rollout, CLS row
    rfem_sa   sink-aware RFEM, unnormalized max-rollout head weights
    rfem_gw   sink-aware RFEM, normalized gradient-based head weights

Default data:
    ../sst2_quantitative_analysis/data/SST-2/sst2_train_10k_seed42.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from transformers import BertForSequenceClassification, BertTokenizer


MODEL_NAME = "textattack/bert-base-uncased-SST-2"
K_VALUES = [0.0, 0.3, 0.5, 1.0]
AOPC_K_FRACTIONS = [0.10, 0.20, 0.30, 0.50]
SPECIAL_TOKENS = ("[CLS]", "[SEP]", "[PAD]")
PUNCT_SET = set(".,!?;:'\"()-")
EPS = 1e-12

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = (
    SCRIPT_DIR.parent
    / "sst2_quantitative_analysis"
    / "data"
    / "SST-2"
    / "sst2_train_10k_seed42.tsv"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"

rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 180,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze BERT RFEM variants on the 10k SST-2 subset."
    )
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap for all methods.")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--k-values", type=float, nargs="+", default=K_VALUES)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-grad-weighted", action="store_true", help="Skip rfem_gw entirely.")
    parser.add_argument(
        "--grad-weighted-max-examples",
        type=int,
        default=0,
        help="Max examples for rfem_gw. 0 = all examples. Default 0.",
    )
    parser.add_argument(
        "--aopc",
        "--faithfulness",
        dest="aopc",
        action="store_true",
        help="Compute AOPC comprehensiveness/sufficiency plus IAUC/DAUC faithfulness metrics.",
    )
    parser.add_argument(
        "--aopc-max-examples",
        type=int,
        default=0,
        help="Max examples for faithfulness metrics. 0 = all examples. Default 0.",
    )
    parser.add_argument("--aopc-k-fractions", type=float, nargs="+", default=AOPC_K_FRACTIONS)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str) -> BertForSequenceClassification:
    try:
        return BertForSequenceClassification.from_pretrained(
            model_name,
            attn_implementation="eager",
            output_attentions=True,
        )
    except TypeError:
        model = BertForSequenceClassification.from_pretrained(model_name, output_attentions=True)
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"
        return model


def make_run_dir(output_root: Path, run_name: str | None) -> Path:
    if run_name is None:
        run_name = f"bert_variants_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def read_dataset(path: Path, max_examples: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            text = (row.get("sentence") or "").strip()
            label_raw = (row.get("label") or "").strip()
            if not text or label_raw == "":
                continue
            rows.append(
                {
                    "example_id": len(rows),
                    "source_row_id": row.get("source_row_id", str(idx)),
                    "sentence": text,
                    "label": int(label_raw),
                }
            )
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:4.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes:02d}:{sec:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}"


def progress(current: int, total: int, start: float, prefix: str) -> None:
    if total <= 0:
        return
    width = 34
    frac = current / total
    fill = int(width * frac)
    bar = "#" * fill + "-" * (width - fill)
    elapsed = time.time() - start
    rate = current / elapsed if elapsed > EPS else 0.0
    eta = (total - current) / rate if rate > EPS else 0.0
    sys.stdout.write(
        f"\r{prefix}: [{bar}] {current:>5}/{total:<5} "
        f"{frac * 100:6.2f}% elapsed {format_seconds(elapsed)} eta {format_seconds(eta)}"
    )
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def token_category(token: str) -> str:
    if token in SPECIAL_TOKENS:
        return "special"
    if token in PUNCT_SET:
        return "punct"
    return "content"


def category_indices(tokens: list[str]) -> dict[str, list[int]]:
    out = {"special": [], "punct": [], "content": []}
    for idx, token in enumerate(tokens):
        out[token_category(token)].append(idx)
    return out


def normalized_entropy(values: np.ndarray) -> tuple[float, float]:
    values = np.maximum(values.astype(float), 0.0)
    total = float(values.sum())
    if total <= EPS or len(values) <= 1:
        return 0.0, 0.0
    p = values / total
    p = p[p > EPS]
    entropy = float(-(p * np.log(p)).sum())
    return entropy / math.log(len(values)), float(math.exp(entropy))


def row_metrics(row: torch.Tensor, tokens: list[str]) -> dict[str, Any]:
    values = np.maximum(row.detach().float().cpu().numpy(), 0.0)
    total = float(values.sum())
    cats = category_indices(tokens)
    special_mass = float(values[cats["special"]].sum()) if cats["special"] else 0.0
    content_mass = float(values[cats["content"]].sum()) if cats["content"] else 0.0
    punct_mass = float(values[cats["punct"]].sum()) if cats["punct"] else 0.0
    cls_self = float(values[0]) if len(values) else 0.0
    top_idx = int(values.argmax()) if len(values) else -1
    top_token = tokens[top_idx] if top_idx >= 0 else ""
    top_cat = token_category(top_token) if top_token else "none"
    content_values = values[cats["content"]] if cats["content"] else np.asarray([])
    entropy, effective_tokens = normalized_entropy(values)
    return {
        "token_count": len(tokens),
        "content_token_count": len(cats["content"]),
        "row_total_mass": total,
        "cls_self_share": cls_self / total if total > EPS else 0.0,
        "special_share": special_mass / total if total > EPS else 0.0,
        "content_share": content_mass / total if total > EPS else 0.0,
        "punct_share": punct_mass / total if total > EPS else 0.0,
        "content_survival_rate": float((content_values > EPS).sum() / len(content_values))
        if content_values.size
        else 0.0,
        "row_nonzero_rate": float((values > EPS).sum() / len(values)) if len(values) else 0.0,
        "top_idx": top_idx,
        "top_token": top_token,
        "top_category": top_cat,
        "top_is_special": int(top_cat == "special"),
        "top_is_content": int(top_cat == "content"),
        "normalized_entropy": entropy,
        "effective_tokens": effective_tokens,
    }


def tokenize(tokenizer: BertTokenizer, text: str, device: torch.device, max_length: int) -> dict[str, torch.Tensor]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in inputs.items()}


def forward_attentions(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    text: str,
    device: torch.device,
    max_length: int,
) -> tuple[list[str], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    inputs = tokenize(tokenizer, text, device, max_length)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    if not outputs.attentions:
        raise RuntimeError("Model did not return attentions. Use eager attention implementation.")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].detach().cpu())
    attn_all = torch.stack(outputs.attentions, dim=0).squeeze(1)
    probs = torch.softmax(outputs.logits.squeeze(0), dim=-1)
    return tokens, inputs, attn_all, probs


def gradient_head_weights(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    text: str,
    pred_class: int,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    inputs = tokenize(tokenizer, text, device, max_length)
    outputs = model(**inputs, output_attentions=True)
    if not outputs.attentions:
        raise RuntimeError("Model did not return attentions during gradient pass.")
    for attn in outputs.attentions:
        attn.retain_grad()
    score = outputs.logits[0, pred_class]
    model.zero_grad(set_to_none=True)
    score.backward()

    heads = outputs.attentions[0].shape[1]
    grad_weights = torch.zeros(heads, device=device, dtype=outputs.attentions[0].dtype)
    for layer_attn in outputs.attentions:
        if layer_attn.grad is None:
            raise RuntimeError("Attention gradient was not retained.")
        for h in range(heads):
            a = layer_attn.detach().squeeze(0)[h]
            g = layer_attn.grad.detach().squeeze(0)[h]
            grad_weights[h] += (torch.relu(g) * a).mean()
    model.zero_grad(set_to_none=True)
    return grad_weights.detach()


def attention_rollout(attn_all: torch.Tensor) -> torch.Tensor:
    layers, _, seq_len, _ = attn_all.shape
    eye = torch.eye(seq_len, device=attn_all.device, dtype=attn_all.dtype)
    rollout = eye.clone()
    for layer_idx in range(layers):
        a = attn_all[layer_idx].mean(dim=0)
        a_plus_i = a + eye
        a_norm = a_plus_i / a_plus_i.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        rollout = a_norm @ rollout
    return rollout


def per_head_rollout(attn_all: torch.Tensor) -> torch.Tensor:
    layers, heads, seq_len, _ = attn_all.shape
    eye = torch.eye(seq_len, device=attn_all.device, dtype=attn_all.dtype)
    rolled = []
    for h in range(heads):
        mat = eye.clone()
        for layer_idx in range(layers):
            a = attn_all[layer_idx, h]
            a_plus_i = a + eye
            a_norm = a_plus_i / a_plus_i.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            mat = a_norm @ mat
        rolled.append(mat)
    return torch.stack(rolled, dim=0)


def sink_indices(tokens: list[str]) -> list[int]:
    return [idx for idx, token in enumerate(tokens) if token in SPECIAL_TOKENS]


def sink_aware_filter(
    head_rollouts: torch.Tensor,
    tokens: list[str],
    k_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """mu/sigma from CLS row excluding [CLS]/[SEP]/[PAD]; apply threshold to full matrix."""
    _, seq_len, _ = head_rollouts.shape
    sinks = set(sink_indices(tokens))
    cols = [j for j in range(seq_len) if j not in sinks]
    if not cols:
        cols = list(range(seq_len))

    masks, means, stds, thresholds = [], [], [], []
    for h in range(head_rollouts.shape[0]):
        r_h = head_rollouts[h]
        vals = r_h[0, cols]
        mu = vals.mean()
        sigma = vals.std(unbiased=False)
        thr = mu + k_value * sigma
        mask = torch.where(r_h >= thr, r_h, torch.zeros_like(r_h))
        masks.append(mask)
        means.append(mu)
        stds.append(sigma)
        thresholds.append(thr)
    return torch.stack(masks), torch.stack(means), torch.stack(stds), torch.stack(thresholds)


def aggregate_sa(head_masks: torch.Tensor, head_rollouts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.stack([head_rollouts[h].max() for h in range(head_masks.shape[0])])
    agg = (weights.view(-1, 1, 1) * head_masks).sum(dim=0)
    return agg, weights


def aggregate_gw(head_masks: torch.Tensor, grad_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.relu(grad_weights)
    total = weights.sum()
    if total <= EPS:
        weights = torch.ones_like(weights) / len(weights)
    else:
        weights = weights / total
    agg = (weights.view(-1, 1, 1) * head_masks).sum(dim=0)
    return agg, weights


def common_fields(item: dict[str, Any], tokens: list[str], pred: int, probs: torch.Tensor) -> dict[str, Any]:
    label = int(item["label"])
    return {
        "example_id": item["example_id"],
        "source_row_id": item["source_row_id"],
        "sentence": item["sentence"],
        "label": label,
        "prediction": pred,
        "prediction_prob": float(probs[pred].item()),
        "gold_prob": float(probs[label].item()),
        "correct": int(pred == label),
        "tokens": " ".join(tokens),
    }


def method_row(
    item: dict[str, Any],
    tokens: list[str],
    pred: int,
    probs: torch.Tensor,
    method: str,
    k_value: float | str,
    row: torch.Tensor,
) -> dict[str, Any]:
    return {
        **common_fields(item, tokens, pred, probs),
        "method": method,
        "k": k_value,
        **row_metrics(row, tokens),
    }


def per_head_rows(
    item: dict[str, Any],
    tokens: list[str],
    method: str,
    k_value: float,
    masks: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
    thresholds: torch.Tensor,
    weights: torch.Tensor,
) -> list[dict[str, Any]]:
    cats = category_indices(tokens)
    total_entries = masks.shape[1] * masks.shape[2]
    rows = []
    for h in range(masks.shape[0]):
        mask_np = masks[h].detach().float().cpu().numpy()
        cls_row = mask_np[0]
        content_vals = cls_row[cats["content"]] if cats["content"] else np.asarray([])
        special_vals = cls_row[cats["special"]] if cats["special"] else np.asarray([])
        rows.append(
            {
                "method": method,
                "example_id": item["example_id"],
                "source_row_id": item["source_row_id"],
                "label": item["label"],
                "k": k_value,
                "head": h + 1,
                "mu": float(means[h].detach().cpu().item()),
                "sigma": float(stds[h].detach().cpu().item()),
                "threshold": float(thresholds[h].detach().cpu().item()),
                "weight": float(weights[h].detach().cpu().item()),
                "full_kept_pct": float((mask_np > EPS).sum() / total_entries * 100.0),
                "cls_row_kept_pct": float((cls_row > EPS).sum() / len(cls_row) * 100.0),
                "cls_content_kept_rate": float((content_vals > EPS).sum() / len(content_vals))
                if content_vals.size
                else 0.0,
                "cls_special_kept_rate": float((special_vals > EPS).sum() / len(special_vals))
                if special_vals.size
                else 0.0,
            }
        )
    return rows


def content_rank(row: torch.Tensor, tokens: list[str]) -> list[int]:
    values = row.detach().float().cpu().numpy()
    indices = [idx for idx, token in enumerate(tokens) if token_category(token) == "content"]
    return sorted(indices, key=lambda idx: values[idx], reverse=True)


def normalized_auc(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    span = float(x[-1] - x[0])
    if span <= EPS:
        return 0.0
    widths = x[1:] - x[:-1]
    heights = (y[1:] + y[:-1]) * 0.5
    return float((widths * heights).sum() / span)


def perturb_probs(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    inputs: dict[str, torch.Tensor],
    content_indices: list[int],
    selected: set[int],
    mode: str,
) -> torch.Tensor:
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("Tokenizer has no mask token.")
    perturbed = {k: v.clone() for k, v in inputs.items()}
    if mode == "comprehensiveness":
        to_mask = selected
    elif mode == "sufficiency":
        to_mask = set(content_indices) - selected
    else:
        raise ValueError(mode)
    for pos in to_mask:
        perturbed["input_ids"][0, pos] = mask_id
    with torch.no_grad():
        outputs = model(**perturbed, output_attentions=False)
    return torch.softmax(outputs.logits.squeeze(0), dim=-1)


def faithfulness_rows_for_example(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    inputs: dict[str, torch.Tensor],
    item: dict[str, Any],
    tokens: list[str],
    pred: int,
    probs: torch.Tensor,
    rows_for_rank: list[tuple[str, float | str, torch.Tensor]],
    fractions: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    original = float(probs[pred].item())
    content_indices = [idx for idx, token in enumerate(tokens) if token_category(token) == "content"]
    if not content_indices:
        return [], []

    clean_fractions = sorted({min(1.0, max(0.0, float(frac))) for frac in fractions if float(frac) > 0.0})
    if not clean_fractions:
        return [], []

    all_masked_probs = perturb_probs(model, tokenizer, inputs, content_indices, set(), "sufficiency")
    all_masked_prob = float(all_masked_probs[pred].item())

    aopc_out = []
    auc_out = []
    for method, k_value, row in rows_for_rank:
        ranked = content_rank(row, tokens)
        deletion_x = [0.0]
        deletion_y = [original]
        insertion_x = [0.0]
        insertion_y = [all_masked_prob]
        for frac in clean_fractions:
            n_select = max(1, int(math.ceil(len(ranked) * frac)))
            selected_list = ranked[:n_select]
            selected = set(selected_list)
            comp_probs = perturb_probs(model, tokenizer, inputs, content_indices, selected, "comprehensiveness")
            suff_probs = perturb_probs(model, tokenizer, inputs, content_indices, selected, "sufficiency")
            comp_prob_after = float(comp_probs[pred].item())
            suff_prob_after = float(suff_probs[pred].item())
            comp = original - comp_prob_after
            suff = original - suff_prob_after
            deletion_x.append(frac)
            deletion_y.append(comp_prob_after)
            insertion_x.append(frac)
            insertion_y.append(suff_prob_after)
            aopc_out.append(
                {
                    **common_fields(item, tokens, pred, probs),
                    "method": method,
                    "k": k_value,
                    "fraction": frac,
                    "n_selected": n_select,
                    "selected_positions": " ".join(str(i) for i in selected_list),
                    "selected_tokens": " ".join(tokens[i] for i in selected_list),
                    "original_pred_prob": original,
                    "comprehensiveness": comp,
                    "sufficiency": suff,
                    "comprehensiveness_prob_after": comp_prob_after,
                    "sufficiency_prob_after": suff_prob_after,
                }
            )

        if deletion_x[-1] < 1.0:
            deletion_x.append(1.0)
            deletion_y.append(all_masked_prob)
        else:
            deletion_y[-1] = all_masked_prob
        if insertion_x[-1] < 1.0:
            insertion_x.append(1.0)
            insertion_y.append(original)
        else:
            insertion_y[-1] = original
        auc_out.append(
            {
                **common_fields(item, tokens, pred, probs),
                "method": method,
                "k": k_value,
                "content_token_count": len(content_indices),
                "original_pred_prob": original,
                "all_content_masked_pred_prob": all_masked_prob,
                "dauc": normalized_auc(deletion_x, deletion_y),
                "iauc": normalized_auc(insertion_x, insertion_y),
                "deletion_fractions": " ".join(f"{x:.4g}" for x in deletion_x),
                "deletion_pred_probs": " ".join(f"{y:.6g}" for y in deletion_y),
                "insertion_fractions": " ".join(f"{x:.4g}" for x in insertion_x),
                "insertion_pred_probs": " ".join(f"{y:.6g}" for y in insertion_y),
            }
        )
    return aopc_out, auc_out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], group_keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k, "") for k in group_keys)].append(row)
    out = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        row = {group_keys[i]: key[i] for i in range(len(group_keys))}
        row["n"] = len(group)
        for metric in metrics:
            vals = [float(r[metric]) for r in group if r.get(metric, "") != ""]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            row[f"{metric}_mean"] = float(arr.mean())
            row[f"{metric}_std"] = float(arr.std(ddof=0))
            row[f"{metric}_median"] = float(np.quantile(arr, 0.50))
            row[f"{metric}_p90"] = float(np.quantile(arr, 0.90))
            row[f"{metric}_p95"] = float(np.quantile(arr, 0.95))
            row[f"{metric}_min"] = float(arr.min())
            row[f"{metric}_max"] = float(arr.max())
        out.append(row)
    return out


def display_method(method: str, k_value: Any) -> str:
    if method == "vanilla":
        return "Vanilla"
    if method == "rollout":
        return "Rollout"
    if method == "rfem_sa":
        return f"SA K={float(k_value):g}"
    if method == "rfem_gw":
        return f"GW K={float(k_value):g}"
    return str(method)


def ordered_groups(rows: list[dict[str, Any]]) -> list[tuple[str, Any, str]]:
    keys = {(row["method"], row.get("k", "")) for row in rows}

    def sort_key(item: tuple[str, Any]) -> tuple[int, float]:
        method, k_value = item
        order = {"vanilla": 0, "rollout": 1, "rfem_sa": 2, "rfem_gw": 3}.get(method, 9)
        try:
            k_float = float(k_value)
        except (TypeError, ValueError):
            k_float = -1.0
        return order, k_float

    return [(m, k, display_method(m, k)) for m, k in sorted(keys, key=sort_key)]


def vals_for(rows: list[dict[str, Any]], method: str, k_value: Any, metric: str) -> list[float]:
    return [
        float(row[metric])
        for row in rows
        if row["method"] == method and str(row.get("k", "")) == str(k_value) and row.get(metric, "") != ""
    ]


def save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)


def annotated_matrix(
    fig_dir: Path,
    filename: str,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    fmt: str = "{:.3f}",
    cmap: str = "YlGnBu",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 1.65), max(4.8, len(row_labels) * 0.6 + 2)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix[i, j])
            ax.text(
                j,
                i,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white" if value > vmin + (vmax - vmin) * 0.58 else "black",
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean value")
    fig.tight_layout()
    save_pdf(fig, fig_dir / filename)


def make_figures(run_dir: Path, example_rows: list[dict[str, Any]], per_head: list[dict[str, Any]]) -> None:
    fig_dir = run_dir / "figures"
    groups = ordered_groups(example_rows)
    labels = [label for _, _, label in groups]
    metrics = [
        ("cls_self_share", "CLS self"),
        ("special_share", "Special"),
        ("content_share", "Content"),
        ("content_survival_rate", "Content survive"),
        ("row_nonzero_rate", "Row nonzero"),
        ("top_is_special", "Top special"),
        ("normalized_entropy", "Entropy"),
    ]
    matrix = np.asarray(
        [
            [np.mean(vals_for(example_rows, method, k, metric)) for metric, _ in metrics]
            for method, k, _ in groups
        ],
        dtype=float,
    )
    annotated_matrix(
        fig_dir,
        "method_metric_value_matrix.pdf",
        matrix,
        labels,
        [label for _, label in metrics],
        "Mean Metrics by Method",
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, metric, title in [
        (axes[0], "cls_self_share", "CLS Self Share"),
        (axes[1], "special_share", "Special-Token Share"),
    ]:
        data = [vals_for(example_rows, method, k, metric) for method, k, _ in groups]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Share of CLS-row mass")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", rotation=35)
        for idx, values in enumerate(data, start=1):
            if values:
                mean = float(np.mean(values))
                ax.scatter(idx, mean, marker="D", color="black", s=30, zorder=3)
                ax.text(idx, min(0.98, mean + 0.04), f"{mean:.3f}", ha="center", fontsize=8)
    fig.suptitle("Sink Share by Method", fontsize=18, fontweight="bold")
    fig.tight_layout()
    save_pdf(fig, fig_dir / "sink_share_boxplot.pdf")

    means = [np.mean(vals_for(example_rows, method, k, "content_survival_rate")) for method, k, _ in groups]
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(labels, means, color="#2ca02c")
    ax.set_title("Content Survival by Method", fontweight="bold")
    ax.set_ylabel("Mean content-token survival rate")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    save_pdf(fig, fig_dir / "content_survival_bar.pdf")

    cats = ["special", "punct", "content"]
    colors = {"special": "#d62728", "punct": "#ff7f0e", "content": "#1f77b4"}
    bottoms = np.zeros(len(groups))
    fig, ax = plt.subplots(figsize=(14, 7))
    for cat in cats:
        fractions = []
        for method, k, _ in groups:
            subset = [r for r in example_rows if r["method"] == method and str(r.get("k", "")) == str(k)]
            count = sum(1 for row in subset if row["top_category"] == cat)
            fractions.append(count / len(subset) if subset else 0.0)
        ax.bar(labels, fractions, bottom=bottoms, color=colors[cat], label=cat)
        for idx, value in enumerate(fractions):
            if value >= 0.05:
                ax.text(idx, bottoms[idx] + value / 2, f"{value * 100:.0f}%", ha="center", va="center", fontsize=9)
        bottoms += np.asarray(fractions)
    ax.set_title("Top Token Category by Method", fontweight="bold")
    ax.set_ylabel("Fraction of examples")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.tight_layout()
    save_pdf(fig, fig_dir / "top_token_category_stacked_bar.pdf")

    fig, ax = plt.subplots(figsize=(14, 7))
    data = [vals_for(example_rows, method, k, "normalized_entropy") for method, k, _ in groups]
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("CLS-Row Entropy by Method", fontweight="bold")
    ax.set_ylabel("Normalized entropy")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_pdf(fig, fig_dir / "entropy_boxplot.pdf")

    # Variant sparsity by K.
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, color in [("rfem_sa", "#1f77b4"), ("rfem_gw", "#9467bd")]:
        method_rows = [r for r in example_rows if r["method"] == method]
        if not method_rows:
            continue
        ks = sorted({float(r["k"]) for r in method_rows})
        y = [np.mean([float(r["content_survival_rate"]) for r in method_rows if float(r["k"]) == k]) for k in ks]
        ax.plot(ks, y, marker="o", linewidth=2.5, label=method, color=color)
        for x_val, y_val in zip(ks, y):
            ax.text(x_val, y_val + 0.02, f"{y_val:.2f}", ha="center", fontsize=9, color=color)
    ax.set_title("Per-Variant Content Survival by K", fontweight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel("Mean content survival rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    save_pdf(fig, fig_dir / "per_variant_sparsity_by_k.pdf")

    for method in ["rfem_sa", "rfem_gw"]:
        rows = [r for r in per_head if r["method"] == method]
        if not rows:
            continue
        ks = sorted({float(r["k"]) for r in rows})
        heads = sorted({int(r["head"]) for r in rows})
        mat = np.zeros((len(ks), len(heads)))
        for i, k in enumerate(ks):
            for j, head in enumerate(heads):
                vals = [float(r["full_kept_pct"]) for r in rows if float(r["k"]) == k and int(r["head"]) == head]
                mat[i, j] = float(np.mean(vals)) if vals else 0.0
        fig, ax = plt.subplots(figsize=(15, 6))
        im = ax.imshow(mat, aspect="auto", cmap="magma")
        ax.set_title(f"Head Sparsity Heatmap: {method}", fontweight="bold")
        ax.set_xlabel("Head")
        ax.set_ylabel("K")
        ax.set_xticks(np.arange(len(heads)))
        ax.set_xticklabels([f"H{h}" for h in heads])
        ax.set_yticks(np.arange(len(ks)))
        ax.set_yticklabels([f"{k:g}" for k in ks])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=9, color="white")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("% full matrix entries kept")
        fig.tight_layout()
        save_pdf(fig, fig_dir / f"head_sparsity_heatmap_{method}.pdf")


def mean_for_group(rows: list[dict[str, Any]], method: str, k_value: Any, metric: str) -> float:
    values = vals_for(rows, method, k_value, metric)
    return float(np.mean(values)) if values else 0.0


def make_faithfulness_figures(
    run_dir: Path,
    aopc_rows: list[dict[str, Any]],
    auc_rows: list[dict[str, Any]],
) -> None:
    if not aopc_rows and not auc_rows:
        return
    fig_dir = run_dir / "figures"
    groups = ordered_groups(auc_rows if auc_rows else aopc_rows)
    labels = [label for _, _, label in groups]

    matrix_metrics = [
        ("comprehensiveness", "AOPC comp", aopc_rows),
        ("sufficiency", "AOPC suff", aopc_rows),
        ("iauc", "IAUC", auc_rows),
        ("dauc", "DAUC", auc_rows),
    ]
    matrix = np.asarray(
        [
            [mean_for_group(rows, method, k, metric) for metric, _, rows in matrix_metrics]
            for method, k, _ in groups
        ],
        dtype=float,
    )
    vmin = min(0.0, float(matrix.min())) if matrix.size else 0.0
    vmax = max(1.0, float(matrix.max())) if matrix.size else 1.0
    annotated_matrix(
        fig_dir,
        "faithfulness_metric_matrix.pdf",
        matrix,
        labels,
        [label for _, label, _ in matrix_metrics],
        "Mean Faithfulness Metrics by Method",
        cmap="PuBuGn",
        vmin=vmin,
        vmax=vmax,
    )

    if aopc_rows:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for ax, metric, title in [
            (axes[0], "comprehensiveness", "AOPC Comprehensiveness"),
            (axes[1], "sufficiency", "AOPC Sufficiency"),
        ]:
            data = [vals_for(aopc_rows, method, k, metric) for method, k, _ in groups]
            ax.boxplot(data, labels=labels, showfliers=False)
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel("Original predicted-class prob minus perturbed prob")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.tick_params(axis="x", rotation=35)
            for idx, values in enumerate(data, start=1):
                if values:
                    mean = float(np.mean(values))
                    ax.scatter(idx, mean, marker="D", color="black", s=30, zorder=3)
                    ax.text(idx, mean, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
        fig.suptitle("AOPC Faithfulness by Method", fontsize=18, fontweight="bold")
        fig.tight_layout()
        save_pdf(fig, fig_dir / "aopc_comprehensiveness_sufficiency_boxplot.pdf")

    if auc_rows:
        iauc_means = [mean_for_group(auc_rows, method, k, "iauc") for method, k, _ in groups]
        dauc_means = [mean_for_group(auc_rows, method, k, "dauc") for method, k, _ in groups]
        x = np.arange(len(groups))
        width = 0.38
        fig, ax = plt.subplots(figsize=(16, 7))
        bars_i = ax.bar(x - width / 2, iauc_means, width, label="IAUC", color="#1f77b4")
        bars_d = ax.bar(x + width / 2, dauc_means, width, label="DAUC", color="#d62728")
        ax.set_title("Insertion/Deletion AUC by Method", fontweight="bold")
        ax.set_ylabel("Normalized AUC of predicted-class probability")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend()
        for bars in [bars_i, bars_d]:
            for bar in bars:
                value = float(bar.get_height())
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
        fig.tight_layout()
        save_pdf(fig, fig_dir / "iauc_dauc_bar.pdf")


def top_category_rows(example_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method, k, label in ordered_groups(example_rows):
        subset = [r for r in example_rows if r["method"] == method and str(r.get("k", "")) == str(k)]
        counts = Counter(r["top_category"] for r in subset)
        for cat in ["special", "punct", "content"]:
            count = counts.get(cat, 0)
            out.append(
                {
                    "method": method,
                    "k": k,
                    "method_label": label,
                    "top_category": cat,
                    "count": count,
                    "pct": count / len(subset) * 100.0 if subset else 0.0,
                }
            )
    return out


def faithfulness_summary(
    aopc_summary_rows: list[dict[str, Any]],
    auc_summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = {(row["method"], row.get("k", "")) for row in aopc_summary_rows}
    keys.update((row["method"], row.get("k", "")) for row in auc_summary_rows)
    lookup_aopc = {(row["method"], str(row.get("k", ""))): row for row in aopc_summary_rows}
    lookup_auc = {(row["method"], str(row.get("k", ""))): row for row in auc_summary_rows}
    dummy = [{"method": method, "k": k} for method, k in keys]
    out = []
    for method, k_value, label in ordered_groups(dummy):
        row = {"method": method, "k": k_value, "method_label": label}
        aopc = lookup_aopc.get((method, str(k_value)), {})
        auc = lookup_auc.get((method, str(k_value)), {})
        row["n_aopc"] = aopc.get("n", "")
        row["aopc_comprehensiveness_mean"] = aopc.get("comprehensiveness_mean", "")
        row["aopc_sufficiency_mean"] = aopc.get("sufficiency_mean", "")
        row["n_auc"] = auc.get("n", "")
        row["iauc_mean"] = auc.get("iauc_mean", "")
        row["dauc_mean"] = auc.get("dauc_mean", "")
        out.append(row)
    return out


def rfem_k_summary(example_rows: list[dict[str, Any]], per_head: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in ["rfem_sa", "rfem_gw"]:
        rows = [r for r in example_rows if r["method"] == method]
        if not rows:
            continue
        for k in sorted({float(r["k"]) for r in rows}):
            subset = [r for r in rows if float(r["k"]) == k]
            heads = [r for r in per_head if r["method"] == method and float(r["k"]) == k]
            out.append(
                {
                    "method": method,
                    "k": k,
                    "n_examples": len(subset),
                    "mean_cls_self_share": float(np.mean([float(r["cls_self_share"]) for r in subset])),
                    "mean_special_share": float(np.mean([float(r["special_share"]) for r in subset])),
                    "mean_content_share": float(np.mean([float(r["content_share"]) for r in subset])),
                    "mean_content_survival_rate": float(np.mean([float(r["content_survival_rate"]) for r in subset])),
                    "top_special_rate": float(np.mean([float(r["top_is_special"]) for r in subset])),
                    "mean_row_nonzero_rate": float(np.mean([float(r["row_nonzero_rate"]) for r in subset])),
                    "mean_full_matrix_kept_pct": float(np.mean([float(r["full_kept_pct"]) for r in heads]))
                    if heads
                    else 0.0,
                    "mean_head_weight": float(np.mean([float(r["weight"]) for r in heads])) if heads else 0.0,
                }
            )
    return out


def high_sink_rows(example_rows: list[dict[str, Any]], top_n: int = 25) -> list[dict[str, Any]]:
    out = []
    for method, k, label in ordered_groups(example_rows):
        subset = [r for r in example_rows if r["method"] == method and str(r.get("k", "")) == str(k)]
        subset = sorted(subset, key=lambda r: (float(r["special_share"]), float(r["cls_self_share"])), reverse=True)
        for rank, row in enumerate(subset[:top_n], start=1):
            out.append(
                {
                    "rank_within_method": rank,
                    "method_label": label,
                    "method": method,
                    "k": k,
                    "example_id": row["example_id"],
                    "label": row["label"],
                    "prediction": row["prediction"],
                    "correct": row["correct"],
                    "special_share": row["special_share"],
                    "cls_self_share": row["cls_self_share"],
                    "content_share": row["content_share"],
                    "top_token": row["top_token"],
                    "top_category": row["top_category"],
                    "sentence": row["sentence"],
                }
            )
    return out


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def fmt(value: Any, places: int = 4) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.{places}f}"


def write_report(
    run_dir: Path,
    args: argparse.Namespace,
    examples: list[dict[str, Any]],
    method_summary: list[dict[str, Any]],
    rfem_summary: list[dict[str, Any]],
    aopc_summary_rows: list[dict[str, Any]],
    auc_summary_rows: list[dict[str, Any]],
    accuracy: float,
    elapsed: float,
) -> None:
    lines = [
        "# BERT RFEM Variants Quantitative Report",
        "",
        "## Goal",
        "",
        "Compare sink-aware RFEM (`rfem_sa`) and gradient-weighted RFEM (`rfem_gw`) "
        "against vanilla attention and standard rollout on the 10k SST-2 subset.",
        "",
        "## Run Configuration",
        "",
        markdown_table(
            ["Setting", "Value"],
            [
                ["Data file", str(args.data_file)],
                ["Examples loaded", len(examples)],
                ["Model", args.model_name],
                ["K values", ", ".join(f"{k:g}" for k in args.k_values)],
                ["GW enabled", str(not args.no_grad_weighted)],
                ["GW max examples", args.grad_weighted_max_examples],
                ["Faithfulness enabled", str(args.aopc)],
                ["Faithfulness max examples", args.aopc_max_examples],
                ["Accuracy", f"{accuracy * 100:.2f}%"],
                ["Elapsed seconds", f"{elapsed:.1f}"],
            ],
        ),
        "",
        "## Main Method Summary",
        "",
    ]
    rows = []
    summary_lookup = {(r["method"], str(r.get("k", ""))): r for r in method_summary}
    dummy_rows = []
    for r in method_summary:
        dummy_rows.extend([{"method": r["method"], "k": r.get("k", "")}])
    for method, k, label in ordered_groups(dummy_rows):
        r = summary_lookup[(method, str(k))]
        rows.append(
            [
                label,
                r["n"],
                fmt(r.get("cls_self_share_mean")),
                fmt(r.get("special_share_mean")),
                fmt(r.get("content_share_mean")),
                fmt(r.get("content_survival_rate_mean")),
                fmt(r.get("top_is_special_mean")),
                fmt(r.get("normalized_entropy_mean")),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Method",
                "N",
                "CLS Self",
                "Special",
                "Content",
                "Content Survival",
                "Top Special",
                "Entropy",
            ],
            rows,
        )
    )
    lines.extend(["", "## RFEM K Summary", ""])
    lines.append(
        markdown_table(
            ["Method", "K", "N", "Special", "Content", "Content Survival", "Full Kept %", "Mean Weight"],
            [
                [
                    r["method"],
                    f"{float(r['k']):g}",
                    r["n_examples"],
                    fmt(r["mean_special_share"]),
                    fmt(r["mean_content_share"]),
                    fmt(r["mean_content_survival_rate"]),
                    fmt(r["mean_full_matrix_kept_pct"], 2),
                    fmt(r["mean_head_weight"]),
                ]
                for r in rfem_summary
            ],
        )
    )
    if aopc_summary_rows:
        lines.extend(["", "## AOPC Summary", ""])
        lines.append(
            markdown_table(
                ["Method", "K", "N", "Comprehensiveness", "Sufficiency"],
                [
                    [
                        display_method(r["method"], r.get("k", "")),
                        r.get("k", ""),
                        r["n"],
                        fmt(r.get("comprehensiveness_mean")),
                        fmt(r.get("sufficiency_mean")),
                    ]
                    for r in aopc_summary_rows
                ],
            )
        )
    if auc_summary_rows:
        lines.extend(["", "## IAUC / DAUC Summary", ""])
        lines.append(
            markdown_table(
                ["Method", "K", "N", "IAUC", "DAUC"],
                [
                    [
                        display_method(r["method"], r.get("k", "")),
                        r.get("k", ""),
                        r["n"],
                        fmt(r.get("iauc_mean")),
                        fmt(r.get("dauc_mean")),
                    ]
                    for r in auc_summary_rows
                ],
            )
        )
        lines.extend(
            [
                "",
                "IAUC is the area under the insertion curve after adding top-ranked content tokens back. "
                "Higher IAUC indicates the ranking restores model confidence quickly. DAUC is the area "
                "under the deletion curve after masking top-ranked content tokens. Lower DAUC indicates "
                "the ranking removes model confidence quickly.",
            ]
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `figures/method_metric_value_matrix.pdf`",
            "- `figures/sink_share_boxplot.pdf`",
            "- `figures/content_survival_bar.pdf`",
            "- `figures/top_token_category_stacked_bar.pdf`",
            "- `figures/entropy_boxplot.pdf`",
            "- `figures/per_variant_sparsity_by_k.pdf`",
            "- `figures/head_sparsity_heatmap_rfem_sa.pdf`",
            "- `figures/head_sparsity_heatmap_rfem_gw.pdf`",
            "- `figures/faithfulness_metric_matrix.pdf`",
            "- `figures/aopc_comprehensiveness_sufficiency_boxplot.pdf`",
            "- `figures/iauc_dauc_bar.pdf`",
            "",
        ]
    )
    (run_dir / "bert_variants_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.time()
    device = choose_device(args.device)
    run_dir = make_run_dir(args.output_root, args.run_name)
    config = {key: json_safe(value) for key, value in vars(args).items()}
    config["data_file"] = str(args.data_file.resolve())
    config["run_dir"] = str(run_dir.resolve())
    config["device_resolved"] = str(device)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    examples = read_dataset(args.data_file, args.max_examples)
    if not examples:
        raise RuntimeError("No examples loaded.")

    print(f"Loading tokenizer/model: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = load_model(args.model_name)
    model.to(device)
    model.eval()
    print(f"Device: {device}")
    print(f"Examples: {len(examples)}")

    gw_limit = len(examples) if args.grad_weighted_max_examples == 0 else min(args.grad_weighted_max_examples, len(examples))
    aopc_limit = len(examples) if args.aopc_max_examples == 0 else min(args.aopc_max_examples, len(examples))
    if args.no_grad_weighted:
        gw_limit = 0

    example_rows: list[dict[str, Any]] = []
    per_head_metrics: list[dict[str, Any]] = []
    aopc_rows: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []
    correct: list[int] = []
    progress_start = time.time()

    for idx, item in enumerate(examples, start=1):
        tokens, inputs, attn_all, probs = forward_attentions(
            model, tokenizer, item["sentence"], device, args.max_length
        )
        pred = int(probs.argmax().item())
        correct.append(int(pred == int(item["label"])))

        last_layer_mean = attn_all[-1].mean(dim=0)
        rollout = attention_rollout(attn_all)
        head_rollouts = per_head_rollout(attn_all)

        rows_for_aopc: list[tuple[str, float | str, torch.Tensor]] = []
        example_rows.append(method_row(item, tokens, pred, probs, "vanilla", "", last_layer_mean[0]))
        example_rows.append(method_row(item, tokens, pred, probs, "rollout", "", rollout[0]))
        rows_for_aopc.append(("vanilla", "", last_layer_mean[0].detach().cpu()))
        rows_for_aopc.append(("rollout", "", rollout[0].detach().cpu()))

        # Variant 1: sink-aware RFEM.
        for k_value in args.k_values:
            masks, means, stds, thresholds = sink_aware_filter(head_rollouts, tokens, float(k_value))
            agg, weights = aggregate_sa(masks, head_rollouts)
            example_rows.append(method_row(item, tokens, pred, probs, "rfem_sa", float(k_value), agg[0]))
            per_head_metrics.extend(per_head_rows(item, tokens, "rfem_sa", float(k_value), masks, means, stds, thresholds, weights))
            rows_for_aopc.append(("rfem_sa", float(k_value), agg[0].detach().cpu()))

        # Variant 2: gradient-weighted RFEM.
        if idx <= gw_limit:
            grad_weights = gradient_head_weights(model, tokenizer, item["sentence"], pred, device, args.max_length)
            for k_value in args.k_values:
                masks, means, stds, thresholds = sink_aware_filter(head_rollouts, tokens, float(k_value))
                agg, weights = aggregate_gw(masks, grad_weights)
                example_rows.append(method_row(item, tokens, pred, probs, "rfem_gw", float(k_value), agg[0]))
                per_head_metrics.extend(
                    per_head_rows(item, tokens, "rfem_gw", float(k_value), masks, means, stds, thresholds, weights)
                )
                rows_for_aopc.append(("rfem_gw", float(k_value), agg[0].detach().cpu()))

        if args.aopc and idx <= aopc_limit:
            aopc_part, auc_part = faithfulness_rows_for_example(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                item=item,
                tokens=tokens,
                pred=pred,
                probs=probs,
                rows_for_rank=rows_for_aopc,
                fractions=args.aopc_k_fractions,
            )
            aopc_rows.extend(aopc_part)
            auc_rows.extend(auc_part)

        if not args.no_progress_bar:
            progress(idx, len(examples), progress_start, "Processing")
        elif args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(examples)):
            print(f"Processed {idx}/{len(examples)}")

    metrics = [
        "cls_self_share",
        "special_share",
        "content_share",
        "content_survival_rate",
        "row_nonzero_rate",
        "top_is_special",
        "normalized_entropy",
    ]
    method_summary = summarize(example_rows, ["method", "k"], metrics)
    rfem_summary = rfem_k_summary(example_rows, per_head_metrics)
    per_head_summary = summarize(
        per_head_metrics,
        ["method", "k", "head"],
        ["full_kept_pct", "cls_row_kept_pct", "cls_content_kept_rate", "cls_special_kept_rate", "threshold", "weight"],
    )
    top_rows = top_category_rows(example_rows)
    high_rows = high_sink_rows(example_rows)
    aopc_summary: list[dict[str, Any]] = []
    aopc_fraction_summary: list[dict[str, Any]] = []
    auc_summary: list[dict[str, Any]] = []
    if aopc_rows:
        aopc_summary = summarize(aopc_rows, ["method", "k"], ["comprehensiveness", "sufficiency"])
        aopc_fraction_summary = summarize(aopc_rows, ["method", "k", "fraction"], ["comprehensiveness", "sufficiency"])
    if auc_rows:
        auc_summary = summarize(auc_rows, ["method", "k"], ["iauc", "dauc"])

    write_csv(run_dir / "example_metrics.csv", example_rows)
    write_csv(run_dir / "method_summary.csv", method_summary)
    write_csv(run_dir / "rfem_k_summary.csv", rfem_summary)
    write_csv(run_dir / "per_head_metrics.csv", per_head_metrics)
    write_csv(run_dir / "per_head_summary.csv", per_head_summary)
    write_csv(run_dir / "top_token_categories.csv", top_rows)
    write_csv(run_dir / "high_sink_examples.csv", high_rows)
    if aopc_rows:
        write_csv(run_dir / "aopc_metrics.csv", aopc_rows)
        write_csv(run_dir / "aopc_summary.csv", aopc_summary)
        write_csv(run_dir / "aopc_by_fraction.csv", aopc_fraction_summary)
    if auc_rows:
        write_csv(run_dir / "auc_metrics.csv", auc_rows)
        write_csv(run_dir / "auc_summary.csv", auc_summary)
        write_csv(run_dir / "faithfulness_summary.csv", faithfulness_summary(aopc_summary, auc_summary))

    if not args.no_figures:
        make_figures(run_dir, example_rows, per_head_metrics)
        make_faithfulness_figures(run_dir, aopc_rows, auc_rows)

    accuracy = float(np.mean(correct)) if correct else 0.0
    elapsed = time.time() - start
    write_report(run_dir, args, examples, method_summary, rfem_summary, aopc_summary, auc_summary, accuracy, elapsed)

    print(f"\nDone. Outputs written to:\n  {run_dir}")
    print(f"Start with:\n  {run_dir / 'bert_variants_report.md'}")


if __name__ == "__main__":
    main()
