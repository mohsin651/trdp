# BERT RFEM Variants Analysis on SST-2 10k

This folder is separate from `sst2_quantitative_analysis/`. It covers the two
BERT variants that are not part of the existing 10k standard-RFEM analysis:

1. Sink-Aware RFEM, method tag `rfem_sa`
2. Gradient-Weighted RFEM, method tag `rfem_gw`

The script is self-contained and does not import from the existing analysis
folder or from the qualitative plotting scripts.

## Shared Setup

Model:

```text
textattack/bert-base-uncased-SST-2
```

Data file:

```text
../sst2_quantitative_analysis/data/SST-2/sst2_train_10k_seed42.tsv
```

Python environment:

```text
../sst2_quantitative_analysis/.venv
```

The model is loaded with:

```text
attn_implementation="eager"
```

because newer Transformers versions may use `sdpa`, which does not return
attention tensors when `output_attentions=True`.

## Methods

Baselines:

- `vanilla`: last-layer mean attention, CLS row.
- `rollout`: standard attention rollout, CLS row.

Variant 1:

- `rfem_sa`: Sink-aware RFEM.
- Per-head rollout is standard: `A + I`, row-normalize, multiply over 12 layers.
- K-sigma statistics are computed from the CLS row only, excluding `[CLS]`,
  `[SEP]`, and `[PAD]` columns.
- Threshold is applied to the full matrix.
- Filtering is value-preserving.
- Aggregation is unnormalized: `sum_h max(A_hat_h) * A_bar_h`.

Variant 2:

- `rfem_gw`: Gradient-weighted RFEM.
- Uses the same standard per-head rollout matrices.
- Head weights come from gradients:

```text
w_h = sum_l mean(ReLU(d logit_pred / d A_h^l) * A_h^l)
```

- Weights are ReLUed and normalized to sum to 1.
- K-sigma filtering is the same sink-aware filter as `rfem_sa`.

K values:

```text
0.0, 0.3, 0.5, 1.0
```

## Metrics

Per-example metrics:

- `cls_self_share`
- `special_share`
- `content_share`
- `content_survival_rate`
- `row_nonzero_rate`
- `top_is_special`
- `normalized_entropy`

Faithfulness metrics, when enabled with `--aopc` or `--faithfulness`:

- ERASER-style comprehensiveness
- ERASER-style sufficiency
- IAUC / insertion AUC
- DAUC / deletion AUC
- Fractions: `10%, 20%, 30%, 50%`

## Runners

Fast sink-aware-only run over the full 10k subset:

```bat
sst2_bert_variants_analysis\run_bert_variants_sa_only.bat
```

SA + GW over the full 10k subset:

```bat
sst2_bert_variants_analysis\run_bert_variants_10k.bat
```

SA + GW + AOPC over the full 10k subset:

```bat
sst2_bert_variants_analysis\run_bert_variants_aopc.bat
```

This runner also computes IAUC and DAUC from the same insertion/deletion
perturbation curves used for AOPC.

The Python script and all three batch runners use the full 10k subset by
default. For GW/AOPC this can be very slow on CPU. To use a runtime-safe cap
instead:

```bat
sst2_bert_variants_analysis\run_bert_variants_aopc.bat --aopc-max-examples 1000 --grad-weighted-max-examples 2000
```

## Output Files

Each run writes to:

```text
sst2_bert_variants_analysis/outputs/<run-name>/
```

Required CSVs:

- `example_metrics.csv`
- `method_summary.csv`
- `rfem_k_summary.csv`
- `per_head_metrics.csv`
- `high_sink_examples.csv`

Additional CSVs:

- `per_head_summary.csv`
- `top_token_categories.csv`
- `aopc_metrics.csv`
- `aopc_summary.csv`
- `aopc_by_fraction.csv`
- `auc_metrics.csv`
- `auc_summary.csv`
- `faithfulness_summary.csv`

Figures:

- `figures/method_metric_value_matrix.pdf`
- `figures/sink_share_boxplot.pdf`
- `figures/content_survival_bar.pdf`
- `figures/top_token_category_stacked_bar.pdf`
- `figures/entropy_boxplot.pdf`
- `figures/per_variant_sparsity_by_k.pdf`
- `figures/faithfulness_metric_matrix.pdf`
- `figures/aopc_comprehensiveness_sufficiency_boxplot.pdf`
- `figures/iauc_dauc_bar.pdf`
- `figures/head_sparsity_heatmap_rfem_sa.pdf`
- `figures/head_sparsity_heatmap_rfem_gw.pdf` when GW is enabled

Auto-written report:

```text
bert_variants_report.md
```
