# RFEM-on-BERT / CLIP — Complete Project Context

This file is the **single source of truth** for a future Claude session
picking up this project cold. It captures: what was built, why each choice
was made, every user-driven decision, every deviation from the paper, the
code layout, the output layout, and the "don't break these" guardrails.

> If anything in this file contradicts the code, the **code wins** — update
> this file. Memory of past decisions is only useful if it stays true.

## Files at a glance

| File | Status | Output dir | Purpose |
|---|---|---|---|
| `rfem_pipeline.py` | active | `figs_pdf/<sid>/` | **BERT/SST-2** — standard RFEM. Works correctly. 10 sentences × 34 PDFs = 340. |
| `rfem_bert_sink_aware.py` | active | `figs_bert_sink_aware/<sid>/` | **BERT sink-aware** — fixes CLS/SEP sink. Content words surface clearly. 10 sentences. |
| `rfem_clip_pipeline.py` | active | `figs_pdf_clip/<sid>/` | **CLIP text** (causal). K-σ μ/σ over lower triangle. **Bars empty** — BOS sink, reportable phenomenon. 10 sentences. |
| `rfem_clip_text_sink_aware.py` | active | `figs_clip_text_sink_aware/<sid>/` | **CLIP text sink-aware** — fixes BOS/EOS sink. Content tokens surface (small values ~0.001–0.004). 10 sentences. |
| `rfem_clip_eos_pipeline.py` | active | `figs_pdf_clip_eos/<sid>/` | **CLIP text** alt calibration — EOS row μ/σ. Also empty bars. Kept as diagnostic evidence. |
| `rfem_clip_vision_pipeline.py` | active | `figs_pdf_clip_vision/<img_id>/` | **CLIP vision standard** (ViT-B/32). Stage-subfoldered. 10 images (I1–I10). |
| `rfem_clip_vision_sink_aware.py` | active | `figs_clip_vision_sink_aware/<img_id>/` | **CLIP vision sink-aware** — fixes CLS-self sink. Darkened variable-alpha jet overlays. 10 images. |
| `report_rfem_comparison.tex` | active | `report_rfem_comparison.pdf` | LaTeX visual comparison report. 10 pages. Methodology page frozen. |
| `input_images/` | input dir | — | Drop images here for vision pipelines (recursively globbed, renamed I1–I10). |
| `context.md` | meta | — | You are reading it. |

**Both CLIP text files are kept intentionally** — together they demonstrate
the BOS-sink failure of RFEM-on-causal-text from two different statistical
calibrations. They are evidence, not bugs. Do not "fix" them by changing
the threshold logic without checking with the user. See §16 for the story.

**CLIP vision** is the paper's home substrate (bidirectional attention,
[CLS] at position 0). Math works directly per Eqs. 2–5 — see §17 for the
full design notes (structured folders, per-layer rollout grids, PowerNorm
γ=0.3 applied to RFEM only, magma colormap, ViT-B/32 vs B/16 choice, CLS-CLS
dominance phenomenon).

---

## 1. What this project is

Re-implementation of **RFEM** (Rollout Feature Explanation Method) — from
Ayyar, Benois-Pineau, Zemmari 2025, *"There is More to Attention: Statistical
Filtering Enhances Explanations in Vision Transformers"* (arXiv:2510.06070) —
applied to **BERT (SST-2)** for **text classification** instead of a Vision
Transformer for image classification.

The point: RFEM was designed for ViT-B16 (12 layers × 12 heads, 197 tokens).
BERT-base is also 12 layers × 12 heads, so the math transfers cleanly. The
"tokens" are now wordpieces rather than image patches, and the visualisations
read along token strings instead of 14×14 grids.

**Model used**: `textattack/bert-base-uncased-SST-2` (a BERT-base fine-tuned
on SST-2 sentiment classification). Forward pass only — no gradients
required, no training.

**Why this exists**: presentation / report illustrations for the user's
research. The PDFs are composed manually into the paper, so the script must
output one PDF per figure (no multi-panel side-by-sides — the user arranges
those in their layout tool).

---

## 2. Quick start

```bash
cd /home/mohsin/UB_DFILES/IPCV/UAM/TRDP/Final
source .venv/bin/activate
python rfem_pipeline.py
```

Runtime: about ~30–60 seconds for all 10 sentences on CPU. Writes 34 PDFs
per sentence into `figs_pdf/<sid>/`. Total ≈340 PDFs.

Re-running overwrites in place — slug-named files, so no orphan accumulation.

---

## 3. Files in the project root

| File | Role |
|------|------|
| `rfem_pipeline.py` | The script. Single entry point. ~720 lines. |
| `RFEM_BERT_presentation_final_v2 (1).ipynb` | The original notebook this was derived from. Kept for reference only — the script supersedes it and is now the canonical implementation. |
| `2510.06070v1.pdf` | The official RFEM paper. Read pages 4 (Methodology / Eqs. 1–5) and 15 (K-parameter ablation) before touching the pipeline. |
| `context.md` | This file. |
| `figs_pdf/` | Output. One subfolder per sentence (`S1/`, `S2/`, …, `S10/`). |
| `.venv/` | Local Python 3.12 virtualenv with `torch`, `transformers`, `matplotlib`, `numpy`. |
| `__pycache__/rfem_pipeline.cpython-312.pyc` | Compile cache. Ignore. |

---

## 4. The RFEM paper — formulas the code implements

The paper's RFEM pipeline has three stages (called "key stages" in §3.1):

### Stage A — Layerwise aggregation (per-head rollout)

For each head `h ∈ {1, …, H}` and layer `l ∈ {1, …, L}`:

$$\hat{A}_h^{(l)} = A_h^{(l)} + I \quad\text{(Eq. 2 — residual added)}$$

$$\hat{A}_h = \prod_{l=1}^{L} \hat{A}_h^{(l)} \quad\text{(Eq. 3 — multiplied across layers)}$$

Key point: heads are **never averaged**, unlike the standard Abnar–Zuidema
rollout. They are kept separate to expose head-specific behaviour.

### Stage B — Statistical thresholding (K-σ filter)

For each head `h`, compute `μ_h` and `σ_h` from `Â_h`, then:

$$\bar{A}_h(i,j) = \begin{cases} 1 & \text{if } \hat{A}_h(i,j) \geq \mu_h + K\sigma_h \\ 0 & \text{otherwise} \end{cases} \quad\text{(Eq. 4 — binary)}$$

K controls the threshold (higher K = stricter). Paper sweeps K ∈ {-0.5, 0,
0.5, 1, 1.5, 2} in the appendix.

### Stage C — Weighted head aggregation

$$A_{\text{rfem}} = \sum_{h=1}^{H} w_h \cdot \bar{A}_h, \quad w_h = \max(\hat{A}_h) \quad\text{(Eq. 5)}$$

The weight per head is the maximum value in that head's rolled-out matrix.
Heads with stronger peaks contribute more.

### Stage D (BERT-specific) — CLS-row token importance

For text, the [CLS] token attends to all others. So token-`j` importance is
just `A_rfem[0, j]`. (For ViT this is the row of the CLS query over all
patches, then reshaped to a 14×14 heatmap.)

---

## 5. **User modifications** (these are not in the paper — DO NOT REMOVE)

### 5a. Value-preserving K-σ filter (instead of binary 1/0)

**Paper says**: `Ā_h(i,j) = 1[Â_h(i,j) ≥ μ_h + Kσ_h]` (binary).

**User says** (very first instruction in the project):
> "in rfem filtering you will use value instead of assigning 1 and 0 to 0"

**My implementation** at `rfem_pipeline.py: rfem_k_sigma_filter`:
```python
mask_h = torch.where(R_h >= threshold_h, R_h, torch.zeros_like(R_h))
```

i.e.

$$\bar{A}_h(i,j) = \begin{cases} \hat{A}_h(i,j) & \text{if } \hat{A}_h(i,j) \geq \mu_h + K\sigma_h \\ 0 & \text{otherwise} \end{cases}$$

The surviving entries keep their actual rolled-attention value (not 1). The
intuition: weighting by survival is fine, but you also want magnitude
information — a value that just barely passes the threshold shouldn't count
the same as one that strongly exceeds it.

This is the **whole point of the project** from the user's perspective —
the "modification" of standard RFEM that they're investigating.

### 5b. Weighted aggregation (not the supervisor's `1/H` average)

The reference notebook had a "supervisor's unweighted" variant:
$$A_{\text{rfem}} = \frac{1}{H}\sum_h \bar{A}_h$$

User said (also at the start):
> "make sure the rfem is weighted"

So the code uses the **original RFEM weighted form** (paper Eq. 5) with
`w_h = max(Â_h)`. The supervisor's version is **not** in the code.

### 5c. Histogram excludes [CLS], [SEP], [PAD]

The histogram in `*_M3_step2_hist_*.pdf` is the distribution of values inside
the rolled matrix Â_h for the debug head, used to visualise the K-σ
threshold.

Without filtering, [CLS] and [SEP] dominate the rolled values (residual-plus-
identity acts as an attention sink for special tokens), so the threshold line
appears in a misleading place relative to the bulk of "real" tokens.

User said:
> "in histogram i am removing CLS and SEP token did you notice that"

So the histogram does:
```python
non_special_idx = np.array([i for i,t in enumerate(tokens) if t not in SPECIAL_TOKENS])
vals_flat = mat_h[np.ix_(non_special_idx, non_special_idx)].flatten()
```

But — and this is **critical** — the μ and threshold drawn as vertical lines
on the histogram are **still the actual filter values** (computed from the
full matrix). Only the histogram bars exclude special tokens. User confirmed:
> "yes yes ofc.. i want them i dont wanna change the attention but just change viz.. haha thanks!"

The title carries `([CLS] / [SEP] / [PAD] excluded)` so this is obvious to
the reader.

### 5d. No `/Σw_h` normalisation in aggregation

An earlier version had `A_rfem = (Σ w_h · Ā_h) / Σ w_h` (a weighted average).
After a back-and-forth where the user inspected the paper formula carefully,
they asked to remove the division so the code matches Eq. 5 exactly:
> "no don't do that.. change it replot the graphs!"

(Context: they had just said "are you sure.. check again!" about whether the
implementation matched the paper, and I flagged the `/Σw_h` deviation. Their
response was: drop it.)

Current state matches paper Eq. 5 verbatim. **Do not put `/Σw_h` back**.

---

## 6. Pipeline-level deviations from the paper (not user-driven, but worth knowing)

### 6a. Step 1 row-normalisation

Paper Eq. 3 literally reads `Π(A_h^(l) + I)` with no normalisation. In
practice, each row of `(A + I)` sums to ~2 (since `A` is row-stochastic and
`I` adds 1 on the diagonal), so a 12-layer product without normalisation
gives diagonal entries of ~2^12 and the K-σ filter becomes meaningless.

The code does **row-normalise after `+I`**:
```python
A_norm = A_plus_I / A_plus_I.sum(dim=-1, keepdim=True).clamp(min=1e-9)
mat    = A_norm @ mat
```

This is the standard Abnar–Zuidema convention. The paper builds on
Abnar–Zuidema and the convention is implied. The original notebook also
normalises. If the user ever questions this, the answer is: numerical
stability + standard convention.

### 6b. Vanilla baseline computed two ways

The notebook only had "vanilla = last-layer mean over heads". The user later
asked to **also** add Layer 1 / Head 1 raw attention as another baseline:

> "can you also plot first layer 1st head in the vanilla attention... please"

So Method 1 now emits **four** PDFs per sentence (two views × matrix+bar).

---

## 7. Code architecture (file walkthrough)

The script is one file because it's tightly coupled and small. Top-to-bottom
structure:

### 7.1 Imports & plot style (lines ~1–40)
- Stock `torch`, `numpy`, `matplotlib`, `transformers`.
- **Matplotlib defaults** for fonts (NOT Times New Roman / serif — user
  explicitly asked for matplotlib defaults).
- Sizes bumped: `FONT_SZ=18, TITLE_SZ=20, TICK_SZ=16, SUPTI_SZ=24, DPI=150`.
- Why big: user wants zoomable PDFs where labels stay legible.

### 7.2 Output directory machinery (lines ~40–80)

- `OUT_DIR = ./figs_pdf` — created at import time.
- `_CURRENT_OUT_DIR` — module-level global, **mutated per sentence** by
  `process_sentence` so each sentence's PDFs land in `figs_pdf/<sid>/`.
- `save_fig_pdf(fig, name)` — wraps each figure in its own `PdfPages`. One
  PDF per figure. Slug-cleans the name.
- **Do not** factor this differently: the user wants one PDF per figure,
  and the per-sentence subfolder is keyed off this single global.

### 7.3 Config block (lines ~85–120)

```python
MODEL_NAME     = "textattack/bert-base-uncased-SST-2"
DEBUG_HEAD     = 0     # which head's intermediate matrices to show
HEAD_TO_SHOW   = 0     # which head's final rolled matrix to show solo
K_VALUES       = [0.0, 0.3, 0.5, 1.0]
DROP_SPECIAL   = True
SPECIAL_TOKENS = ("[CLS]", "[SEP]", "[PAD]")
PUNCT_SET      = set(".,!?;:'\"()-")
```

The `sentences` list now has 10 entries (S1–S10). All 10 run by default.

### 7.4 Plot helpers (lines ~125–340)

Every plot helper has the same shape: build the figure, save via
`save_fig_pdf` if `save_name` is given, return the figure.

| Function | Purpose | μ/σ behaviour |
|---|---|---|
| `plot_matrix_heatmap` | Single matrix heatmap. Used for vanilla matrices, rollout matrix, intermediate Step 1 matrices, K-σ filtered matrix, aggregated map. | μ and σ go in the **title** (above the axes). NEVER as a text-box overlay on the matrix. |
| `plot_token_bar` | Vertical bar chart over tokens. Used for CLS rows (vanilla, rollout, RFEM). | n/a |
| `plot_all_head_matrices` | **3×4 grid** of all 12 head rolled matrices in ONE figure. The only multi-panel figure in the project (user asked for grids to stay combined). | μ and σ per head are in the per-axis title. Cell annotations use `fontsize=6` and `.2f` so they don't overflow (user explicitly asked for this). |
| `plot_k_sweep_importance` | The 1×4 grid showing token importance at K ∈ {0.0, 0.3, 0.5, 1.0} side-by-side. | n/a |
| `plot_rfem_sparsity_per_head` | Bar chart: % entries surviving per head at a given K. | n/a |
| `plot_value_histogram` | Histogram of rolled attention values for the debug head, with μ and threshold drawn on. | Drops [CLS]/[SEP]/[PAD] rows AND columns before flattening. Threshold line still uses the full-matrix μ. |

### 7.5 Model loading (lines ~345–355)

`BertTokenizer` + `BertForSequenceClassification(output_attentions=True)`.
Loaded once, set to eval mode.

### 7.6 Core RFEM functions (lines ~360–500)

| Function | Signature | Returns |
|---|---|---|
| `get_attentions(text)` | Tokenises and forward-passes. | `(tokens, attn_all (12,12,T,T), logits, outputs)` |
| `attention_rollout(attentions)` | Standard Abnar–Zuidema rollout (heads averaged per layer). | `(rollout_matrix (T,T), debug)` |
| `rfem_per_head_rollout(attentions, debug_head=0)` | Per-head rollout, keeping heads separate. Captures intermediate matrices for `debug_head`. | `(head_rollouts (H,T,T), step_debug list)` |
| `rfem_k_sigma_filter(head_rollouts, k)` | **Value-preserving** K-σ filter. Prints μ/σ/threshold/kept table to console. | `(head_masks (H,T,T), means, stds, thresholds)` |
| `rfem_aggregate_heads_weighted(head_masks, head_rollouts)` | Weighted sum with `w_h = max(Â_h)`. **No `/Σw_h`** (matches paper Eq. 5). Prints per-head weights to console. | `(aggregated (T,T), weights (H,))` |
| `rfem_extract_token_relevance(agg, tokens, drop_special=True)` | CLS row + filtered subset. | `(cls_row, cls_filtered, filtered_token_labels)` |
| `compute_and_print_mu_sigma(rolled, sid)` | Console table of μ, σ, max, min per head. | `list[(mu, sigma)]` |

### 7.7 `process_sentence(sid, text, label)` (lines ~510–700)

The main per-sentence driver. Order of operations:

1. **Set `_CURRENT_OUT_DIR`** to `figs_pdf/<sid>/` (this is where all PDFs go).
2. **Forward pass** → print prediction and tokens.
3. **Method 1 — Vanilla**:
   - L1H1 raw matrix + CLS bar (2 PDFs)
   - Last-layer mean matrix + CLS bar (2 PDFs)
4. **Method 2 — Standard rollout**:
   - Rollout matrix + CLS bar (2 PDFs)
5. **Method 3 — RFEM**:
   - **Step 1**: per-head rollout
     - Intermediate diagnostics for debug head Layer 1: raw, +I, normalised (3 PDFs)
     - Final rolled matrix for HEAD_TO_SHOW (1 PDF)
     - All-12-heads grid with μ/σ (1 PDF — the big one)
   - **Step 2 + 3 + 4 loop over K ∈ K_VALUES**:
     - Histogram with μ and threshold lines (1 PDF/K)
     - Filtered matrix for debug head (1 PDF/K)
     - Per-head sparsity bar (1 PDF/K)
     - Weighted aggregated map (1 PDF/K)
     - RFEM CLS bar (filtered tokens) (1 PDF/K)
   - **Stage 6 standalone bars** (emitted once, outside the K-loop):
     - Vanilla CLS bar over filtered tokens (`*_M4_vanilla_filtered_cls_bar.pdf`)
     - Rollout CLS bar over filtered tokens (`*_M4_rollout_filtered_cls_bar.pdf`)
   - **Stage 7**: K-sweep summary (1 PDF)

**Total: 34 PDFs per sentence**.

### 7.8 `__main__`

Loops over `sentences` calling `process_sentence`. Prints the final
`Done. All PDFs written to: …`.

---

## 8. Output layout & reading order

```
figs_pdf/
├── S1/
│   ├── S1_M1_vanilla_L1H1_matrix.pdf           # Stage 1, plot 1
│   ├── S1_M1_vanilla_L1H1_cls_bar.pdf          # Stage 1, plot 2
│   ├── S1_M1_vanilla_matrix.pdf                # Stage 1, plot 3
│   ├── S1_M1_vanilla_cls_bar.pdf               # Stage 1, plot 4
│   ├── S1_M2_rollout_matrix.pdf                # Stage 2, plot 1
│   ├── S1_M2_rollout_cls_bar.pdf               # Stage 2, plot 2
│   ├── S1_M3_step1_Araw_h1_L1.pdf              # Stage 3, plot 1
│   ├── S1_M3_step1_AplusI_h1_L1.pdf            # Stage 3, plot 2
│   ├── S1_M3_step1_Anorm_h1_L1.pdf             # Stage 3, plot 3
│   ├── S1_M3_step1_final_rollout_h1.pdf        # Stage 3, plot 4
│   ├── S1_M3_step1_all12heads_grid.pdf         # Stage 3, plot 5 — the BIG one
│   ├── S1_M3_step2_hist_h1_K0.0.pdf            # Stage 4, K=0.0
│   ├── S1_M3_step2_mask_h1_K0.0.pdf
│   ├── S1_M3_step2_sparsity_K0.0.pdf
│   ├── S1_M3_step3_agg_weighted_K0.0.pdf       # Stage 5, K=0.0
│   ├── S1_M4_rfem_filtered_cls_bar_K0.0.pdf    # Stage 6 (RFEM bar), K=0.0
│   ├── … (repeats for K=0.3, 0.5, 1.0) …
│   ├── S1_M4_vanilla_filtered_cls_bar.pdf      # Stage 6 (vanilla bar — K-independent)
│   ├── S1_M4_rollout_filtered_cls_bar.pdf      # Stage 6 (rollout bar — K-independent)
│   └── S1_M3_step4_K_sweep_importance.pdf      # Stage 7 — summary
├── S2/   (same 34 files)
├── S3/   (same 34 files)
├── …
└── S10/  (same 34 files)
```

### Recommended reading order (per sentence)

1. **Stage 1 — Vanilla**: see what a single-head-single-layer attention map
   looks like (L1H1), then see what averaging over heads at the last layer
   looks like (the conventional "vanilla attention" baseline). Conclusion:
   only one layer → loses information.
2. **Stage 2 — Standard rollout**: the field's standard improvement. Heads
   averaged per layer, +I, multiplied. Better, but blurs head differences.
3. **Stage 3 — Per-head rollout**: walk through one layer's transformation
   (Araw → AplusI → Anorm), then see all 12 heads after full rollout side-by-
   side. Heads disagree → motivates per-head filtering.
4. **Stage 4 — K-σ filter (pick K=0.5 first)**: histogram shows the
   distribution and threshold; mask shows what survived for one head;
   sparsity bar shows survival rate across all heads.
5. **Stage 5 — Weighted aggregation**: heads recombined.
6. **Stage 6 — Three CLS bars on the same filtered token set** (vanilla,
   rollout, RFEM at chosen K). The headline comparison the user composes in
   their paper.
7. **Stage 7 — K-sweep summary**: the effect of the threshold parameter.

If only five plots: pick **Stage 1 plot 1, Stage 2 plot 2, Stage 3 plot 5,
Stage 5, Stage 7**.

---

## 9. Conversation history — every decision and its rationale

Chronological log of decisions baked into the current code:

| # | Decision | Driver | Rationale |
|---|----------|--------|-----------|
| 1 | Start from the notebook | User: "see the code... implement that here again" | Notebook had the working logic; just needed to be cleaned up. |
| 2 | RFEM aggregation is **weighted** | User: "make sure the rfem is weighted" | The notebook's `unweighted (1/H)` version was the "supervisor's" version; the user wants standard RFEM Eq. 5. |
| 3 | K-σ filter is **value-preserving** (not 1/0) | User: "in rfem filtering you will use value instead of assigning 1 and 0 to 0" | The main "modification" being investigated. |
| 4 | One PDF per figure | User: "make everyplot in pdf in matplotlib" | Easier to manually arrange in the paper. |
| 5 | Multi-grid figures (12-head grid, K-sweep) stay one figure | User: "the grid based like all 12 layer vizualization and all should be in a same graph" | These are meant to be read holistically. |
| 6 | Matplotlib default font (not Times New Roman) | User: "matplotlib default font but big enough so that zooming in should see the numbers clearly" | Default = portable; big = zoomable. |
| 7 | μ and σ **above** the matrix in the title | User: "move the mu and sigma above matrix not on the matrix" | Text box overlay was ugly. |
| 8 | Only sentence S1 initially | User: "i just want it for the first sentence remove all others" | Iteration speed. |
| 9 | Add L1H1 vanilla view | User: "can you also plot first layer 1st head in the vanilla attention" | Wanted a "before" baseline that's truly raw. |
| 10 | Reading order documented in chat | User: "tell me in which order i should understand the graphs" | The chat answer is reproduced in §8 of this file. |
| 11 | Reduce annotation font on 12-head grid | User: "in all 12 heads... values are coming out of cells. just make them smaller" | Fixed to `fontsize=6` and `.2f` (was 10 and `.3f`). |
| 12 | Histogram drops [CLS]/[SEP]/[PAD] | User: "in histogram i am removing CLS and SEP token did you notice that" | Special tokens dominate the distribution; threshold line stays from full matrix. |
| 13 | Confirm filter doesn't change, only viz | User: "i dont wanna change the attention but just change viz" | μ/σ used by the filter is on the full matrix. |
| 14 | Method 1+2+3 comparison as **separate PDFs, not 3-panel** | User: "produce all of them seperately not in graph ok ... adjusting in apper would be easier" | Triggered me to delete the `plot_three_way_comparison` function entirely; now each method is its own bar plot. |
| 15 | Question: is it weighted? | User: "is this weighted or unweighted aggregation?" | I confirmed weighted via inspection of `rfem_aggregate_heads_weighted`. |
| 16 | Question: are you sure (deeper)? | User: "are you sure.. check again!" | I flagged that my code had `/Σw_h` extra normalisation vs. the slide formula. |
| 17 | Removed `/Σw_h` to match paper Eq. 5 | User: "no don't do that.. change it replot the graphs!" | Final aggregation form matches the paper verbatim. |
| 18 | Extended to all 10 sentences | User: provided S2–S10 list and "do the same for 10 sentences" | Sentences hard-coded in the `sentences` list. |
| 19 | Per-sentence subfolders | (implementation choice driven by 18) | Avoids 340 files in one directory; one click per sentence. |
| 20 | This `context.md` | User: "create a context.md... to maintain the context for future claude session" | You are reading it. |
| 21 | CLIP **text** port → causal pipeline | User asked "do the same with CLIP text encoder" | Created `rfem_clip_pipeline.py` with lower-triangle μ/σ. **Bars empty** — BOS sink. |
| 22 | EOS-row-calibrated variant of CLIP text | User asked for an alternative calibration | Created `rfem_clip_eos_pipeline.py`. Also empty bars. |
| 23 | Keep both CLIP text pipelines as evidence | User: "see this is the phenonmenon so i have to report.. so leave it like that!" | Both saved as parallel diagnostic evidence; §16 documents it. |
| 24 | CLIP **vision** pipeline (`rfem_clip_vision_pipeline.py`) | User: "do the same for CLIP vision encoder" | The paper's home substrate; works correctly. |
| 25 | ViT-B/32, not B/16 | Implementation choice during brainstorming | B/32's 7×7=49 patches keep per-patch value annotations readable. |
| 26 | Values-overlay variant (numbers printed on patches) | User: "in addition for everyimage extract CLS row and lay that over on image with values so that i know which token is projecting which part of the image" | Implemented as `with_values=True` flag in `plot_overlay`. |
| 27 | Auto-discover images recursively from `input_images/` | User's images were nested in `input_images/images/` subfolder | Changed `glob` → `rglob`. |
| 28 | Matrices in **magma** colormap | User: "could you please change the color of matrices to that magenta theme" → then screenshot of magma | Initial guess was `RdPu`; corrected to `magma` after the screenshot. |
| 29 | Per-layer rollout matrix grid (per-head Head 1) | User: "i want all 12 layer matrix visualization as well after rollout before statistically filtering in RFEM" | `plot_per_layer_rollout_grid` added; captured cumulative `mat` per layer for debug head. |
| 30 | Remove histograms from CLIP vision | Same message as #29: "i am not interested in those histogram kind of viz" | `plot_value_histogram` deleted from CLIP vision; histogram calls removed. |
| 31 | Structured stage-subfolder layout per image | User: "please organize the figures of clip vision in structured folders its hard to see!" | `00_original/` → `06_rfem_step4_K_sweep/`. `save_fig_pdf` extended to accept paths with `/`. |
| 32 | PowerNorm γ=0.3 in CLIP vision | User: "is there any way to use more fine grained color matrix because values are small so 0.014 is also black as 0?" | Display-only; applied to all matrix + overlay imshow calls. |
| 33 | PowerNorm **RFEM-only** (not vanilla/rollout) | User: "please do that gamma norm thing for increasing range in heatmaps only in RFEM not in vanilla and rollout!" | `use_power_norm` param added; defaults to True; `False` passed for all 01_/02_ calls. |
| 34 | Heads-averaged per-layer rollout matrix AND overlay grids | User: "attention rollout layover per layer I need and also the attention matrix for all images" | `plot_per_layer_overlays_grid` added; both go in `02_rollout/`, no PowerNorm. |
| 35 | Rejected fine-tuning CLIP on Flickr 8k | User asked, I argued against | Reasons: Flickr 8k too small; RFEM is visualisation, not model improvement; pretrained attention is the object of study. |
| 36 | Rejected caption-conditioned RFEM-Class (both text and vision) | Discussed twice in two different exchanges | User said no both times. Method documented in §17 if they ever change their mind. |
| 37 | This big `context.md` update | User: "now add the things in context which are not there (all the work we have done till now)" | Captures #21–#36 plus §17 (CLIP vision design notes), updated guardrails §10, refreshed TL;DR §15. |
| 38 | Sink-aware RFEM implemented for all 3 encoders | Prof. Benois-Pineau's suggestion; user: "implement the sink aware thing" | `rfem_bert_sink_aware.py`, `rfem_clip_text_sink_aware.py`, `rfem_clip_vision_sink_aware.py`. Fix: μ_h/σ_h from read-out row excluding sink columns. See §18. |
| 39 | Darkened variable-alpha jet overlay for CLIP vision | User: "the maps are like too much on image" / "perfect thank you" | Background darkened ×0.6; alpha proportional to `sqrt(norm_spatial)*0.65`. Applied to both vision scripts. See §19. |
| 40 | Input images renamed I1–I10 | User: "rename images to I1, I2 and re run both vision codes" | Avoids hash names in report captions; both vision scripts regenerated. |
| 41 | SA-RFEM K-sweep titles use "SA-RFEM" not "RFEM" | User: "in sink aware rfem in vision code images in label use SA-RFEM" | Distinguishes standard and sink-aware outputs visually. |
| 42 | Visual comparison report `report_rfem_comparison.tex` created | User: "just create a simple report with all visualisations" | LaTeX report, 10 pages, one section per encoder, methodology page frozen. See §21. |
| 43 | Report restructured to exact layout spec | User: detailed per-section layout instructions (multiple rounds) | BERT/CLIP-Text: vanilla→rollout→head rollout→agg matrix→std 4K row→SA K-sweep. Vision: original+vanilla+rollout→head overlays→agg matrix→agg overlay→std 4K→SA 4K. |
| 44 | CLIP-Text K-sweep changed to show raw (unnormalised) bars | User: "instead of using normalize bars, use un normalized ones" | Reverted normalisation in `plot_k_sweep_importance`; bars now show raw 0.001–0.004 values. |
| 45 | Stage D SA-RFEM uses single K-sweep importance figure | User: "here are the location... they are already in a one" pointing to K-sweep file | Replaced 4 individual subfigures with `S1_M3_step4_K_sweep_importance.pdf` (full width) for both BERT and CLIP-Text SA-RFEM. |
| 46 | CLIP-Vision report: added weighted aggregation matrix step | User: "you missed few steps" | Added the 50×50 aggregation matrix comparison (std vs SA) BEFORE the overlay comparison. Both `matrix.pdf` and `overlay.pdf` shown separately. |
| 47 | `context.md` updated again | User: "update in context.md file what we have done till now" | This update. Captures §18–§22. |

### Conversational style notes

The user prefers:
- **Short, plain-language answers** when explaining concepts. Not academic.
- **Math when it's actually clarifying**, not for show.
- **Direct verdicts** ("yes / no / it's weighted") followed by code citation
  (`rfem_pipeline.py:NN`).
- When asked "are you sure?" — re-derive, don't just restate. They are
  cross-checking against the paper.
- They sometimes change their mind mid-instruction; **back out cleanly**
  (e.g., the 3-panel comparison was built and then ripped out within one
  exchange).
- They typo and write fast — interpret intent, don't pedant.

---

## 10. **Do not break these** — guardrails

A future session must NOT, without an explicit ask:

1. **Re-add `/Σw_h`** to `rfem_aggregate_heads_weighted`. The current form
   matches paper Eq. 5 exactly. Removing this was a deliberate, considered
   choice (decision #17).
2. **Binarise the K-σ filter** back to 1/0. The value-preserving form is
   the project's whole reason for existing.
3. **Switch matplotlib fonts to Times New Roman / serif**. Default is the
   ask. Sizes are bumped on purpose.
4. **Overlay μ/σ on the matrix** as a text box. They go in the title.
5. **Make the comparison plots multi-panel**. Each method is its own PDF.
   The user composes panels manually in their paper.
6. **Bump the 12-head grid annotation font** above 6 or change the format
   from `.2f`. Values overflow.
7. **Drop the histogram's CLS/SEP/PAD exclusion**. It's viz-only and matters
   for readability.
8. **Compute μ/σ on the filtered subset for the filter itself**. The filter
   uses full-matrix stats; only the histogram visualisation excludes special
   tokens.
9. **Remove the L1H1 vanilla plot**. User added it deliberately.
10. **Add the supervisor's `(1/H) Σ Ā_h` aggregation back**. It's the
    version the user rejected.
11. **"Fix" the empty bars in the CLIP text pipelines** by changing μ/σ
    source, introducing BOS exclusion, switching `w_h` to EOS-row max,
    etc. The user has decided these empty bars are the **phenomenon to
    report**. Do not produce a "working" CLIP-text variant unless the
    user explicitly asks. See §16.
12. **Delete or rename either CLIP text pipeline file.** Both are kept
    as parallel evidence.

### CLIP vision pipeline (`rfem_clip_vision_pipeline.py`) — guardrails

13. **Switch CLIP vision from ViT-B/32 to B/16** without asking. B/32 was
    chosen so the values-overlay (per-patch numbers on the image) stays
    readable (32×32 px per patch). B/16 would make per-patch numbers
    illegible.
14. **Apply PowerNorm γ to vanilla or rollout figures.** PowerNorm γ=0.3
    is **deliberately RFEM-only** — applied to `03_*`, `04_*`, `05_*`,
    `06_*` folders, but NOT to `01_vanilla/` or `02_rollout/`. The user
    wants the baselines rendered with plain linear normalisation so
    they look like "raw" attention/rollout — only RFEM gets the
    contrast boost. `use_power_norm=False` is passed for every
    vanilla/rollout call in `process_image`.
15. **Re-add the value-distribution histogram to CLIP vision.** User
    said histograms aren't informative for vision; sparsity bar + the
    spatial overlays already cover what the filter does. The
    `plot_value_histogram` function was removed entirely from
    `rfem_clip_vision_pipeline.py`.
16. **Flatten the per-image output back to a single folder.** The
    structured layout (`00_original/`, `01_vanilla/`, `02_rollout/`,
    `03_rfem_step1_per_head/`, `04_rfem_step2_filter/K{K}/`,
    `05_rfem_step3_aggregate/K{K}/`, `06_rfem_step4_K_sweep/`) was a
    user request — they found the flat 42-PDF folder hard to navigate.
17. **Drop the per-layer rollout grids.** Two were added on user
    request: `02_rollout/per_layer_rollout_matrices_grid.pdf` and
    `02_rollout/per_layer_rollout_overlays_grid.pdf` (heads-averaged),
    plus the existing
    `03_rfem_step1_per_head/per_layer_rollout_grid_h1.pdf` (per-head).
18. **Drop the CLS-self entry in overlays** — `cls_row_to_spatial` does
    `patch_scores = cls_row[1:]`. CLS is a position-less aggregator
    token, so it can't be drawn on the image. This is a fundamental
    architectural constraint, not a choice. Don't try to map `(0,0)`
    onto a pixel.
19. **Fine-tune CLIP on Flickr 8k** "to improve RFEM". User explicitly
    rejected this — fine-tuning is overkill for visualisation work
    and Flickr 8k is too small for a model of CLIP's scale. See §17
    for full reasoning.
20. **Build RFEM-Class** unless explicitly asked. We've discussed it
    twice (CLIP text caption-conditioned; CLIP vision caption-
    conditioned) — the user said no both times.

---

## 11. Numbers & shapes (for sanity-checking)

For an SST-2 sentence of ~15 wordpieces (including [CLS], [SEP]):

| Tensor | Shape | Notes |
|---|---|---|
| `attn_all` | `(12, 12, T, T)` | Layers × heads × seq × seq. T ≈ 13–18 for these sentences. |
| `outputs.attentions` | tuple of 12, each `(1, 12, T, T)` | Same data, list form. |
| `head_rollouts` | `(12, T, T)` | Per-head rolled matrices. After Step 1. |
| `head_masks` | `(12, T, T)` | Per-head **value-preserving** filtered matrices. After Step 2. |
| `weights` | `(12,)` | `max(Â_h)` per head. Typical range 0.2–0.6. |
| Aggregated map | `(T, T)` | Weighted sum. Row 0 = CLS importance. |

Per-head means after rollout are uniform at `1/T` (e.g. `0.0667` for T=15,
`0.0625` for T=16) — that's a sanity check that row-normalisation in Step 1
is doing the right thing (each row sums to 1, matrix average = 1/T).

---

## 12. Style choices baked in

| Setting | Value | Reason |
|---|---|---|
| `font.family` | (default, matplotlib sans-serif) | User asked for defaults. |
| `font.size` | 18 | Zoom legibility. |
| `axes.titlesize` | 20 | Title carries μ/σ; needs to be readable. |
| `xtick.labelsize`, `ytick.labelsize` | 16 | Tokens are short; can afford big ticks. |
| `figure.dpi`, `savefig.dpi` | 150 | Crisp PDFs without bloat. |
| `savefig.bbox` | `tight` | No edge clipping. |
| 12-head grid annotation size | 6 (with `.2f`) | Overflow prevention. |
| Single-matrix annotation size | 12 (with `.2f`) | Plenty of room. |
| Color schemes (BERT / CLIP text) | `Blues` for matrices, `Greens` for filtered masks, orange `#ff7f0e` for vanilla bars, blue `#1f77b4` for rollout bars, green `#2ca02c` for RFEM bars | Visual distinction across methods. |
| Color scheme (CLIP vision matrices) | `magma` everywhere | User asked for the "purple-magenta theme" matching the screenshot they shared. |
| Color scheme (CLIP vision overlays) | `jet` with α = 0.55 | Paper convention; sharp visibility against image. |
| PowerNorm γ (CLIP vision) | `0.3`, **RFEM-only** | Compresses high values, expands small ones — see §17. NOT applied to vanilla or rollout figures. |

---

## 13. Environment & reproducibility

- Python 3.12, virtualenv at `.venv/`.
- Key deps (from `.venv/lib`): `torch`, `transformers`, `matplotlib`,
  `numpy`. No `pip freeze` committed — but `pip install torch transformers
  matplotlib numpy` in a fresh venv reproduces.
- Model downloaded from HuggingFace on first run; cached in `~/.cache/`.
- No GPU required. CPU forward pass is fast enough.
- Deterministic: forward pass only, no sampling, no dropout. Re-runs produce
  byte-identical PDFs (modulo embedded timestamps in PDF metadata).

---

## 14. Open / potential next tasks

Things the user might ask for next; not done yet:

- **Larger K-sweep** like the paper appendix (K ∈ {-0.5, 0, 0.5, 1, 1.5, 2}).
  Currently all pipelines use {0.0, 0.3, 0.5, 1.0}. Edit `K_VALUES` in
  whichever pipeline. We discussed this and the user is *aware* — they
  haven't asked to change it.
- **Caption-conditioned RFEM-Class for CLIP** (either branch). Discussed
  twice; user said no twice. If they ever change their mind, the maths
  is: compute `∂(image_embed · text_embed) / ∂A_h^(l)` per layer,
  modulate attention per paper Eq. 6, then run the rest of RFEM.
- **Fine-tuning CLIP on Flickr 8k**. User asked, we said no — Flickr 8k
  is too small, RFEM is a visualisation method, fine-tuning would only
  *change* the attention pattern, not improve the explanation method
  itself. They were convinced. See §17.
- **Perturbation-based faithfulness metrics** (insertion / deletion AUCs).
  Not present in any pipeline.
- **Plausibility evaluation against GFDMs**. Would need a dataset with
  gaze data (Flickr 8k has none — would need SALICON, MexCulture,
  MIT1003, etc.).
- **The notebook** (`RFEM_BERT_presentation_final_v2 (1).ipynb`) is
  superseded by `rfem_pipeline.py`. User hasn't asked to delete it.

If the user asks "redo this on Roberta / DistilBERT / a different
classification task" — only `MODEL_NAME` and the `sentences` list need to
change, provided the model exposes `output_attentions=True` with the same
shape conventions.

---

## 15. TL;DR for a future Claude

You have inherited **four Python scripts**:

1. `rfem_pipeline.py` — **BERT/SST-2** — primary deliverable, works.
2. `rfem_clip_pipeline.py` — **CLIP text** (causal). Empty bars — BOS
   attention-sink phenomenon (see §16). Reportable, not a bug.
3. `rfem_clip_eos_pipeline.py` — **CLIP text** alt calibration. Also
   empty bars. Kept as parallel evidence of the same phenomenon.
4. `rfem_clip_vision_pipeline.py` — **CLIP vision** (ViT-B/32). Paper's
   home substrate (bidirectional). Works correctly. Structured stage
   subfolders per image. See §17.

Empty CLIP-text bars are **deliberate findings**, not bugs. Do not
"fix" them. See §10 (guardrails) and §16.

Outputs:

- BERT: `figs_pdf/<sid>/` — flat, 34 PDFs/sentence, 10 sentences.
- CLIP text (both variants): `figs_pdf_clip[_eos]/<sid>/` — flat, 34/sentence.
- CLIP vision: `figs_pdf_clip_vision/<img_id>/` — **stage-subfoldered**
  (`00_original/` → `06_rfem_step4_K_sweep/`), 44 PDFs/image, 10 images.

Two user modifications baked into all four pipelines:
1. K-σ filter keeps **values** instead of binarising to 1/0.
2. Aggregation is **weighted** (paper Eq. 5: `Σ w_h · Ā_h`, no normalisation).

CLIP vision has two extra display-only choices:
- **PowerNorm γ=0.3 applied to RFEM figures only** (not vanilla, not
  rollout) so small values stay visible against the CLS-CLS spike.
- **`magma` for matrices, `jet` for image overlays**.

Run any of them with `python <name>.py`. For the vision script, drop
images into `input_images/` first (recursively globbed).

Do not undo §10. When in doubt, **read the code, then this file, then
the paper §3.1 / §3.2 (pages 4–5)** in that order — and ask the user
before changing anything in §10.

---

## 16. CLIP text encoder — the BOS attention-sink finding

Both CLIP pipelines (`rfem_clip_pipeline.py`, `rfem_clip_eos_pipeline.py`)
produce **empty importance bars** at every K — and this is a **reportable
finding**, not a bug. The user has explicitly chosen to keep both files
as evidence.

### What the user is reporting

RFEM, as formulated in Ayyar et al. 2025 for bidirectional ViT attention,
**does not transfer to causal text transformers like CLIP** because of
the BOS (`<|startoftext|>`) attention sink.

### The math of the failure

Under the residual + per-layer multiplication of `(A + I)` in Eq. 3 of the
paper, repeated multiplication of causal lower-triangular row-stochastic
matrices causes attention mass to accumulate exponentially on column 0
(BOS). Empirically, after 12 layers, the **EOS row of Â_h is ≈ 99.4 %
concentrated on the BOS column**:

```
pos  0 BOS         : 0.99357     <-- attention sink
pos  1 the         : 0.00022
pos  2 film        : 0.00100
pos  3 is          : 0.00094
pos 13 EOS         : 0.00060
```

Three consequences:

1. **`w_h = max(Â_h) = 1` for every head**, achieved at `Â_h[0, 0] = 1`.
   The per-head weighting in Eq. 5 degenerates to uniform: all w_h equal.
2. **The K-σ threshold (computed on the full lower triangle) is too high
   for content tokens.** μ_full ≈ 2/(T+1) ≈ 0.14; content tokens cluster
   at ~0.0005. Threshold filters everything except BOS.
3. **Even calibrating μ, σ to the EOS row only doesn't help**, because
   the EOS row itself is BOS-dominated: μ_eos = 1/T ≈ 0.07 (driven by the
   BOS spike), threshold still kills all content tokens, BOS is then
   dropped from the bar plot for display, leaving 0 surviving entries.

### Two CLIP pipelines, two failure modes (both saved as evidence)

| File | μ_h, σ_h source | Why bars are empty |
|---|---|---|
| `rfem_clip_pipeline.py` | Full lower triangle of Â_h | μ ≈ 2/(T+1) ≫ content scale of 1/T. Threshold filters content. |
| `rfem_clip_eos_pipeline.py` | EOS row of Â_h | μ ≈ 1/T, but EOS row is 99.4 % BOS. Threshold still filters content. |

### What WOULD make CLIP bars non-empty (NOT IMPLEMENTED — user said no)

For completeness, in case the user changes their mind:

1. Compute μ_h, σ_h from EOS row **excluding the BOS column** (and
   possibly the EOS-self column). The threshold then calibrates to the
   informative content-token scale (~0.0005).
2. Compute `w_h = max(Â_h[eos_pos, 1:eos_pos])` so head weighting reflects
   each head's content-token peak rather than its BOS spike.

(1) alone fixes the empty-bar visualisation. (2) makes weighting non-trivial.
The user explicitly rejected applying these on 2026-05-15 ("see this is the
phenonmenon so i have to report.. so leave it like that!") — keep them
out of the codebase unless they re-open the question.

### Implications for the user's writeup

- The user is likely reporting this as a **negative result / limitation
  of standard RFEM** when ported to autoregressive text models.
- The natural framing: "RFEM works for bidirectional attention (BERT,
  ViT) but the BOS attention-sink in causal text transformers (CLIP,
  GPT-style) collapses the explanation onto position 0, requiring
  model-specific modifications."
- Both CLIP pipelines are diagnostic: pipeline #1 shows the failure
  under paper-faithful statistics; pipeline #2 shows that naively
  switching to EOS-row statistics doesn't help — pinpointing the BOS
  sink as the root cause, not the statistical region used for thresholding.

---

## 17. CLIP vision encoder pipeline — design notes

`rfem_clip_vision_pipeline.py` runs RFEM on CLIP's vision branch
(ViT-B/32 from `openai/clip-vit-base-patch32`). This is the paper's
home substrate (bidirectional attention, [CLS] at position 0), so
the maths is the paper's exactly — Eqs. 1–5 verbatim (with the same
two user modifications baked in: value-preserving filter, no `/Σw_h`).

### Architecture & choice of variant

| Property | Value | Why this choice |
|---|---|---|
| Model | `openai/clip-vit-base-patch32` | Vision branch only — `CLIPVisionModel`. |
| Layers × heads | 12 × 12 | Identical to BERT, identical to paper's ViT-B/16. |
| Patch size | **32** | Chosen over B/16 so the values-overlay (per-patch numbers on the 224×224 image) stays readable. Each patch is 32×32 px → comfortable for `0.014`-style annotations. B/16 would shrink each patch to 16×16 px and the values become illegible. |
| Tokens per forward | 50 (1 CLS + 7×7 patches) | Smaller than B/16's 197 — also makes the 50×50 matrices inspectable. |
| Attention | Bidirectional (no causal mask) | Math transfers directly from the paper; no BOS-sink-type pathology like in CLIP text. |

### Input / output layout

Inputs: drop images into `input_images/` (recursively globbed —
`.jpg/.jpeg/.png/.bmp/.webp`). The current set is 10 images that
look like Flickr 8k samples (`<numeric>_<hash>.jpg`).

Outputs: `figs_pdf_clip_vision/<image_basename>/`, **stage-
subfoldered**:

```
<img_id>/
├── 00_original/
│   └── original.pdf
├── 01_vanilla/                                     ← NO PowerNorm
│   ├── L1H1_matrix.pdf
│   ├── L1H1_overlay.pdf
│   ├── L1H1_overlay_values.pdf
│   ├── last_layer_matrix.pdf
│   ├── last_layer_overlay.pdf
│   └── last_layer_overlay_values.pdf
├── 02_rollout/                                     ← NO PowerNorm
│   ├── matrix.pdf
│   ├── overlay.pdf
│   ├── overlay_values.pdf
│   ├── per_layer_rollout_matrices_grid.pdf        ← 3×4 grid (heads-avg)
│   └── per_layer_rollout_overlays_grid.pdf        ← 3×4 grid on image
├── 03_rfem_step1_per_head/                        ← PowerNorm γ=0.3
│   ├── Araw_h1_L1.pdf
│   ├── AplusI_h1_L1.pdf
│   ├── Anorm_h1_L1.pdf
│   ├── final_rollout_h1_matrix.pdf
│   ├── final_rollout_h1_overlay.pdf
│   ├── per_layer_rollout_grid_h1.pdf              ← 3×4 grid (Head 1)
│   └── all12heads_grid.pdf                        ← THE killer figure
├── 04_rfem_step2_filter/                          ← PowerNorm γ=0.3
│   ├── K0.0/ {mask_h1_matrix, mask_h1_overlay, sparsity}
│   ├── K0.3/ …
│   ├── K0.5/ …
│   └── K1.0/ …
├── 05_rfem_step3_aggregate/                       ← PowerNorm γ=0.3
│   ├── K0.0/ {matrix, overlay, overlay_values}
│   ├── K0.3/ …
│   ├── K0.5/ …
│   └── K1.0/ …
└── 06_rfem_step4_K_sweep/                         ← PowerNorm γ=0.3
    └── K_sweep_overlays.pdf
```

**44 PDFs per image × 10 images = 440 PDFs total.**

The stage-subfolder layout was a user request after seeing the
flat 42-PDF version was hard to navigate. Naming order (00→06)
matches the reading order.

### Spatial overlay machinery

For a `(50,)` CLS attention row:
1. Drop position 0 (CLS-self) → 49 patch scores.
2. Reshape to `(7, 7)`.
3. Bilinearly upsample to `(224, 224)`.
4. Overlay on the resized image with jet colormap, α=0.55.

The values-overlay variant additionally prints each patch's raw
score in the patch's centre. Text colour switches white↔black based
on whether the displayed brightness at that location is above 0.55.

**The CLS-self entry is structurally not drawable** — CLS has no
spatial position on the image. The matrix view (`*_matrix.pdf`)
preserves it; the overlay (`*_overlay*.pdf`) drops it. This is
**why a matrix can look very different from its overlay** — see the
phenomenon documented below.

### Display-only choices (NOT part of the maths)

- **Colormap (matrices)**: `magma` (deep purple → magenta → cream).
  Originally was `Blues` — user asked for the magenta theme, then
  showed a magma screenshot to clarify.
- **Colormap (image overlays)**: `jet` with α=0.55, matching the
  paper.
- **PowerNorm γ = 0.3**, applied to RFEM figures **only** (folders
  03–06). NOT applied to vanilla (01) or rollout (02). Reason: the
  CLS-CLS entry in the rolled matrix is so dominant that it crushes
  all other values into the bottom of any linear colormap —
  PowerNorm γ=0.3 expands the low end so 0.014 looks meaningfully
  brighter than 0. User explicitly wanted baselines (vanilla,
  rollout) rendered linearly so they look "raw" and only RFEM gets
  the contrast boost.
- `use_power_norm=False` is passed for every vanilla/rollout call
  in `process_image`. All plot helpers have a `use_power_norm`
  parameter defaulting to `True`.

### Histograms removed

`plot_value_histogram` was removed. User said the value-distribution
histogram (μ/threshold lines on the rolled-value distribution) is
not informative for vision — the per-head sparsity bar plus the
spatial overlays already cover what the filter is doing. Do not
add it back.

### The CLS-CLS dominance / overlay-empty phenomenon

At K = 0.0 the user observed: matrix has a bright `(0, 0)` entry,
overlay has two highlighted patches with values 0.014 and 0.009.
At K ≥ 0.3 the overlay goes completely black even though the matrix
still has a bright `(0, 0)` entry.

This is **expected and correct**. The K-σ filter uses μ and σ from
the FULL matrix. The CLS-CLS entry (`Â_h[0, 0]` ≈ 1.0) is far above
all the patch-level entries (~0.02). Its presence inflates σ_h,
which pushes the threshold up. At K = 0.3, threshold reaches a
value above which no patch-attention entries survive — only
`(0, 0)` does. The matrix then shows the bright `(0, 0)`, but
the overlay drops `(0, 0)` (CLS-self has no spatial position), so
the overlay is empty.

This is the *same family* of pathology as the CLIP-text BOS sink,
just milder: there it's the BOS column that hoards mass, here it's
the CLS-CLS diagonal entry. The user is aware and has NOT asked us
to fix it (e.g. by computing μ/σ on the lower triangle excluding
CLS row/col). Leave as-is.

### Per-layer rollout grids (added on user request)

Two figures show how the **cumulative** rollout accumulates across
layers. Each is a 3×4 grid (layers 1–12):

1. `03_rfem_step1_per_head/per_layer_rollout_grid_h1.pdf` —
   per-head rollout (Head 1) cumulative matrix at each layer.
   Title format: "Layer N\nCLS↓ (r1,c1), (r2,c2)" showing the top-2
   patch coordinates in that layer's CLS row.
2. `02_rollout/per_layer_rollout_matrices_grid.pdf` — heads-averaged
   standard rollout cumulative matrix per layer.
3. `02_rollout/per_layer_rollout_overlays_grid.pdf` — same matrices'
   CLS rows overlaid on the image (one cell per layer).

The per-head grid lives in 03 and uses PowerNorm; the standard-
rollout grids live in 02 and do NOT use PowerNorm.

### What we explicitly chose NOT to do

- **Fine-tune CLIP on Flickr 8k.** User asked. Reasons we said no:
  Flickr 8k is too small (~0.002 % of CLIP's pretraining); RFEM is
  a visualisation method, not a model-improvement method;
  pretrained CLIP attention is what we want to study, not a
  Flickr 8k specialist version.
- **Build caption-conditioned RFEM-Class for CLIP vision.** Natural
  formulation would be `∂(image · text) / ∂A_h^(l)` per layer,
  modulate attention per Eq. 6, run the rest of RFEM. Discussed
  twice; user said no twice.
- **Switch to ViT-B/16** for sharper heatmaps. Considered and
  rejected — would break the values-overlay readability.
- **Add a value-distribution histogram per K**. Removed; not
  informative for vision.

### Recommended reading order per image

Same as in chat answer; reproduced here for posterity:

| # | File | What it shows |
|---|---|---|
| 0 | `00_original/original.pdf` | Reference image. |
| 1 | `01_vanilla/L1H1_matrix.pdf` | Single-head/single-layer raw matrix. |
| 2 | `01_vanilla/L1H1_overlay_values.pdf` | Same as spatial overlay with values. |
| 3 | `01_vanilla/last_layer_matrix.pdf` | Last-layer mean attention. |
| 4 | `01_vanilla/last_layer_overlay_values.pdf` | Spatial. |
| 5 | `02_rollout/matrix.pdf` | Standard rollout matrix. |
| 6 | `02_rollout/overlay_values.pdf` | Standard rollout spatial. |
| 7 | `02_rollout/per_layer_rollout_overlays_grid.pdf` | How rollout spreads across layers. |
| 8 | `03_rfem_step1_per_head/per_layer_rollout_grid_h1.pdf` | Same for Head 1 (per-head rollout). |
| 9 | `03_rfem_step1_per_head/all12heads_grid.pdf` | ★ Head specialisations — motivates filtering. |
| 10 | `04_rfem_step2_filter/K0.5/mask_h1_overlay.pdf` | What Head 1 survives at K=0.5. |
| 11 | `04_rfem_step2_filter/K0.5/sparsity.pdf` | Per-head survival rates. |
| 12 | `05_rfem_step3_aggregate/K0.5/overlay_values.pdf` | ★★ Final RFEM result. |
| 13 | `06_rfem_step4_K_sweep/K_sweep_overlays.pdf` | ★ Effect of K. |

If only five: 0 → 4 → 6 → 9 → 12 → 13.

### Math fidelity check (summary)

| Step | Paper (Eq.) | CLIP vision code | Status |
|---|---|---|---|
| 1 | `Â_h = Π(A_h^(l) + I)` (Eqs. 2–3) | `rfem_per_head_rollout` with row-norm | ✓ same standard convention as BERT pipeline |
| 2 | Binary `1[Â_h ≥ μ+Kσ]` (Eq. 4) | Value-preserving `torch.where(R_h ≥ thr, R_h, 0)` | ✓ user modification, identical to BERT pipeline |
| 3 | `Σ w_h · Ā_h, w_h = max(Â_h)` (Eq. 5) | Same, no `/Σw_h` | ✓ matches paper Eq. 5 exactly |
| 4 | CLS-row extraction → upsample → overlay | Same | ✓ |

`w_h` is non-degenerate here (heads have different maxes — e.g.
0.45, 0.52, 0.66, 0.70 …) because bidirectional attention doesn't
have the CLIP-text BOS-row collapse. The weighting is doing real work.

---

## 18. Sink-Aware RFEM — new pipelines (May 2026)

After establishing the "empty bars are the phenomenon" narrative (§16, §17),
the user decided to implement a **fix** as the positive contribution of the
work. The fix was proposed by Prof. Jenny Benois-Pineau:

> "μ_h and σ_h are computed from the read-out row only, **excluding the
> sink token columns**. The threshold is calibrated to the content-token
> distribution, not the sink-dominated whole-matrix distribution."

### Three new scripts

| Script | Output dir | Sink token(s) excluded |
|---|---|---|
| `rfem_bert_sink_aware.py` | `figs_bert_sink_aware/<sid>/` | `[CLS]` and `[SEP]` columns from CLS row |
| `rfem_clip_text_sink_aware.py` | `figs_clip_text_sink_aware/<sid>/` | BOS and EOS columns from EOS row |
| `rfem_clip_vision_sink_aware.py` | `figs_clip_vision_sink_aware/<img_id>/` | CLS-self column (position 0) from CLS row |

All three inherit ALL previous user modifications:
- Value-preserving K-σ filter (not binary).
- Weighted aggregation, no `/Σw_h`.

### The fix in code (same pattern across all three)

```python
def rfem_k_sigma_filter_sink_aware(head_rollouts, read_out_pos, k, sink_indices):
    for h in range(H):
        row = head_rollouts[h, read_out_pos, :]          # read-out row only
        content = [v for i,v in enumerate(row) if i not in sink_indices]
        mu_h  = mean(content)
        sig_h = std(content)
        threshold_h = mu_h + k * sig_h
        head_masks[h] = where(head_rollouts[h] >= threshold_h,
                              head_rollouts[h], 0)        # value-preserving
```

Key distinction from the standard pipeline:
- **Standard**: μ_h and σ_h from the full lower-triangle (or full matrix).
  Sink entries inflate σ_h → threshold exceeds all content tokens.
- **Sink-aware**: μ_h and σ_h from the **read-out row excluding sink columns**.
  Threshold calibrates to content-token scale → content tokens survive.

### BERT sink-aware results

With `[CLS]`/`[SEP]` excluded from CLS row statistics:
- Content words (e.g. *beautiful*, *resilience*, *portrait*) receive scores
  in the 0.02–0.14 range at K=0.5.
- Scores stabilise (token ranking converges) around K=0.5.
- `[CLS]`/`[SEP]` are suppressed from the final bar because sink columns are
  zeroed out by the threshold being calibrated to content values.

### CLIP-Text sink-aware results

With BOS/EOS excluded from EOS row statistics:
- Content token scores are in the 0.001–0.004 range (causal attention
  dilutes EOS row values — inherently 10-30× smaller than BERT CLS row).
- This is not a bug; it's a structural property of causal LM attention.
- Content tokens (film, portrait, resilience) still rank correctly.
- The K-sweep plot shows raw unnormalised scores with 4-decimal labels
  so the values are legible despite being small.

### CLIP-Vision sink-aware results

With CLS-self position (col 0) excluded from CLS row statistics:
- The CLS-CLS spike no longer inflates σ_h.
- Content patches survive at K=0.5 and produce a focused spatial heatmap.
- Overlay uses a **darkened-background variable-alpha jet** style (see §19).

---

## 19. CLIP-Vision overlay style — darkened variable-alpha (May 2026)

The original vision overlay used `jet` with fixed α=0.55 on the original
image. The user asked for a "more focused" style that makes salient patches
stand out more aggressively.

### New overlay recipe (applied in both vision scripts)

```python
darkened = (image_disp.astype(float32) * 0.6).clip(0, 255).astype(uint8)
ax.imshow(darkened)                             # step 1: darken background
norm_spatial = (spatial - sp_min) / (sp_max - sp_min)
alpha_map    = (norm_spatial ** 0.5) * 0.65    # step 2: variable alpha
cmap_fn      = plt.get_cmap("jet")
rgba         = cmap_fn(norm_spatial)
rgba[..., 3] = alpha_map
ax.imshow(rgba)                                 # step 3: jet layer on top
```

- Background is darkened to 60% brightness so low-alpha overlay areas recede.
- Alpha is proportional to `sqrt(norm_spatial)` so low-salience patches are
  transparent, high-salience patches are fully opaque jet.
- Applied to: K-sweep overlays, per-head overlays, aggregated overlays.
- Applied to **both** `rfem_clip_vision_pipeline.py` AND
  `rfem_clip_vision_sink_aware.py`.

### Guardrails for this style

- Do NOT revert to flat α=0.55 on the original image. The darkened style
  is the current canonical look for vision overlays.
- K=0.0 and K=0.3 were called "diffuse" by the user and excluded from the
  report at one point. Later reinstated for the full 4-K-value row. The
  four-value row is the current spec.

---

## 20. Image renaming (May 2026)

The input images for CLIP vision had hash-based filenames like
`10815824_2997e03d76.jpg`. Renamed to `I1.jpg` through `I10.jpg`:

```
10815824_2997e03d76.jpg → I1.jpg
(9 others similarly renamed)
```

Both vision scripts were re-run after renaming. Output directories are now
`figs_pdf_clip_vision/I1/`, `figs_clip_vision_sink_aware/I1/`, etc.

The report macros were updated to match:
```latex
\newcommand{\CLIPVstd}{figs_pdf_clip_vision/I1}
\newcommand{\CLIPVsa}{figs_clip_vision_sink_aware/I1}
```

---

## 21. Visual comparison report — report_rfem_comparison.tex (May 2026)

A LaTeX report was created to present Standard RFEM vs Sink-Aware RFEM
side by side for all three encoders (BERT, CLIP-Text, CLIP-Vision).

### File

`report_rfem_comparison.tex` → compiles to `report_rfem_comparison.pdf`
(currently 10 pages).

### Page 1 — Methodology (FIXED, do not change)

Explains RFEM stages A–D with the exact formulas. Also explains the
sink problem and the sink-aware fix. This page must remain unchanged
between report versions.

### Report structure per encoder

**BERT S1** and **CLIP-Text S1** follow identical structure:

1. Last-layer mean attention matrix + vanilla CLS/EOS score bar (side by side)
2. Standard attention rollout CLS/EOS bar (full width)
3. `\subsection*{RFEM — Stage A}` — Head 1 + Head 12 rolled-out matrices (side by side)
4. `\subsection*{RFEM — Stage C}` — Weighted aggregation matrix: Std vs SA-RFEM (side by side)
5. `\subsection*{Stage D — Standard RFEM}` — 4 K-value individual bar charts in one row
6. `\subsection*{Stage D — Sink-Aware RFEM}` — Single K-sweep importance figure (full width, all 4 K panels)

**CLIP-Vision I1**:

1. Original + vanilla overlay + rollout overlay (3 in one row)
2. Head 1 + Head 12 overlays (side by side, standard only — same for RFEM/SA-RFEM at this stage)
3. Weighted aggregation **matrix** (Std vs SA-RFEM side by side) ← matrix heatmap
4. Weighted aggregation **overlay** (Std vs SA-RFEM side by side) ← jet overlay on image
5. Standard RFEM K-sweep overlays (all 4 K values in one row)
6. SA-RFEM K-sweep overlays (all 4 K values in one row)

### Figure file sources

| Report macro | Path |
|---|---|
| `\BERT` | `figs_pdf/S1` |
| `\BERTsa` | `figs_bert_sink_aware/S1` |
| `\CLIPTxt` | `figs_clip_text_sink_aware/S1` |
| `\CLIPVstd` | `figs_pdf_clip_vision/I1` |
| `\CLIPVsa` | `figs_clip_vision_sink_aware/I1` |

CLIP-Text **standard** RFEM figures (stage C and D standard) use hardcoded
path `figs_pdf_clip/S1/` (no macro) because `\CLIPTxt` points to the
sink-aware directory.

### K-sweep importance figures (Stage D SA-RFEM)

For BERT and CLIP-Text, the Stage D SA-RFEM figure is the **single K-sweep
importance file** (`S1_M3_step4_K_sweep_importance.pdf`) showing all 4 K
panels in one figure with horizontal bars (tokens on y-axis, raw unnormalised
importance on x-axis). This is generated by `plot_k_sweep_importance` in each
sink-aware script.

At one point the CLIP-Text K-sweep was normalised (bars filled plot space but
raw values on labels). This was reverted — both scripts now output raw values.

### Guardrails for the report

- Page 1 (methodology) is frozen. Do not restructure it.
- All standalone (non-subfigure) figures use `\linewidth`.
- 4-subfigure rows use `0.235\linewidth` per subfigure.
- 2-subfigure rows use `0.48\linewidth` per subfigure.
- 3-subfigure rows use `0.30\linewidth` per subfigure.
- Captions must be descriptive: explain WHAT is shown AND what the
  sink/sink-aware difference means. Do not use one-word captions.

---

## 22. Open questions / things not yet resolved

- **`plot_matrix_heatmap` stats from whole matrix vs CLS row**: The μ and σ
  shown in the title of the vanilla attention matrix (`S1_M1_vanilla_matrix.pdf`)
  are computed from the **full T×T matrix** (`matrix.mean()`, `matrix.std()`).
  The user asked about this in May 2026. It has NOT yet been changed.
  If they want CLS-row-specific stats, change line ~127:
  ```python
  # current:
  mu  = float(matrix.mean())
  sig = float(matrix.std())
  # to show CLS-row stats instead:
  mu  = float(matrix[0].mean())
  sig = float(matrix[0].std())
  ```
  This would only affect the title annotation — not the filter, not the figure content.

- **Sentences S2–S10** for sink-aware pipelines: Only S1 figures are used in
  the report. All 10 sentences are processed by the scripts, but only S1 is
  shown. The report could be extended to compare across sentences.

- **SA-RFEM for CLIP-Vision using K-sweep single figure**: Currently the
  CLIP-Vision SA-RFEM K-sweep uses 4 individual subfigures in the report.
  BERT and CLIP-Text SA-RFEM use the single `K_sweep_importance.pdf` file.
  Vision doesn't have an equivalent (it uses spatial overlays, not token bars).

