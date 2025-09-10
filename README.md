# TabTransformer++ — Residualized, Calibrated Transformer for Tabular Data (PS‑S5E9)

**TabTransformer++** extends TabTransformer for tabular ML with four practical innovations that make it work in weak‑signal settings:  
1) **Residual learning** against a strong ensemble baseline,  
2) **Leak‑free per‑fold quantile tokenization** (incl. special bins for model predictions),  
3) **Gated value towers** that fuse standardized raw values into token embeddings, and  
4) **Fold‑wise isotonic calibration** in z‑space.  

On Kaggle **Playground Series S5E9**, this yields a reproducible **~0.005 OOF RMSE** gain over robust GBDT ensembles.

---

## TL;DR
- **Task:** Beats‑per‑Minute regression (PS‑S5E9).  
- **Why it matters:** Dataset has *tiny* signal; most models collapse to “predict ~120”. TT++ reliably extracts residual structure.  
- **Result:** Consistent OOF improvements; CV↔LB alignment; all artifacts (OOF/TEST) include MD5 checksums.  
- **Run:** `pip install -r requirements.txt && python scripts/06_tabtransformer_pp.py`

---

## Results (OOF RMSE)

| Stage | OOF RMSE | Δ vs prior |
|---|---:|---:|
| XGB (5‑seed, sub‑bag, nested affine) | 26.4559 | — |
| GPU Trio (LGB/XGB/CAT, per‑fold bins, per‑model iso → blend → iso) | 26.4533 | −0.0026 |
| Quad Ridge stack (base + DT + quad terms) | 26.4531 | −0.0002 |
| Residual ElasticNet (kicker; negative result) | 26.4536 | +0.0005 |
| Micro‑stack (ridge on [quad, enet]) | **26.4531** | ~0 |
| **TabTransformer++ (residualized, gated, calibrated)** | **26.4480** | **−0.0050** |

> Public LB tracked CV within expected variance. All stages write OOF/TEST CSVs with MD5s.

---

## Method Highlights

- **Residualization:** Train TT++ on `residual = y − y_base`, where `y_base` is a strong ensemble (Quad Ridge). Stabilizes learning in low‑signal regimes.  
- **Leak‑free tokenization:** Quantile bin edges computed on *train folds only*; separate high‑resolution bins for `base_pred` and `dt_pred`.  
- **Gated value towers:** For each token, an MLP(1→D→D) embeds the standardized raw value and is fused with the token embedding via a learnable sigmoid gate.  
- **Calibration:** Per‑fold **isotonic regression** in z‑space; applied to OOF and test predictions.

---

## Repo Layout

```
scripts/
  00_fetch_data.py
  01_xgb_oof.py
  02_gpu_trio_oof.py
  03_quad_ridge.py
  04_residual_enet.py
  05_micro_stack.py
  06_tabtransformer_pp.py
  07_calibration_utils.py
  utils.py
src/ttpp/
  model.py          # transformer + gated value towers
  tokenization.py   # per‑fold binning/digitization
  dataset.py        # PyTorch Dataset/DataLoader
  train.py          # loops, EMA, cosine schedule, grad‑clip
  calibrate.py      # per‑fold isotonic in z‑space
  metrics.py
data/
  raw/  interim/  oof/  sub/
results/
  tables/ plots/ logs/
reproducibility/
  run_end_to_end.sh
  checksums.md
  env_report.txt
```

---

## Quick Start

```bash
git clone https://github.com/LEDazzio01/Tab_Transformer_Plus_Plus.git
cd Tab_Transformer_Plus_Plus

python -m pip install -r requirements.txt

# Place PS‑S5E9 train/test CSVs under data/raw/
# or run (requires Kaggle API credentials configured locally):
# python scripts/00_fetch_data.py

# Full pipeline (baseline → ensembles → TT++)
python scripts/01_xgb_oof.py
python scripts/02_gpu_trio_oof.py
python scripts/03_quad_ridge.py
python scripts/05_micro_stack.py
python scripts/06_tabtransformer_pp.py
```

Artifacts are written to `data/oof/` and `data/sub/` with **MD5** appended. A consolidated list is in `reproducibility/checksums.md`. Environment snapshot is in `reproducibility/env_report.txt`.

---

## Plots to Expect (in `results/plots/`)
- CV vs LB alignment across stages  
- Calibration curves (pre/post iso per fold)  
- OOF residual histograms (base vs TT++)  
- Learning curves (train size vs OOF)  
- Correlation matrix of OOF predictions (XGB/LGB/CAT/DT/TT++)  
- Ablations (−value tower / −gating / −per‑fold bins / −iso)

---

## Requirements

See `requirements.txt` (pinned versions). Typical stack:

```
numpy, pandas, scikit‑learn, scipy
xgboost, lightgbm, catboost
torch (CUDA if available)
matplotlib, seaborn, tqdm
```

---

## License & Citation

- **License:** MIT (or Apache‑2.0).  
- **Cite this work:** add a `CITATION.cff` like:

```yaml
cff-version: 1.2.0
title: "TabTransformer++: Residualized, Calibrated Transformer for Tabular Data (PS‑S5E9)"
message: "If you use this software, please cite it as below."
authors:
  - family-names: Dazzio
    given-names: L. Elaine
repository-code: "https://github.com/LEDazzio01/Tab_Transformer_Plus_Plus"
version: "0.1.0"
date-released: "2025-09-09"
```
