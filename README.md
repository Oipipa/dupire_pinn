# Dupire-PINN

Physics-informed neural net for calibrating a Dupire local volatility surface from option prices. Trains a two-headed network for call prices $C(x,T)$ and log-variance $\nu(x,T)$ with PDE, boundary, no-arbitrage, and regularization losses. Includes scripts to prepare data, train, evaluate, visualize, score vs market, price a CSV, and run an end-to-end pipeline.



## Repo layout

```
dupire_pinn/
  autodiff.py
  coords.py
  config.py
  data/
    boundaries.py
    collocation.py
    market.py
  eval/
    grids.py
    metrics.py
  losses/
    pde.py
    penalties.py
    reg.py
    mass.py
  networks/
    heads.py
    wrappers.py
  sampling.py
  train/
    loop.py
    objectives.py
    optimizer.py
  utils/
    io.py
    seed.py
    checkpoint.py
    estimation.py
  cli/
    prepare_csv.py
    train.py
    eval.py
    visualize.py
    score.py
    gate.py
    pipeline.py
    price_csv.py
    estimate_rq.py
configs/
  example.toml
data/
  spx_eod_202301.csv
processed/
env.sh
run_single.sh
```



## Environment

Python 3.10+ and PyTorch.

Create venv and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy pandas matplotlib tomli
```

Set local env helpers:

```bash
# env.sh should export PYTHONPATH and any defaults you want
source env.sh
```



## Data

Raw EOD options CSV expected columns (case-insensitive, cleaned internally):

```
QUOTE_DATE (date), DTE (days to expiry), STRIKE,
C_BID, C_ASK, P_BID, P_ASK, UNDERLYING_LAST
```

The CLI will standardize to a training CSV with columns:

```
K, T, C
```

where $T = \text{DTE}/365$, $C$ is call mid.



## Quickstart (manual)

Prepare standardized market CSV:

```bash
source env.sh
python -m dupire_pinn.cli.prepare_csv \
  --in data/spx_eod_202301.csv --date 2023-01-04 \
  --out spx_std.csv --tmin 0.01 --tmax 2.0 --min_bid 0.05 --kq_lo 0.02 --kq_hi 0.98
```

Train (single pass, creates `spx_fast.pt`):

```bash
export S0=3853.39
export R=0.04974181812769106
export Q=0.023177742794908366
export INT_N=16384

python -m dupire_pinn.cli.train --csv spx_std.csv --epochs 220 --lr 3e-4 \
  --ckpt spx_fast.pt \
  --width 128 --hidden 4 --batch_int 8192 --batch_mkt 4096 \
  --w_pde 1.0 --w_mkt 1.0 --w_arb 0.06 --w_bc 0.10 --w_reg1 1e-5 --w_reg2 0.0 \
  --sigma_ref 0.25 --lam_sigma 1e-3 --lam_pos 5e-3 --lam_tie 3e-2 \
  --w_mass 0.2 --mass_nx 512 --mass_nt 24
```

Evaluate, score, visualize, price:

```bash
python -m dupire_pinn.cli.eval  --ckpt spx_fast.pt --csv spx_std.csv
python -m dupire_pinn.cli.score --ckpt spx_fast.pt --csv spx_std.csv
python -m dupire_pinn.cli.visualize --ckpt spx_fast.pt --csv spx_std.csv --out surface.csv
python -m dupire_pinn.cli.price_csv --ckpt spx_fast.pt --in spx_std.csv --out prices.csv
```

Outputs:

* `surface.csv` grid of $x,T,K,C,\nu,\sigma$
* `prices.csv` model prices at market points



## End-to-end pipeline

Config-driven run that prepares data, trains, evaluates, saves surface and prices, and writes metrics.

Edit `configs/example.toml` to point to your raw CSV and desired hyperparameters. You can set `S0/r/q` to `"auto"` to estimate via put-call parity regression.

Run:

```bash
source env.sh
python -m dupire_pinn.cli.pipeline --config configs/example.toml
```

Outputs in `processed/<run_name>/`:

```
config.toml
market.csv
model.pt
surface.csv
prices.csv
metrics.json
```

`metrics.json` contains:

* `nv` (no-arbitrage violations on eval grid)
* `resid` (normalized Dupire residual)
* `mass` (forward density mass error)
* `rmse_abs`, `rmse_rel` (fit on market)
* `passed` boolean against gate thresholds in config



## Command reference

Prepare CSV:

```bash
python -m dupire_pinn.cli.prepare_csv \
  --in RAW.csv --date YYYY-MM-DD --out OUT.csv \
  --tmin 0.01 --tmax 2.0 --min_bid 0.05 --kq_lo 0.02 --kq_hi 0.98
```

Train:

```bash
python -m dupire_pinn.cli.train --csv OUT.csv --epochs 200 --lr 3e-4 \
  --ckpt model.pt [--resume] \
  --width 128 --hidden 4 --batch_int 8192 --batch_mkt 4096 \
  --w_pde 1.0 --w_mkt 1.0 --w_arb 0.06 --w_bc 0.10 --w_reg1 1e-5 --w_reg2 0.0 \
  --sigma_ref 0.25 --lam_sigma 1e-3 --lam_pos 5e-3 --lam_tie 2e-2 \
  --w_mass 0.2 --mass_nx 512 --mass_nt 24 \
  [--al --eta 0.5]
```

Eval (grid metrics):

```bash
python -m dupire_pinn.cli.eval --ckpt model.pt --csv OUT.csv
```

Market fit:

```bash
python -m dupire_pinn.cli.score --ckpt model.pt --csv OUT.csv
```

Visualize surface:

```bash
python -m dupire_pinn.cli.visualize --ckpt model.pt --csv OUT.csv --out surface.csv
```

Price a CSV of $(K,T)$:

```bash
python -m dupire_pinn.cli.price_csv --ckpt model.pt --in OUT.csv --out prices.csv
```

Gate (fail/exit if thresholds not met):

```bash
python -m dupire_pinn.cli.gate --ckpt model.pt --csv OUT.csv \
  --max_nv 0 --max_resid 1e-3 --max_mass 1e-3 --max_rmse_rel 1e-3
```

Estimate `S0,r,q` from raw:

```bash
python -m dupire_pinn.cli.estimate_rq --in RAW.csv --date YYYY-MM-DD \
  --tmin 0.01 --tmax 2.0 --min_bid 0.05 --kq_lo 0.02 --kq_hi 0.98
```

Pipeline:

```bash
python -m dupire_pinn.cli.pipeline --config configs/example.toml
```



## Configuration

`dupire_pinn/config.py` provides defaults and reads env overrides:

```
S0, r, q, INT_N, SEED, etc.
```

Most CLIs also accept flags to override network width/hidden, batches, and loss weights. The pipeline uses TOML (`configs/example.toml`).

Important knobs:

* `--w_pde`, `--lam_tie`: tighten Dupire PDE
* `--w_arb` or `--al --eta`: push no-arbitrage
* `--w_mass`, `--mass_nx`, `--mass_nt`: enforce density mass
* `--sigma_ref`, `--lam_sigma`: smooth local vol field



## Metrics and targets

Strict targets (theory-driven):

* `nv = 0`
* `resid ≤ 1e-3`
* `mass ≤ 1e-3`
* `rmse_rel ≤ 1e-3` (when bid/ask unavailable)

Practical targets (often achievable on noisy EOD):

* `nv ≤ 20`
* `resid ≤ 2e-2`
* `mass ≤ 3e-2`
* `rmse_rel ≤ 8e-3`



## Repro and tips

* Seed: set in `dupire_pinn/config.py` (`SEED`), used in CLIs and pipeline. For exact reproducibility keep the same `INT_N`, batches, and device.
* First run vs resume: pass `--resume` only if the checkpoint exists.
* Don’t mix different CSVs or hyperparameter recipes mid-run unless you intend to.
* If you see the warning about converting tensors with requires\_grad, it’s benign in logs; we detach metrics in pipeline and gate.
* GPU: training benefits from GPU. Set `torch.set_default_device("cuda")` if you extend code; current CLIs run on CPU by default.



## Troubleshooting

* `FileNotFoundError: spx_fast.pt`: train first (no `--resume`), then eval/score/price.
* Size mismatch on load: you changed `--width/--hidden`. Retrain or keep the same architecture as the checkpoint.
* `Unexpected key(s) inner.*`: checkpoints saved via `wrappers` expect loading with the same wrapper; keep the same CLI path.
* Too many violations: use `--al --eta 0.5` for a short polish pass and increase `--w_arb`, `--lam_tie`.
* High mass error: increase `--w_mass` and `--mass_nx`/`--mass_nt`.
* High PDE residual: increase `--w_pde` and `--lam_tie`, and raise `INT_N`.



## Example one-shot script

`run_single.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source env.sh

export S0=3853.39
export R=0.04974181812769106
export Q=0.023177742794908366
export INT_N=16384

CSV=spx_std.csv
CKPT=spx_fast.pt
DATE=2023-01-04

rm -f "$CKPT"

python -m dupire_pinn.cli.prepare_csv \
  --in data/spx_eod_202301.csv --date "$DATE" \
  --out "$CSV" --tmin 0.01 --tmax 2.0 --min_bid 0.05 --kq_lo 0.02 --kq_hi 0.98

python -m dupire_pinn.cli.train --csv "$CSV" --epochs 220 --lr 3e-4 \
  --ckpt "$CKPT" \
  --width 128 --hidden 4 --batch_int 8192 --batch_mkt 4096 \
  --w_pde 1.0 --w_mkt 1.0 --w_arb 0.06 --w_bc 0.10 --w_reg1 1e-5 --w_reg2 0.0 \
  --sigma_ref 0.25 --lam_sigma 1e-3 --lam_pos 5e-3 --lam_tie 3e-2 \
  --w_mass 0.2 --mass_nx 512 --mass_nt 24

python -m dupire_pinn.cli.eval  --ckpt "$CKPT" --csv "$CSV"
python -m dupire_pinn.cli.score --ckpt "$CKPT" --csv "$CSV"
python -m dupire_pinn.cli.visualize --ckpt "$CKPT" --csv "$CSV" --out surface.csv
python -m dupire_pinn.cli.price_csv --ckpt "$CKPT" --in "$CSV" --out prices.csv

echo "surface.csv"
echo "prices.csv"
```
