### What this project does

```
GOAL
────
Forecast next-day log-return of CHDVD.SW (iShares Swiss Dividend ETF),
turn forecasts into trading positions, and compare ML models vs baselines.

The pipeline uses CHDVD.SW + top holdings data, builds rolling/statistical
features and rule-based signal features, then evaluates out-of-sample IC,
Sharpe, and annualized return.
```

### Setup (what to install)

```bash
# 1) Go to repo root
cd /tmp/workspace/vstrozzi/ml_for_finance

# 2) Create environment (recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Alternative (conda):

```bash
cd /tmp/workspace/vstrozzi/ml_for_finance
conda env create -f environment.yml
conda activate mlfin
```

### How to run the repository

```bash
cd /tmp/workspace/vstrozzi/ml_for_finance

# (A) Download market data (CHDVD.SW + holdings)
python utils/extract_data.py

# (B) Clean/impute missing values (default: ffill)
python utils/clean_data.py --method ffill

# (C) Open the end-to-end notebook pipeline
jupyter notebook notebooks/pipeline.ipynb
```

Inside `notebooks/pipeline.ipynb`:
- Set `DOWNLOAD_DATA = True/False`
- Set `CLEAN_DATA = True/False`
- Set `CLEAN_METHOD = 'ffill' | 'bfill' | 'xgboost'`
- Set `TRAIN = True` to retrain, `False` to reuse saved weights from `data/weights/`
- Run all cells top-to-bottom

### File structure (important folders/files)

```
ml_for_finance/
├── README.txt
├── requirements.txt
├── environment.yml
├── model.py                    # Linear, XGBoost, MaskedVAE models
├── test.py                     # sanity tests
├── notebooks/
│   └── pipeline.ipynb          # main end-to-end workflow
├── utils/
│   ├── config.py               # paths, tickers, seeds, hyperparameters
│   ├── extract_data.py         # download raw data
│   ├── clean_data.py           # imputation/cleaning
│   ├── data.py                 # load + validate price data
│   ├── features.py             # feature engineering
│   ├── signals.py              # signal rules
│   ├── dataset.py              # chronological splits + scaling/PCA
│   ├── metrics.py              # IC/Sharpe and metrics helpers
│   ├── backtest.py             # position mapping + backtest logic
│   ├── results.py              # save/load run JSON files
│   └── compare.py              # compare runs from results/
├── data/
│   ├── raw/                    # downloaded raw OHLCV + close panel
│   ├── processed/              # cleaned panel output
│   └── weights/                # saved model weights
└── results/                    # per-run metrics JSON outputs
```

### How to reproduce results

1. Install dependencies (see Setup).
2. Run data extraction:
   - `python utils/extract_data.py`
3. Run cleaning:
   - `python utils/clean_data.py --method ffill`
4. Run notebook:
   - `jupyter notebook notebooks/pipeline.ipynb`
   - keep `set_seed(42)` and `cfg = Config()` as in notebook
   - keep chronological split defaults from `utils/config.py` (`train=0.70`, `val=0.15`, `test=0.15`)
5. Execute all cells and wait for training/evaluation to finish.
6. Check outputs:
   - model run files in `results/`
   - optional comparison report:
     - `python utils/compare.py --latest`

### Quick check

```bash
cd /tmp/workspace/vstrozzi/ml_for_finance
pytest -q test.py
```
