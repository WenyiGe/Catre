#!/usr/bin/env python3
# HEC-compatible runner for CaTRe ICLR experiments.
# Generated from catre_iclr_colab_experiments.ipynb, with Google Drive/notebook code removed.

import os, re, json, math, random, time, warnings, argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, log_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from IPython.display import display
except Exception:
    def display(x):
        if hasattr(x, "to_string"):
            print(x.to_string())
        else:
            print(x)


def parse_args():
    parser = argparse.ArgumentParser(description="CaTRe HEC experiment runner")
    parser.add_argument("--root", type=str, default=os.environ.get("CATRE_ROOT", "/mmfs1/storage/users/gew6/CaTRe_ICLR"))
    parser.add_argument("--run_mode", type=str, default="full", choices=["smoke", "full"])
    parser.add_argument("--experiment", type=str, default="main", choices=["synthetic", "main", "noise", "missingness", "ablation", "finite_shot", "inspect"])
    parser.add_argument("--dataset", type=str, default="room_occupancy")
    parser.add_argument("--backbone", type=str, default="gru", choices=["gru", "grud", "ode_rnn", "ltc"])
    parser.add_argument("--variant", type=str, default="base", choices=["base", "catre", "gate_only", "cal_loss_only", "stability_only", "full_catre"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--phi_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--window_length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--gaussian_std", type=float, default=0.0)
    parser.add_argument("--max_windows", type=int, default=-1, help="Limit windows per split; -1 means no explicit limit")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--validate_job_only", action="store_true", help="Print job mapping without loading data or training")
    return parser.parse_args()

ARGS = parse_args()
RUN_MODE = ARGS.run_mode
ROOT = Path(ARGS.root).expanduser().resolve()
DIRS = {
    'root': ROOT,
    'raw': ROOT / 'data_raw',
    'processed': ROOT / 'data_processed',
    'models': ROOT / 'models',
    'predictions': ROOT / 'predictions',
    'tables': ROOT / 'tables',
    'figures': ROOT / 'figures',
    'logs': ROOT / 'logs',
    'configs': ROOT / 'configs',
}
for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE, flush=True)
print('Project folder:', ROOT, flush=True)

GLOBAL_SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

set_seed(GLOBAL_SEED)

@dataclass
class ExperimentConfig:
    window_length: int = 32
    stride: int = 8
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    hidden_dim: int = 64
    phi_dim: int = 64
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 10
    seeds: Tuple[int, ...] = (1,2,3,4,5)
    ece_bins: int = 15
    max_features: int = 128
    min_class_count: int = 20
    max_windows_smoke: int = 500
    max_windows_full: Optional[int] = None

CFG = ExperimentConfig()
CFG.window_length = ARGS.window_length
CFG.stride = ARGS.stride
CFG.hidden_dim = ARGS.hidden_dim
CFG.phi_dim = ARGS.phi_dim
CFG.batch_size = ARGS.batch_size
CFG.lr = ARGS.lr
CFG.weight_decay = ARGS.weight_decay
CFG.max_epochs = ARGS.max_epochs
CFG.patience = ARGS.patience
CFG.seeds = (ARGS.seed,)

if RUN_MODE == 'smoke':
    CFG.max_epochs = min(CFG.max_epochs, 3)
    CFG.patience = min(CFG.patience, 2)
    CFG.hidden_dim = min(CFG.hidden_dim, 32)
    CFG.phi_dim = min(CFG.phi_dim, 32)
    CFG.batch_size = min(CFG.batch_size, 32)
    MAX_WINDOWS_PER_SPLIT = CFG.max_windows_smoke if ARGS.max_windows < 0 else ARGS.max_windows
else:
    MAX_WINDOWS_PER_SPLIT = None if ARGS.max_windows < 0 else ARGS.max_windows



DATASET_SPECS = {
    'opportunity': {'uci_id': 226, 'name': 'OPPORTUNITY Activity Recognition', 'label_candidates': ['activity','label','locomotion','gesture']},
    'ambient_har': {'uci_id': 506, 'name': 'Human Activity Recognition from Continuous Ambient Sensor Data', 'label_candidates': ['activity','label','class']},
    'wisdm': {'uci_id': 507, 'name': 'WISDM Smartphone and Smartwatch Activity and Biometrics', 'label_candidates': ['activity','activity_code','label','class']},
    'room_occupancy': {'uci_id': 864, 'name': 'Room Occupancy Estimation', 'label_candidates': ['room_occupancy_count','occupancy','label','class','target']},
    'occupancy_detection': {'uci_id': 357, 'name': 'Occupancy Detection', 'label_candidates': ['occupancy','label','class','target']},
}


# V2: preserve one configuration record per independent scheduled job.
CONFIG_KEY = (f"{ARGS.experiment}__{ARGS.dataset}__{ARGS.backbone}__"
              f"{ARGS.variant}__seed{ARGS.seed}__epochs{CFG.max_epochs}__patience{CFG.patience}")
with open(DIRS['configs'] / f'{CONFIG_KEY}.json', 'w') as f:
    json.dump({'args': vars(ARGS), 'run_mode': RUN_MODE, **asdict(CFG)}, f, indent=2)



# Cell 3. UCI loading and preprocessing utilities
from ucimlrepo import fetch_ucirepo

def safe_lower(x):
    return str(x).strip().lower().replace(' ', '_').replace('-', '_')

def fetch_uci_as_dataframe(dataset_key, force_refresh=False):
    spec = DATASET_SPECS[dataset_key]
    cache_path = DIRS['raw'] / f'{dataset_key}_ucimlrepo_raw.parquet'
    meta_path = DIRS['raw'] / f'{dataset_key}_metadata.json'
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path), json.loads(meta_path.read_text()) if meta_path.exists() else {}
    ds = fetch_ucirepo(id=spec['uci_id'])
    metadata = {}
    try:
        metadata['metadata'] = {str(k): str(v) for k, v in dict(ds.metadata).items()}
    except Exception as e:
        metadata['metadata'] = f'unavailable: {e}'
    try:
        if isinstance(ds.variables, pd.DataFrame):
            metadata['variables'] = ds.variables.astype(str).to_dict(orient='records')
    except Exception as e:
        metadata['variables'] = f'unavailable: {e}'
    df = None
    try:
        original = getattr(ds.data, 'original', None)
        if isinstance(original, pd.DataFrame) and len(original) > 0:
            df = original.copy()
    except Exception:
        pass
    if df is None:
        try:
            features = getattr(ds.data, 'features', None)
            targets = getattr(ds.data, 'targets', None)
            if isinstance(features, pd.DataFrame) and len(features) > 0:
                df = features.copy()
                if isinstance(targets, pd.DataFrame) and len(targets) > 0:
                    for c in targets.columns:
                        name = c if c not in df.columns else f'target_{c}'
                        df[name] = targets[c].values
        except Exception:
            pass
    if df is None or len(df) == 0:
        meta_path.write_text(json.dumps(metadata, indent=2))
        raise RuntimeError(f'No clean dataframe from ucimlrepo for {dataset_key}. Metadata saved to {meta_path}.')
    metadata['original_columns'] = list(map(str, df.columns))
    df.columns = [safe_lower(c) for c in df.columns]
    metadata['normalised_columns'] = list(df.columns)
    df.to_parquet(cache_path)
    meta_path.write_text(json.dumps(metadata, indent=2))
    return df, metadata

def read_wisdm_raw_folder(folder: Path):
    records = []
    for fp in list(folder.rglob('*.txt')) + list(folder.rglob('*.csv')):
        with open(fp, 'r', errors='ignore') as f:
            for line in f:
                line = line.strip().strip(';')
                if not line or ',' not in line: continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 6: continue
                try:
                    records.append({'user': parts[0], 'activity': parts[1], 'timestamp': float(parts[2]), 'x': float(parts[3]), 'y': float(parts[4]), 'z': float(parts[5]), 'source_file': fp.name})
                except Exception:
                    continue
    if not records: raise RuntimeError(f'No WISDM-style rows under {folder}')
    return pd.DataFrame(records)

def read_opportunity_raw_folder(folder: Path, label_col=-1):
    frames = []
    for fp in sorted(list(folder.rglob('*.dat')) + list(folder.rglob('*.txt'))):
        try:
            arr = np.loadtxt(fp)
            if arr.ndim != 2 or arr.shape[1] < 5: continue
            lab = arr[:, label_col]
            X = np.delete(arr, label_col % arr.shape[1], axis=1)
            df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
            df['activity'] = lab.astype(int)
            df['timestamp'] = np.arange(len(df), dtype=float)
            df['source_file'] = fp.name
            frames.append(df)
        except Exception as e:
            print('skip', fp, e)
    if not frames: raise RuntimeError(f'No OPPORTUNITY-style .dat rows under {folder}')
    return pd.concat(frames, ignore_index=True)

def read_ambient_har_raw_folder(folder: Path):
    frames = []
    for fp in list(folder.rglob('*.csv')) + list(folder.rglob('*.txt')) + list(folder.rglob('*.data')):
        for sep in [',', '	', None]:
            try:
                tmp = pd.read_csv(fp, sep=sep, engine='python', on_bad_lines='skip')
                if tmp.shape[1] >= 3 and len(tmp) > 0:
                    tmp.columns = [safe_lower(c) for c in tmp.columns]
                    tmp['source_file'] = fp.name
                    frames.append(tmp)
                    break
            except Exception:
                pass
    if not frames: raise RuntimeError(f'No ambient-HAR-style files under {folder}')
    return pd.concat(frames, ignore_index=True)

def load_dataset_with_fallback(dataset_key):
    try:
        return fetch_uci_as_dataframe(dataset_key)[0]
    except Exception as e:
        print(f'ucimlrepo dataframe loading failed for {dataset_key}: {e}')
        folder = DIRS['raw'] / dataset_key
        if not folder.exists():
            raise RuntimeError(f'Create folder {folder} and place extracted raw files there.')
        if dataset_key == 'wisdm': df = read_wisdm_raw_folder(folder)
        elif dataset_key == 'opportunity': df = read_opportunity_raw_folder(folder)
        elif dataset_key == 'ambient_har': df = read_ambient_har_raw_folder(folder)
        else:
            frames = []
            for fp in list(folder.rglob('*.csv')) + list(folder.rglob('*.txt')):
                try:
                    tmp = pd.read_csv(fp, engine='python', on_bad_lines='skip')
                    tmp.columns = [safe_lower(c) for c in tmp.columns]
                    frames.append(tmp)
                except Exception: pass
            if not frames: raise
            df = pd.concat(frames, ignore_index=True)
        df.to_parquet(DIRS['raw'] / f'{dataset_key}_parsed_fallback.parquet')
        return df

def infer_label_column(df, dataset_key):
    cols = list(df.columns)
    for cand in DATASET_SPECS[dataset_key]['label_candidates']:
        for col in cols:
            if cand in safe_lower(col): return col
    terms = ['target','label','class','activity','occupancy','room_occupancy_count']
    cands = [c for c in cols if any(t in safe_lower(c) for t in terms)]
    if cands: return cands[-1]
    raise ValueError(f'Could not infer label column for {dataset_key}; columns={cols[:50]}')

def infer_timestamp(df):
    cols = list(df.columns)
    date_cols = [c for c in cols if 'date' in safe_lower(c)]
    time_cols = [c for c in cols if 'time' in safe_lower(c) or 'timestamp' in safe_lower(c)]
    if date_cols and time_cols:
        dt = pd.to_datetime(df[date_cols[0]].astype(str) + ' ' + df[time_cols[0]].astype(str), errors='coerce')
        if dt.notna().mean() > 0.5: return dt
    for c in cols:
        lc = safe_lower(c)
        if any(t in lc for t in ['datetime','timestamp','time','date']):
            dt = pd.to_datetime(df[c], errors='coerce')
            if dt.notna().mean() > 0.5: return dt
            num = pd.to_numeric(df[c], errors='coerce')
            if num.notna().mean() > 0.5: return pd.Series(num.values, index=df.index)
    return pd.Series(np.arange(len(df), dtype=float), index=df.index)

def encode_features(df, label_col, timestamp_cols=None, max_features=128):
    drop = {label_col}
    if timestamp_cols: drop.update(timestamp_cols)
    cand = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore').replace(['?','NA','N/A','nan','NaN','None',''], np.nan)
    num_cols, cat_cols = [], []
    for c in cand.columns:
        s = pd.to_numeric(cand[c], errors='coerce')
        if s.notna().mean() > 0.8:
            cand[c] = s; num_cols.append(c)
        else:
            nunique = cand[c].astype(str).nunique(dropna=True)
            if 1 < nunique <= 100: cat_cols.append(c)
    Xn = cand[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)
    for c in Xn.columns: Xn[c] = Xn[c].fillna(Xn[c].median())
    Xn = Xn.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    X = pd.concat([Xn, pd.get_dummies(cand[cat_cols].astype(str), dummy_na=True)], axis=1) if cat_cols else Xn
    X = X.loc[:, X.nunique(dropna=False) > 1]
    if X.shape[1] > max_features:
        keep = X.astype(float).var(axis=0).sort_values(ascending=False).head(max_features).index
        X = X[keep]
    return X.astype(np.float32), list(X.columns)

def chronological_split_indices(n, train_ratio=0.6, val_ratio=0.2):
    n_train = int(n*train_ratio); n_val = int(n*val_ratio)
    return np.arange(n_train), np.arange(n_train, n_train+n_val), np.arange(n_train+n_val, n)

def make_windows(X, y, times, L, stride, max_windows=None, label_rule='last'):
    if len(y) < L: return np.empty((0,L,X.shape[1]),np.float32), np.empty((0,),np.int64), np.empty((0,L,1),np.float32)
    ts_series = pd.Series(times)
    if np.issubdtype(ts_series.dropna().values.dtype, np.datetime64):
        ts = ts_series.astype('datetime64[ns]').astype('int64').values / 1e9
    else:
        ts = pd.to_numeric(ts_series, errors='coerce').ffill().fillna(0).values.astype(float)
    diffs = np.diff(ts); pos = diffs[np.isfinite(diffs) & (diffs > 0)]
    denom = np.median(pos) if len(pos) else 1.0
    if not np.isfinite(denom) or denom <= 0: denom = 1.0
    delta_all = np.zeros(len(y), dtype=np.float32); delta_all[1:] = np.clip(diffs/denom, 0, 100).astype(np.float32)
    Xw, yw, Dw = [], [], []
    for start in range(0, len(y)-L+1, stride):
        end = start+L
        lab = y[end-1] if label_rule == 'last' else np.bincount(y[start:end]).argmax()
        Xw.append(X[start:end]); yw.append(lab); Dw.append(delta_all[start:end].reshape(L,1))
        if max_windows is not None and len(yw) >= max_windows: break
    return np.stack(Xw).astype(np.float32), np.asarray(yw, dtype=np.int64), np.stack(Dw).astype(np.float32)

def prepare_temporal_dataset(dataset_key, force_refresh=False, label_rule='last'):
    out = DIRS['processed'] / f'{dataset_key}_L{CFG.window_length}_S{CFG.stride}_{RUN_MODE}.npz'
    meta_path = DIRS['processed'] / f'{dataset_key}_meta_{RUN_MODE}.json'
    if out.exists() and not force_refresh:
        data = dict(np.load(out, allow_pickle=True)); data['meta'] = json.loads(meta_path.read_text()); return data
    df = load_dataset_with_fallback(dataset_key).copy()
    df.columns = [safe_lower(c) for c in df.columns]
    label_col = infer_label_column(df, dataset_key)
    times = infer_timestamp(df)
    order = np.argsort(pd.Series(times).fillna(pd.Series(times).dropna().min() if pd.Series(times).notna().any() else 0).values)
    df = df.iloc[order].reset_index(drop=True); times = pd.Series(times).iloc[order].reset_index(drop=True)
    label_raw = df[label_col].astype(str).replace({'nan': np.nan, 'None': np.nan})
    valid = label_raw.notna(); df = df.loc[valid].reset_index(drop=True); times = times.loc[valid].reset_index(drop=True); label_raw = label_raw.loc[valid].reset_index(drop=True)
    counts = label_raw.value_counts(); keep_classes = counts[counts >= CFG.min_class_count].index
    keep = label_raw.isin(keep_classes); df = df.loc[keep].reset_index(drop=True); times = times.loc[keep].reset_index(drop=True); label_raw = label_raw.loc[keep].reset_index(drop=True)
    le = LabelEncoder(); y_all = le.fit_transform(label_raw).astype(np.int64)
    time_cols = [c for c in df.columns if any(t in safe_lower(c) for t in ['date','time','timestamp'])]
    Xdf, feature_names = encode_features(df, label_col, time_cols, CFG.max_features)
    X_all = Xdf.values.astype(np.float32)
    tr, va, te = chronological_split_indices(len(y_all), CFG.train_ratio, CFG.val_ratio)
    scaler = StandardScaler(); Xtr0 = scaler.fit_transform(X_all[tr]).astype(np.float32); Xva0 = scaler.transform(X_all[va]).astype(np.float32); Xte0 = scaler.transform(X_all[te]).astype(np.float32)
    Xtr,ytr,Dtr = make_windows(Xtr0, y_all[tr], times.iloc[tr].values, CFG.window_length, CFG.stride, MAX_WINDOWS_PER_SPLIT, label_rule)
    Xva,yva,Dva = make_windows(Xva0, y_all[va], times.iloc[va].values, CFG.window_length, CFG.stride, MAX_WINDOWS_PER_SPLIT, label_rule)
    Xte,yte,Dte = make_windows(Xte0, y_all[te], times.iloc[te].values, CFG.window_length, CFG.stride, MAX_WINDOWS_PER_SPLIT, label_rule)
    meta = {'dataset_key': dataset_key, 'dataset_name': DATASET_SPECS[dataset_key]['name'], 'label_col': label_col, 'classes': list(map(str, le.classes_)), 'n_classes': int(len(le.classes_)), 'n_features': int(X_all.shape[1]), 'feature_names': feature_names[:200], 'splits_windows': {'train': len(ytr), 'val': len(yva), 'test': len(yte)}, 'window_length': CFG.window_length, 'stride': CFG.stride}
    np.savez_compressed(out, X_train=Xtr, y_train=ytr, delta_train=Dtr, X_val=Xva, y_val=yva, delta_val=Dva, X_test=Xte, y_test=yte, delta_test=Dte)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2)[:1500])
    return {'X_train': Xtr, 'y_train': ytr, 'delta_train': Dtr, 'X_val': Xva, 'y_val': yva, 'delta_val': Dva, 'X_test': Xte, 'y_test': yte, 'delta_test': Dte, 'meta': meta}

def inspect_all_uci_datasets():
    rows=[]
    for key, spec in DATASET_SPECS.items():
        try:
            df,_ = fetch_uci_as_dataframe(key)
            rows.append({'dataset': key, 'status': 'ok', 'rows': len(df), 'cols': df.shape[1], 'columns': ', '.join(df.columns[:20])})
        except Exception as e:
            rows.append({'dataset': key, 'status': f'failed: {e}', 'rows': None, 'cols': None, 'columns': ''})
    s = pd.DataFrame(rows); s.to_csv(DIRS['tables']/'dataset_fetch_summary.csv', index=False); return s




# Cell 4. Dataset wrapper and feature-noise config
@dataclass
class NoiseConfig:
    gaussian_std: float = 0.0
    channel_missing_rate: float = 0.0
    temporal_thinning_rate: float = 0.0
    dropout_rate: float = 0.0
    finite_shots: Optional[int] = None

class TemporalWindowDataset(Dataset):
    def __init__(self, X, y, delta, noise=None, training=True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.delta = torch.tensor(delta, dtype=torch.float32)
        self.noise = noise or NoiseConfig()
        self.training = training
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx].clone(); delta = self.delta[idx].clone(); y = self.y[idx]
        L, d = x.shape; mask = torch.ones_like(x)
        if self.training:
            if self.noise.channel_missing_rate > 0:
                keep = torch.rand_like(x) > self.noise.channel_missing_rate
                x = x * keep.float(); mask = mask * keep.float()
            if self.noise.temporal_thinning_rate > 0:
                keep_t = torch.rand(L) > self.noise.temporal_thinning_rate
                keep_t[0] = True; keep_t[-1] = True
                x = x * keep_t.float().view(L,1); mask = mask * keep_t.float().view(L,1)
                new_delta = delta.clone(); acc = torch.tensor(0.0)
                for t in range(L):
                    if t == 0: new_delta[t,0] = 0.0; acc = torch.tensor(0.0); continue
                    acc = acc + delta[t,0]; new_delta[t,0] = acc
                    if keep_t[t]: acc = torch.tensor(0.0)
                delta = new_delta
        z = torch.cat([x, mask, delta], dim=-1)
        return z, y

def make_loaders(data, noise_train, noise_eval=None, eval_noise_active=False):
    noise_eval = noise_eval or NoiseConfig()
    train_ds = TemporalWindowDataset(data['X_train'], data['y_train'], data['delta_train'], noise_train, training=True)
    val_ds = TemporalWindowDataset(data['X_val'], data['y_val'], data['delta_val'], noise_eval, training=eval_noise_active)
    test_ds = TemporalWindowDataset(data['X_test'], data['y_test'], data['delta_test'], noise_eval, training=eval_noise_active)
    return (DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=0),
            DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=0),
            DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=0))


# Cell 5. Models: GRU, GRU-D style, ODE-RNN style, LTC style, and CaTRe gate
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, phi_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, phi_dim), nn.ReLU())
    def forward(self, z): return self.net(z)

class TemporalModel(nn.Module):
    def __init__(self, input_dim, n_classes, phi_dim=64, hidden_dim=64, backbone='gru', use_catre=False, use_gate=True, noise_config=None):
        super().__init__()
        self.input_dim=input_dim; self.n_classes=n_classes; self.phi_dim=phi_dim; self.hidden_dim=hidden_dim; self.backbone=backbone
        self.use_catre=use_catre; self.use_gate=use_gate and use_catre; self.noise_config=noise_config or NoiseConfig()
        self.encoder = FeatureEncoder(input_dim, phi_dim, hidden_dim)
        if backbone in ['gru','grud','ode_rnn']: self.gru_cell = nn.GRUCell(phi_dim, hidden_dim)
        if backbone in ['ltc','ode_rnn']:
            self.f_h = nn.Linear(hidden_dim, hidden_dim); self.f_phi = nn.Linear(phi_dim, hidden_dim)
        if backbone == 'ltc':
            self.tau_h = nn.Linear(hidden_dim, hidden_dim); self.tau_phi = nn.Linear(phi_dim, hidden_dim)
        if backbone == 'grud': self.decay = nn.Linear(1, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim + phi_dim + phi_dim + 1, hidden_dim), nn.Sigmoid()) if self.use_gate else None
        self.out = nn.Linear(hidden_dim, n_classes)
    def apply_rep_noise(self, phi, sample_noise=True, override_noise=None):
        noise = override_noise or self.noise_config
        phihat = phi; sigma_diag = torch.zeros_like(phi)
        if sample_noise and noise.gaussian_std > 0:
            phihat = phihat + torch.randn_like(phi) * noise.gaussian_std
            sigma_diag = sigma_diag + noise.gaussian_std**2
        if sample_noise and noise.dropout_rate > 0:
            phihat = F.dropout(phihat, p=noise.dropout_rate, training=True)
            sigma_diag = sigma_diag + noise.dropout_rate * (phi.detach()**2)
        if sample_noise and noise.finite_shots is not None:
            S = noise.finite_shots
            if S != math.inf and S > 0:
                q = torch.tanh(phihat)
                if S >= 100:
                    var = torch.clamp((1-q**2)/S, min=1e-8)
                    qhat = torch.clamp(q + torch.randn_like(q)*torch.sqrt(var), -1, 1)
                else:
                    prob_plus = torch.clamp((q+1)/2, 1e-6, 1-1e-6)
                    count_plus = torch.distributions.Binomial(total_count=S, probs=prob_plus).sample()
                    qhat = 2*count_plus/S - 1
                    var = torch.clamp((1-q**2)/S, min=1e-8)
                phihat = qhat; sigma_diag = sigma_diag + var
        return phihat, sigma_diag
    def step(self, h, phi_hat, sigma_diag, delta):
        if self.backbone == 'gru':
            new_h = self.gru_cell(phi_hat, h); inc = new_h - h; scale = 1.0
        elif self.backbone == 'grud':
            gamma = torch.exp(-F.softplus(self.decay(delta))); new_h = self.gru_cell(phi_hat, gamma*h); inc = new_h - h; scale = 1.0
        elif self.backbone == 'ode_rnn':
            h_ode = h + delta * torch.tanh(self.f_h(h)); new_h = self.gru_cell(phi_hat, h_ode); inc = new_h - h; scale = 1.0
        elif self.backbone == 'ltc':
            tau = torch.sigmoid(self.tau_h(h) + self.tau_phi(phi_hat)) + 0.1
            inc = -h/tau + torch.tanh(self.f_h(h) + self.f_phi(phi_hat)); scale = delta
        else:
            raise ValueError(self.backbone)
        if self.use_gate:
            u = torch.log(torch.clamp(sigma_diag, min=1e-8))
            g = self.gate(torch.cat([h, phi_hat, u, delta], dim=-1))
            inc = g * inc
        return h + scale * inc
    def forward(self, z, sample_noise=True, override_noise=None, return_hidden=False):
        B,L,Fdim = z.shape; h = torch.zeros(B, self.hidden_dim, device=z.device); hs=[]
        for t in range(L):
            zt = z[:,t,:]; delta = zt[:,-1:].clamp(0,100)
            phi = self.encoder(zt); phihat, sigma = self.apply_rep_noise(phi, sample_noise, override_noise)
            h = self.step(h, phihat, sigma, delta)
            if return_hidden: hs.append(h)
        logits = self.out(h)
        return (logits, torch.stack(hs, dim=1)) if return_hidden else logits

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Metrics and plots

def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(y_true, probs, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append({'bin': i, 'lo': lo, 'hi': hi, 'count': 0, 'acc': np.nan, 'conf': np.nan, 'gap': np.nan})
            continue
        acc = correct[m].mean()
        cf = conf[m].mean()
        gap = abs(acc - cf)
        ece += m.mean() * gap
        mce = max(mce, gap)
        rows.append({'bin': i, 'lo': lo, 'hi': hi, 'count': int(m.sum()), 'acc': acc, 'conf': cf, 'gap': gap})
    return float(ece), float(mce), pd.DataFrame(rows)


def multiclass_brier(y_true, probs):
    y_true = np.asarray(y_true).astype(int)
    Y = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - Y) ** 2, axis=1)))


def compute_metrics(y_true, probs):
    y_true = np.asarray(y_true).astype(int)
    C = probs.shape[1]
    pred = probs.argmax(1)
    p = np.clip(probs, 1e-12, 1.0)
    p = p / p.sum(1, keepdims=True)
    out = {
        'accuracy': float(accuracy_score(y_true, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
        'macro_f1': float(f1_score(y_true, pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, pred, average='weighted', zero_division=0)),
        'nll': float(log_loss(y_true, p, labels=list(range(C)))) if len(np.unique(y_true)) > 1 else np.nan,
        'brier': multiclass_brier(y_true, p),
    }
    ece, mce, _ = expected_calibration_error(y_true, p, CFG.ece_bins)
    out['ece'] = ece
    out['mce'] = mce
    try:
        if C == 2:
            out['auroc'] = float(roc_auc_score(y_true, p[:, 1]))
            out['ap'] = float(average_precision_score(y_true, p[:, 1]))
        else:
            out['auroc'] = float(roc_auc_score(y_true, p, multi_class='ovr', average='macro'))
            out['ap'] = float(average_precision_score(np.eye(C)[y_true], p, average='macro'))
    except Exception:
        out['auroc'] = np.nan
        out['ap'] = np.nan
    return out


def plot_reliability(y_true, probs, out_path, title):
    ece, mce, bin_df = expected_calibration_error(y_true, probs, CFG.ece_bins)
    fig = plt.figure(figsize=(5, 4.5))
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    valid = bin_df[bin_df['count'] > 0]
    plt.plot(valid['conf'], valid['acc'], marker='o')
    plt.xlabel('Confidence')
    plt.ylabel('Empirical accuracy')
    plt.title(f'{title}\nECE={ece:.4f}, MCE={mce:.4f}')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return bin_df


def plot_confidence_hist(y_true, probs, out_path, title):
    conf = probs.max(1)
    correct = probs.argmax(1) == y_true
    fig = plt.figure(figsize=(5, 4))
    hist_bins = np.linspace(0.0, 1.0, CFG.ece_bins + 1)
    if np.any(correct):
        plt.hist(conf[correct], bins=hist_bins, alpha=.7, label='correct')
    if np.any(~correct):
        plt.hist(conf[~correct], bins=hist_bins, alpha=.7, label='incorrect')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# Cell 7. Training and evaluation
@torch.no_grad()
def evaluate_model(model, loader, noise_override=None, sample_noise=False, save_prefix=None):
    model.eval(); logits_all=[]; y_all=[]; hts_vals=[]
    noise_override = noise_override or NoiseConfig()
    for z,y in loader:
        z = z.to(DEVICE)
        logits,hseq = model(z, sample_noise=sample_noise, override_noise=noise_override, return_hidden=True)
        logits_all.append(logits.detach().cpu().numpy()); y_all.append(y.numpy())
        if sample_noise:
            _,hseq2 = model(z, sample_noise=True, override_noise=noise_override, return_hidden=True)
            hts_vals.append(float(F.mse_loss(hseq,hseq2).detach().cpu()))
    logits_np = np.concatenate(logits_all); y_np = np.concatenate(y_all); probs = softmax_np(logits_np)
    metrics = compute_metrics(y_np, probs); metrics['hts'] = float(np.mean(hts_vals)) if hts_vals else np.nan
    if save_prefix:
        pred = pd.DataFrame(probs, columns=[f'prob_{i}' for i in range(probs.shape[1])]); pred.insert(0,'y_true',y_np); pred['y_pred']=probs.argmax(1)
        pred.to_csv(DIRS['predictions']/f'{save_prefix}_predictions.csv', index=False)
        bins = plot_reliability(y_np, probs, DIRS['figures']/f'{save_prefix}_reliability.png', save_prefix)
        bins.to_csv(DIRS['tables']/f'{save_prefix}_calibration_bins.csv', index=False)
        plot_confidence_hist(y_np, probs, DIRS['figures']/f'{save_prefix}_confidence_hist.png', save_prefix)
    return metrics, y_np, probs

def train_one_model(dataset_key, data, backbone, use_catre, seed, train_noise, eval_noise=None, lambda_cal=0.05, lambda_stab=0.01, ablation_name='full', use_gate=True):
    set_seed(seed); eval_noise = eval_noise or NoiseConfig()
    input_dim = data['X_train'].shape[-1]*2 + 1; n_classes = int(data['meta']['n_classes'])
    model = TemporalModel(input_dim,n_classes,CFG.phi_dim,CFG.hidden_dim,backbone,use_catre,use_gate,train_noise).to(DEVICE)
    train_loader,val_loader,test_loader = make_loaders(data, train_noise, eval_noise, eval_noise_active=False)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    best_state=None; best_val=float('inf'); patience=0; history=[]
    for epoch in range(1, CFG.max_epochs+1):
        model.train(); losses=[]
        for z,y in train_loader:
            z,y = z.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            logits,hseq = model(z, sample_noise=True, return_hidden=True)
            ce = F.cross_entropy(logits,y); probs = F.softmax(logits,dim=-1); yoh = F.one_hot(y,n_classes).float()
            cal = torch.mean(torch.sum((probs-yoh)**2,dim=-1)) if use_catre and lambda_cal>0 else torch.tensor(0.,device=DEVICE)
            if use_catre and lambda_stab>0:
                _,hseq2 = model(z, sample_noise=True, return_hidden=True); stab = F.mse_loss(hseq,hseq2)
            else: stab = torch.tensor(0.,device=DEVICE)
            loss = ce + lambda_cal*cal + lambda_stab*stab; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach().cpu()))
        val_metrics,_,_ = evaluate_model(model,val_loader,NoiseConfig(),sample_noise=False)
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)), **{f'val_{k}':v for k,v in val_metrics.items()}})
        if val_metrics['nll'] < best_val:
            best_val = val_metrics['nll']; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
        else: patience += 1
        print(f'[{dataset_key}|{backbone}|catre={use_catre}|seed={seed}] epoch={epoch} train_loss={np.mean(losses):.4f} val_nll={val_metrics["nll"]:.4f} val_ece={val_metrics["ece"]:.4f}')
        if patience >= CFG.patience: break
    if best_state: model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
    run_id = f'{dataset_key}_{backbone}_{"catre" if use_catre else "base"}_{ablation_name}_seed{seed}'
    torch.save({'model_state_dict': model.state_dict(), 'dataset_meta': data['meta'], 'backbone': backbone, 'use_catre': use_catre, 'train_noise': asdict(train_noise), 'params': count_parameters(model), 'config': asdict(CFG)}, DIRS['models']/f'{run_id}.pt')
    pd.DataFrame(history).to_csv(DIRS['logs']/f'{run_id}_history.csv', index=False)
    test_metrics,_,_ = evaluate_model(model,test_loader,NoiseConfig(),sample_noise=False,save_prefix=run_id)
    test_metrics.update({'dataset':dataset_key,'model':backbone,'method':'CaTRe' if use_catre else 'Base','ablation':ablation_name,'seed':seed,'params':count_parameters(model),'model_path':str(DIRS['models']/f'{run_id}.pt')})
    return test_metrics


# Cell 8. Synthetic dataset and experiment runners

def generate_synthetic_temporal_dataset(n_train=500,n_val=200,n_test=200,L=32,d=8,n_classes=2,seed=42):
    rng=np.random.default_rng(seed); n=n_train+n_val+n_test; y=rng.integers(0,n_classes,size=n)
    X=np.zeros((n,L,d),np.float32); delta=np.ones((n,L,1),np.float32); delta[:,0,0]=0
    for i in range(n):
        cls=y[i]; freq=.12+.08*cls; amp=1+.4*cls; phase=rng.uniform(0,2*np.pi); t=np.arange(L)
        for j in range(d): X[i,:,j]=amp*np.sin(freq*t+phase+j*.25)+.1*rng.normal(size=L)
        X[i,:,0]+=cls*np.linspace(0,1,L)
    i1=n_train; i2=n_train+n_val
    return {'X_train':X[:i1],'y_train':y[:i1].astype(np.int64),'delta_train':delta[:i1], 'X_val':X[i1:i2],'y_val':y[i1:i2].astype(np.int64),'delta_val':delta[i1:i2], 'X_test':X[i2:],'y_test':y[i2:].astype(np.int64),'delta_test':delta[i2:], 'meta':{'dataset_key':'synthetic','dataset_name':'Synthetic controlled temporal task','n_classes':n_classes,'n_features':d,'classes':[str(i) for i in range(n_classes)]}}

synthetic_data = generate_synthetic_temporal_dataset(n_train=500 if RUN_MODE=='smoke' else 3000, n_val=200 if RUN_MODE=='smoke' else 800, n_test=200 if RUN_MODE=='smoke' else 800, L=CFG.window_length)

BASE_BACKBONES = ['gru','grud','ode_rnn','ltc']
MAIN_DATASETS = ['opportunity','ambient_har','wisdm','room_occupancy']
if RUN_MODE == 'smoke':
    BASE_BACKBONES = ['gru','ltc']
    MAIN_DATASETS = ['occupancy_detection']

def run_main_benchmark(dataset_keys, backbones):
    rows=[]
    for dk in dataset_keys:
        try: data = prepare_temporal_dataset(dk)
        except Exception as e: print('skip dataset',dk,e); continue
        for backbone in backbones:
            for seed in CFG.seeds:
                row = train_one_model(dk,data,backbone,False,seed,NoiseConfig(),NoiseConfig(),0,0,'base',False); rows.append(row)
                pd.DataFrame(rows).to_csv(DIRS['tables']/ 'main_benchmark_partial.csv', index=False)
                row = train_one_model(dk,data,backbone,True,seed,NoiseConfig(gaussian_std=.10),NoiseConfig(),.05,.01,'full',True); rows.append(row)
                pd.DataFrame(rows).to_csv(DIRS['tables']/ 'main_benchmark_partial.csv', index=False)
    df=pd.DataFrame(rows); df.to_csv(DIRS['tables']/ 'main_benchmark_results.csv', index=False); return df

def run_noise_robustness(dataset_key, backbone='gru', sigmas=(0,.05,.10,.20,.40)):
    data=prepare_temporal_dataset(dataset_key); rows=[]
    for use_catre in [False, True]:
        for seed in CFG.seeds:
            set_seed(seed)
            train_noise=NoiseConfig(gaussian_std=.10 if use_catre else 0)
            input_dim=data['X_train'].shape[-1]*2+1; n_classes=int(data['meta']['n_classes'])
            model=TemporalModel(input_dim,n_classes,CFG.phi_dim,CFG.hidden_dim,backbone,use_catre,use_catre,train_noise).to(DEVICE)
            train_loader,val_loader,test_loader=make_loaders(data,train_noise,NoiseConfig())
            opt=torch.optim.AdamW(model.parameters(),lr=CFG.lr,weight_decay=CFG.weight_decay)
            best_state=None; best_val=float('inf'); patience=0
            for epoch in range(1, CFG.max_epochs+1):
                model.train()
                for z,y in train_loader:
                    z,y=z.to(DEVICE),y.to(DEVICE); opt.zero_grad(); logits,hseq=model(z,True,return_hidden=True)
                    # Gate-only CaTRe for noise robustness:
                    # use uncertainty-gated temporal updates, but no auxiliary calibration/stability losses.
                    ce=F.cross_entropy(logits,y)
                    loss=ce
                    loss.backward()
                    opt.step()
                vm,_,_=evaluate_model(model,val_loader,NoiseConfig(),False)
                if vm['nll']<best_val: best_val=vm['nll']; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
                else: patience+=1
                if patience>=CFG.patience: break
            if best_state: model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
            for sigma in sigmas:
                metrics,_,_=evaluate_model(model,test_loader,NoiseConfig(gaussian_std=sigma),sample_noise=(sigma>0),save_prefix=f'noise_{dataset_key}_{backbone}_{"catre" if use_catre else "base"}_sigma{sigma}_seed{seed}')
                metrics.update({'dataset':dataset_key,'backbone':backbone,'method':'CaTRe' if use_catre else 'Base','seed':seed,'sigma':sigma})
                rows.append(metrics); pd.DataFrame(rows).to_csv(DIRS['tables']/f'noise_robustness_{dataset_key}_{backbone}.csv', index=False)
    return pd.DataFrame(rows)

def run_catre_ablation(dataset_key, backbone='gru'):
    data=prepare_temporal_dataset(dataset_key); rows=[]
    variants=[('base',False,False,0,0),('gate_only',True,True,0,0),('cal_loss_only',True,False,.05,0),('stability_only',True,False,0,.01),('full_catre',True,True,.05,.01)]
    for name,use_catre,use_gate,lc,ls in variants:
        for seed in CFG.seeds:
            row=train_one_model(dataset_key,data,backbone,use_catre,seed,NoiseConfig(gaussian_std=.10 if use_catre else 0),NoiseConfig(),lc,ls,name,use_gate)
            row['ablation_variant']=name; rows.append(row); pd.DataFrame(rows).to_csv(DIRS['tables']/f'ablation_{dataset_key}_{backbone}.csv', index=False)
    return pd.DataFrame(rows)

def run_finite_shot_case_study(dataset_key='occupancy_detection', backbone='ltc', shots=(10,50,100,500,math.inf)):
    data=prepare_temporal_dataset(dataset_key); rows=[]
    for use_catre in [False, True]:
        for seed in CFG.seeds:
            # Train once, evaluate under finite-shot representation estimators.
            train_noise=NoiseConfig(finite_shots=50 if use_catre else None)
            row_train=train_one_model(dataset_key,data,backbone,use_catre,seed,train_noise,NoiseConfig(),.05 if use_catre else 0,.01 if use_catre else 0,'finite_train',use_catre)
            input_dim=data['X_train'].shape[-1]*2+1; n_classes=int(data['meta']['n_classes'])
            ckpt=torch.load(row_train['model_path'], map_location=DEVICE)
            model=TemporalModel(input_dim,n_classes,CFG.phi_dim,CFG.hidden_dim,backbone,use_catre,use_catre,train_noise).to(DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            _,_,test_loader=make_loaders(data,NoiseConfig(),NoiseConfig())
            for S in shots:
                ev=NoiseConfig(finite_shots=None if S==math.inf else int(S))
                metrics,_,_=evaluate_model(model,test_loader,ev,sample_noise=(S!=math.inf),save_prefix=f'finite_shot_{dataset_key}_{backbone}_{"catre" if use_catre else "base"}_S{S}_seed{seed}')
                metrics.update({'dataset':dataset_key,'backbone':backbone,'method':'CaTRe' if use_catre else 'Base','seed':seed,'shots':'inf' if S==math.inf else int(S)})
                rows.append(metrics); pd.DataFrame(rows).to_csv(DIRS['tables']/f'finite_shot_{dataset_key}_{backbone}.csv', index=False)
    return pd.DataFrame(rows)

def plot_noise_curves(df, out_prefix):
    for metric in ['accuracy','nll','brier','ece','hts']:
        if metric not in df.columns: continue
        fig=plt.figure(figsize=(5.5,4))
        for method,sub in df.groupby('method'):
            agg=sub.groupby('sigma')[metric].agg(['mean','std']).reset_index()
            plt.plot(agg['sigma'],agg['mean'],marker='o',label=method)
            plt.fill_between(agg['sigma'], agg['mean']-agg['std'].fillna(0), agg['mean']+agg['std'].fillna(0), alpha=.2)
        plt.xlabel('Representation noise std'); plt.ylabel(metric); plt.title(f'{out_prefix}: {metric}'); plt.legend(); plt.tight_layout(); fig.savefig(DIRS['figures']/f'{out_prefix}_{metric}_curve.png',dpi=200); plt.close(fig)


# Cell 8b. Missingness and temporal-sparsification robustness runner

def run_missingness_robustness(dataset_key, backbone='gru', channel_rates=(0.0,0.1,0.3,0.5), temporal_rates=(0.0,0.2,0.4,0.6)):
    data = prepare_temporal_dataset(dataset_key)
    rows = []
    # Train a clean Base model and a CaTRe model with mild representation noise.
    trained = []
    for use_catre in [False, True]:
        for seed in CFG.seeds:
            train_noise = NoiseConfig(gaussian_std=.10 if use_catre else 0.0)
            input_dim = data['X_train'].shape[-1]*2 + 1
            n_classes = int(data['meta']['n_classes'])
            model = TemporalModel(input_dim,n_classes,CFG.phi_dim,CFG.hidden_dim,backbone,use_catre,use_catre,train_noise).to(DEVICE)
            train_loader,val_loader,_ = make_loaders(data, train_noise, NoiseConfig())
            opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
            best_state=None; best_val=float('inf'); patience=0
            for epoch in range(1, CFG.max_epochs+1):
                model.train()
                for z,y in train_loader:
                    z,y = z.to(DEVICE), y.to(DEVICE); opt.zero_grad()
                    logits,hseq = model(z, sample_noise=True, return_hidden=True)
                    ce = F.cross_entropy(logits,y)
                    probs = F.softmax(logits,dim=-1); yoh = F.one_hot(y,n_classes).float()
                    # Gate-only CaTRe for missingness robustness:
                    # use uncertainty-gated temporal updates, but no auxiliary calibration/stability losses.
                    loss = ce
                    loss.backward()
                    opt.step()
                vm,_,_ = evaluate_model(model,val_loader,NoiseConfig(),False)
                if vm['nll'] < best_val:
                    best_val = vm['nll']; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
                else:
                    patience += 1
                if patience >= CFG.patience: break
            if best_state: model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
            run_id = f'missing_train_{dataset_key}_{backbone}_{"catre" if use_catre else "base"}_seed{seed}'
            torch.save({'model_state_dict': model.state_dict(), 'dataset_meta': data['meta'], 'backbone': backbone, 'use_catre': use_catre, 'train_noise': asdict(train_noise), 'params': count_parameters(model), 'config': asdict(CFG)}, DIRS['models']/f'{run_id}.pt')
            trained.append((model, use_catre, seed))
    # Evaluate channel missingness and temporal thinning. Input-level noise is activated in the Dataset wrapper.
    for model,use_catre,seed in trained:
        for rho_c in channel_rates:
            noise_eval = NoiseConfig(channel_missing_rate=float(rho_c))
            _,_,test_loader = make_loaders(data, NoiseConfig(), noise_eval, eval_noise_active=True)
            metrics,_,_ = evaluate_model(model,test_loader,NoiseConfig(),sample_noise=False,save_prefix=f'missing_channel_{dataset_key}_{backbone}_{"catre" if use_catre else "base"}_rho{rho_c}_seed{seed}')
            metrics.update({'dataset':dataset_key,'backbone':backbone,'method':'CaTRe' if use_catre else 'Base','seed':seed,'missingness_type':'channel','rho':rho_c})
            rows.append(metrics)
        for rho_t in temporal_rates:
            noise_eval = NoiseConfig(temporal_thinning_rate=float(rho_t))
            _,_,test_loader = make_loaders(data, NoiseConfig(), noise_eval, eval_noise_active=True)
            metrics,_,_ = evaluate_model(model,test_loader,NoiseConfig(),sample_noise=False,save_prefix=f'missing_temporal_{dataset_key}_{backbone}_{"catre" if use_catre else "base"}_rho{rho_t}_seed{seed}')
            metrics.update({'dataset':dataset_key,'backbone':backbone,'method':'CaTRe' if use_catre else 'Base','seed':seed,'missingness_type':'temporal','rho':rho_t})
            rows.append(metrics)
        pd.DataFrame(rows).to_csv(DIRS['tables']/f'missingness_robustness_{dataset_key}_{backbone}.csv', index=False)
    return pd.DataFrame(rows)

def plot_missingness_curves(df, out_prefix):
    for metric in ['accuracy','nll','brier','ece']:
        if metric not in df.columns: continue
        for miss_type, sub0 in df.groupby('missingness_type'):
            fig=plt.figure(figsize=(5.5,4))
            for method,sub in sub0.groupby('method'):
                agg=sub.groupby('rho')[metric].agg(['mean','std']).reset_index()
                plt.plot(agg['rho'],agg['mean'],marker='o',label=method)
                plt.fill_between(agg['rho'], agg['mean']-agg['std'].fillna(0), agg['mean']+agg['std'].fillna(0), alpha=.2)
            plt.xlabel(f'{miss_type} missingness rate'); plt.ylabel(metric); plt.title(f'{out_prefix}: {miss_type} {metric}'); plt.legend(); plt.tight_layout()
            fig.savefig(DIRS['figures']/f'{out_prefix}_{miss_type}_{metric}_curve.png', dpi=200); plt.close(fig)



# HEC selected-job runner

def prepare_job_data(dataset_key):
    if dataset_key == 'synthetic':
        return synthetic_data
    return prepare_temporal_dataset(dataset_key)


def variant_settings(variant, gaussian_std=0.0):
    if variant == 'base':
        return False, False, 0.0, 0.0, NoiseConfig(gaussian_std=gaussian_std)
    if variant in ['catre', 'full_catre']:
        return True, True, 0.05, 0.01, NoiseConfig(gaussian_std=gaussian_std if gaussian_std > 0 else 0.10)
    if variant == 'gate_only':
        return True, True, 0.0, 0.0, NoiseConfig(gaussian_std=gaussian_std if gaussian_std > 0 else 0.10)
    if variant == 'cal_loss_only':
        return True, False, 0.05, 0.0, NoiseConfig(gaussian_std=gaussian_std if gaussian_std > 0 else 0.10)
    if variant == 'stability_only':
        return True, False, 0.0, 0.01, NoiseConfig(gaussian_std=gaussian_std if gaussian_std > 0 else 0.10)
    raise ValueError(f'Unknown variant: {variant}')


def result_path_for(args):
    name = f"{args.experiment}__{args.dataset}__{args.backbone}__{args.variant}__seed{args.seed}__noise{args.gaussian_std}.csv"
    return DIRS['tables'] / name


def _load_selected_trained_model(data, args, row, use_catre, use_gate, train_noise):
    """Reload exactly the checkpoint trained by this one job."""
    input_dim = data['X_train'].shape[-1] * 2 + 1
    n_classes = int(data['meta']['n_classes'])
    model = TemporalModel(input_dim, n_classes, CFG.phi_dim, CFG.hidden_dim,
                          args.backbone, use_catre, use_gate, train_noise).to(DEVICE)
    ckpt = torch.load(row['model_path'], map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def _train_selected_variant(args, data):
    """Exactly one variant/seed/job; reuse existing training method unchanged."""
    use_catre, use_gate, lc, ls, train_noise = variant_settings(
        args.variant, args.gaussian_std)
    row = train_one_model(args.dataset, data, args.backbone, use_catre,
                          args.seed, train_noise, NoiseConfig(), lc, ls,
                          f'{args.experiment}_{args.variant}', use_gate)
    return row, use_catre, use_gate, lc, ls, train_noise


def _run_selected_robustness(args, data):
    """Train one requested variant/seed, then test its corruption sweep."""
    row, use_catre, use_gate, lc, ls, train_noise = _train_selected_variant(args, data)
    model = _load_selected_trained_model(data, args, row,
                                        use_catre, use_gate, train_noise)
    rows = []
    prefix = f'{args.experiment}_{args.dataset}_{args.backbone}_{args.variant}_seed{args.seed}'
    for kind, rate in (
        [('gaussian', s) for s in (0.0, 0.05, 0.10, 0.20, 0.40)]
        if args.experiment == 'noise' else
        [('channel', r) for r in (0.0, 0.1, 0.3, 0.5)] +
        [('temporal', r) for r in (0.0, 0.2, 0.4, 0.6)]
    ):
        # Deterministic, matched perturbation stream for this test condition.
        condition_id = {'gaussian': 1, 'channel': 2, 'temporal': 3}[kind]
        set_seed(100000 + args.seed * 10000 + condition_id * 1000 + int(round(rate * 1000)))
        if kind == 'gaussian':
            _, _, test_loader = make_loaders(data, NoiseConfig(), NoiseConfig())
            noise_override = NoiseConfig(gaussian_std=float(rate))
            sample_noise = rate > 0
        else:
            evaluation_noise = (NoiseConfig(channel_missing_rate=float(rate))
                                if kind == 'channel' else
                                NoiseConfig(temporal_thinning_rate=float(rate)))
            _, _, test_loader = make_loaders(data, NoiseConfig(),
                                             evaluation_noise,
                                             eval_noise_active=True)
            noise_override = NoiseConfig()
            sample_noise = False
        metrics, _, _ = evaluate_model(
            model, test_loader, noise_override, sample_noise=sample_noise,
            save_prefix=f'{prefix}_{kind}_{rate:g}')
        metrics.update({'dataset': args.dataset, 'backbone': args.backbone,
                        'variant': args.variant, 'seed': args.seed,
                        'experiment': args.experiment,
                        'degradation_type': kind, 'level': rate,
                        'train_gaussian_std': train_noise.gaussian_std,
                        'lambda_brier': lc, 'lambda_stability': ls,
                        'checkpoint': row['model_path'],
                        'max_epochs': CFG.max_epochs, 'patience': CFG.patience})
        rows.append(metrics)
    return pd.DataFrame(rows)


def run_selected_job(args):
    print('V2 selected job:', vars(args), flush=True)
    print('Expected output:', result_path_for(args), flush=True)
    if args.validate_job_only:
        print('VALIDATED: no data loaded, no training performed.', flush=True)
        return
    print('CUDA available:', torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0), flush=True)
    out_path = result_path_for(args)
    if out_path.exists():
        if args.skip_existing:
            print('Skipping existing result:', out_path, flush=True)
            return
        raise FileExistsError(f'Refusing to overwrite prior V2 result: {out_path}')
    if args.experiment == 'inspect':
        df = inspect_all_uci_datasets()
        print(df.to_string(), flush=True)
        return
    if args.experiment == 'synthetic':
        raise RuntimeError('V2 synthetic experiment not migrated; use legacy script read-only.')
    if args.experiment == 'finite_shot':
        raise RuntimeError('V2 finite-shot experiment not migrated; use legacy script read-only.')
    if args.experiment in ('main', 'ablation', 'noise', 'missingness'):
        # Only one dataset/backbone/variant/seed is ever trained by one job.
        set_seed(args.seed)
        data = prepare_temporal_dataset(args.dataset)
        if args.experiment in ('main', 'ablation'):
            row, _, _, _, _, _ = _train_selected_variant(args, data)
            row['experiment'] = args.experiment
            row['variant'] = args.variant
            row['max_epochs'] = CFG.max_epochs
            row['patience'] = CFG.patience
            df = pd.DataFrame([row])
        else:
            df = _run_selected_robustness(args, data)
        df.to_csv(out_path, index=False)
        print(f'Saved {len(df)} evaluation rows:', out_path, flush=True)
        return
    raise ValueError(f'Unknown experiment: {args.experiment}')


if __name__ == '__main__':
    run_selected_job(ARGS)
