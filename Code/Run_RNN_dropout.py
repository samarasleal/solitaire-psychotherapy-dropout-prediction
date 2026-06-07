#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, recall_score, precision_score, f1_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
import math
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# jupyter nbconvert --to script Run_RNN_dropout.ipynb


# ___

# Dataset and collate (Embeddings + tabular)

# In[ ]:


# --- GLOBAL CACHE FOR REUSE ACROSS DATASETS ---
_ARR_CACHE = {}  # key: (path, max_T) -> np.ndarray float32 (T x D)
class SeqNPYDataset(Dataset):
    def __init__(self, df, target_col, tab_cols=None, max_T=256,
                 preload_to_ram=False, verbose=True):
        self.paths   = df["Embedding_npy"].tolist()
        self.targets = df[target_col].values.astype("float32")
        self.tabs    = df[tab_cols].values.astype("float32") if tab_cols else None
        self.max_T   = int(max_T)
        self.preload = bool(preload_to_ram)
        self.verbose = verbose
        self.arrs = None
        if self.preload:
            self.arrs = []
            total_mb = 0.0
            new_count = 0
            for p in self.paths:
                key = (p, self.max_T)
                a = _ARR_CACHE.get(key)
                if a is None:
                    if self.verbose:
                        print(f"[load] {p}")
                    a = np.load(p, mmap_mode=None)   # no mmap
                    if a.shape[0] > self.max_T:
                        a = a[:self.max_T]
                    a = np.ascontiguousarray(a, dtype=np.float32)
                    if not np.isfinite(a).all():
                        print(f"[WARN][DATASET] Non-finite in preloaded array: {p}. Applying nan_to_num.")
                        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                    _ARR_CACHE[key] = a
                    total_mb += a.nbytes / (1024**2)
                    new_count += 1
                self.arrs.append(_ARR_CACHE[key])
            if self.verbose:
                print(f"Preloaded {len(self.arrs)} segments (~{total_mb:.1f} MB new,"
                      f" reused {len(self.arrs)-new_count} from cache).")
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        if self.preload and self.arrs is not None:
            arr = self.arrs[idx]
        else:
            p = self.paths[idx]
            key = (p, self.max_T)
            a = _ARR_CACHE.get(key)
            if a is None:
                a = np.load(p, mmap_mode=None)   # no mmap
                if a.shape[0] > self.max_T:
                    a = a[:self.max_T]
                a = np.ascontiguousarray(a, dtype=np.float32)
                if not np.isfinite(a).all():
                    print(f"[WARN][DATASET] Non-finite values in loaded array: {p}. Applying nan_to_num.")
                    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                _ARR_CACHE[key] = a
            arr = _ARR_CACHE[key]
        if not np.isfinite(arr).all():
            p = self.paths[idx]
            print(f"[WARN][DATASET] Non-finite in cached array: {p}. Fixing in-place.")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            _ARR_CACHE[(p, self.max_T)] = arr
        x = torch.from_numpy(arr)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.clamp(-8.0, 8.0)
        y = torch.tensor([self.targets[idx]], dtype=torch.float32)
        if self.tabs is not None:
            t = torch.from_numpy(self.tabs[idx])
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            return x, y, t
        return x, y


# In[3]:


def make_weighted_sampler_from_df(train_df, target_col):
    labels = train_df[target_col].to_numpy().astype(int)
    class_counts = np.bincount(labels, minlength=2)
    w0 = 1.0 / max(1, class_counts[0])
    w1 = 1.0 / max(1, class_counts[1])
    sample_weights = np.where(labels == 1, w1, w0)
    sample_weights = torch.from_numpy(sample_weights).double()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


# In[4]:


def get_io_loaders_balanced(train_ds, test_ds, train_df, target_col, BS):
    sampler = make_weighted_sampler_from_df(train_df, target_col)
    train_dl = DataLoader(train_ds, batch_size=BS, sampler=sampler, num_workers=0, pin_memory=True, persistent_workers=False)
    test_dl = DataLoader(test_ds, batch_size=max(1, BS//4), shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)
    return train_dl, test_dl


# ____

# GRU

# In[ ]:


class SafeGRURegressor(nn.Module):
    def __init__(self, emb_dim=768, hidden=64, layers=1, bidirectional=False, tab_dim=0, dropout=0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(emb_dim)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim + tab_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1))
    def forward(self, x, tab=None):
        # x: [B, T, D]
        # 1) clamp + nan_to_num (safety)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.clamp(-5.0, 5.0)
        # 2) layernorm for stability
        x = self.input_norm(x)
        # 3) GRU
        _, h = self.rnn(x)   # h: [layers*D, B, hidden]
        if self.rnn.bidirectional:
            h_last = torch.cat([h[-2], h[-1]], dim=1)
        else:
            h_last = h[-1]   # [B, hidden]
        # 4) safety on h_last
        h_last = torch.nan_to_num(h_last, nan=0.0, posinf=0.0, neginf=0.0)
        h_last = h_last.clamp(-10.0, 10.0)
        if tab is not None:
            feat = torch.cat([h_last, tab], dim=1)
        else:
            feat = h_last
        out = self.head(feat)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out


# LSTM

# In[ ]:


class SafeLSTMRegressor(nn.Module):
    def __init__(self, emb_dim=768, hidden=64, layers=1, bidirectional=False, tab_dim=0, dropout=0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(emb_dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim + tab_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1))
    def forward(self, x, tab=None):
        # x: [B, T, D]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.clamp(-5.0, 5.0)
        x = self.input_norm(x)
        _, (h, c) = self.rnn(x)  # h: [layers*D, B, hidden]
        if self.rnn.bidirectional:
            h_last = torch.cat([h[-2], h[-1]], dim=1)
        else:
            h_last = h[-1]  # [B, hidden]
        h_last = torch.nan_to_num(h_last, nan=0.0, posinf=0.0, neginf=0.0)
        h_last = h_last.clamp(-10.0, 10.0)
        if tab is not None:
            feat = torch.cat([h_last, tab], dim=1)
        else:
            feat = h_last
        out = self.head(feat)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out


# Tranformer Light Encoder

# In[7]:


class SafePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, T, d]
    def forward(self, x):
        # x: [B, T, d]
        return x + self.pe[:, : x.size(1), :]


# In[ ]:


class SafeTinyTransformerEncoderReg(nn.Module):
    def __init__(self, d_model=768, nhead=8, layers=1, dim_ff=1024, dropout=0.2, tab_dim=0):
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.pos = SafePositionalEncoding(d_model, max_len=300)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(d_model + tab_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1))
    def forward(self, x, tab=None):
        # safety on input
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.clamp(-5.0, 5.0)
        x = self.input_norm(x)
        z = self.encoder(self.pos(x))  # [B, T, d_model]
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z = z.clamp(-10.0, 10.0)
        pooled = z.mean(dim=1)
        if tab is not None:
            feat = torch.cat([pooled, tab], dim=1)
        else:
            feat = pooled
        out = self.head(feat)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out


# ____

# Train/Evaluate

# In[9]:


def train_one_epoch(model, dataloader, optimizer, criterion, device, has_tab=False):
    model.train()
    total_loss = 0.0
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for batch_idx, batch in enumerate(dataloader):
        if has_tab:
            xb, yb, tb = batch
        else:
            xb, yb = batch
            tb = None
        # CPU sanity
        if not torch.isfinite(xb).all():
            print(f"[FATAL][train] Non-finite xb RIGHT AFTER DATALOADER at batch {batch_idx}")
            xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(yb).all():
            print(f"[FATAL][train] Non-finite yb RIGHT AFTER DATALOADER at batch {batch_idx}")
            yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
        xb = xb.to(device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device, dtype=torch.float32, non_blocking=True)
        if tb is not None:
            tb = tb.to(device, dtype=torch.float32, non_blocking=True)
        # safety before forward
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)
        yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
        optimizer.zero_grad(set_to_none=True)
        # mixed precision forward
        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(xb, tb) if has_tab else model(xb)
            preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
            loss = criterion(preds, yb)
        if not torch.isfinite(loss):
            print(f"[WARN][train] loss non-finite at batch {batch_idx}")
            print("  xb stats:",
                  xb.min().item(), xb.max().item(),
                  xb.mean().item(), xb.std().item())
            print("  preds stats:",
                  preds.min().item(), preds.max().item(),
                  preds.mean().item(), preds.std().item())
            continue
        # backward with GradScaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        # quick progress log every 200 batches 
        if (batch_idx + 1) % 200 == 0:
            print(f"[train] batch {batch_idx+1}/{len(dataloader)} "
                  f"loss={loss.item():.4f}")
    return total_loss / max(1, len(dataloader))


# In[ ]:


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device, has_tab=False):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    use_amp = False
    for batch_idx, batch in enumerate(dataloader):
        if has_tab:
            xb, yb, tb = batch
        else:
            xb, yb = batch
            tb = None
        xb = xb.to(device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device, dtype=torch.float32, non_blocking=True)
        if tb is not None:
            tb = tb.to(device, dtype=torch.float32, non_blocking=True)
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)
        yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xb, tb) if has_tab else model(xb)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            loss   = criterion(logits, yb)
        if not torch.isfinite(loss):
            print("[WARN][eval] loss non-finite at batch", batch_idx)
            print("  xb stats:", xb.min().item(), xb.max().item(), xb.mean().item(), xb.std().item())
            print("  logits stats:", logits.min().item(), logits.max().item(), logits.mean().item(), logits.std().item())
            break
        total_loss += loss.item()
        y_true.append(yb.detach().cpu())

        probs = torch.sigmoid(logits.clamp(-10, 10)).float() 
        y_pred.append(probs.detach().cpu())

    if len(y_true) == 0:
        return total_loss / max(1, len(dataloader)), np.array([]), np.array([])
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return total_loss / max(1, len(dataloader)), y_true, y_pred


# ___

# In[ ]:


def class_lopo_RNN_dropout(patient, df, patient_col, target_col, model_name, tab_cols, BS, device="cuda", pos_weight=None, epochs=5, calibrate_tau="f2", warn_rule="2Session"):
    train_df = df[df[patient_col] != patient].copy()
    test_df  = df[df[patient_col] == patient].copy()

    last_sess = int(test_df["Session"].max())
    dropout_session = last_sess if last_sess < 8 else None  

    tab_cols = tab_cols or []
    has_tab  = bool(tab_cols)

    train_ds = SeqNPYDataset(train_df, target_col, tab_cols if has_tab else None, max_T=256, preload_to_ram=False, verbose=True)
    test_ds  = SeqNPYDataset(test_df,  target_col, tab_cols if has_tab else None, max_T=256, preload_to_ram=False, verbose=False)
    train_dl, test_dl = get_io_loaders_balanced(train_ds, test_ds, train_df, target_col, BS)
    tab_dim = len(tab_cols)
    if model_name.upper() == "GRU":
        model = SafeGRURegressor(tab_dim=tab_dim)
    elif model_name.upper() == "LSTM":
        model = SafeLSTMRegressor(tab_dim=tab_dim)
    else:
        model = SafeTinyTransformerEncoderReg(tab_dim=tab_dim)
    model.to(device)

    # If using balanced sampler, do NOT also use pos_weight (double compensation).
    use_sampler = True  # when using get_io_loaders_balanced

    if use_sampler:
        criterion = nn.BCEWithLogitsLoss()  # <-- plain BCE
    else:
        if pos_weight is None:
            pos = int((train_df[target_col] == 1).sum())
            neg = int((train_df[target_col] == 0).sum())
            pw_val = (neg / max(1, pos)) if (pos + neg) > 0 else 1.0
            pos_weight_t = torch.tensor([float(pw_val)], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
        else:
            pw = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    opt = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4,
        **({"fused": True} if (device == "cuda" and torch.cuda.is_available()) else {}))
    
    # train for a fixed number of epochs (avoid selecting best_state on train_dl)
    for _ in range(epochs):
        train_one_epoch(model, train_dl, opt, criterion, device, has_tab=has_tab)

    if calibrate_tau == "f2":
        # calibration loader (no sampler) to avoid threshold bias
        train_dl_calib = DataLoader(train_ds, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)
        _, y_true_tr, y_prob_tr = eval_epoch(model, train_dl_calib, criterion, device, has_tab=has_tab)
        y_true_tr = np.asarray(y_true_tr, dtype=int).reshape(-1)
        y_prob_tr = np.asarray(y_prob_tr, dtype=np.float32).reshape(-1)
        tau, _ = calibrate_tau_f2(y_true_tr, y_prob_tr, beta=2.0)
    elif calibrate_tau == "quant":
        train_dl_calib = DataLoader(train_ds, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)
        _, y_true_tr, y_prob_tr = eval_epoch(model, train_dl_calib, criterion, device, has_tab=has_tab)
        tau = calibrate_tau_quantile(y_prob_tr.reshape(-1), q=0.85)
    else:
        tau = 0.6

    # test
    _, y_true, y_prob = eval_epoch(model, test_dl, criterion, device, has_tab=has_tab)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)

    # --- patient-level aggregation (top-k mean) ---
    if y_prob.size:
        topk_frac = 0.2  
        k = max(1, int(np.ceil(topk_frac * len(y_prob))))
        p_patient = float(np.mean(np.partition(y_prob, -k)[-k:]))
    else:
        p_patient = float("nan")
    print(f"[DEBUG] tau={tau}, prob.min={np.nanmin(y_prob):.3f}, prob.max={np.nanmax(y_prob):.3f}")
    print(f"[DEBUG] frac sessions >= tau = {np.mean(np.isfinite(y_prob) & (y_prob >= tau)):.2f}")  
        
    # true patient label (majority over sessions - dropout; fallback 0)
    y_patient_true = int(np.max(y_true)) if y_true.size else 0
    # patient decision 
    if warn_rule == "2Session":
        above = np.sum(y_prob >= tau)
        y_patient_pred = int(above >= 2) # 2 sessions above the threshold
    elif warn_rule == "Consec":
        y_patient_pred = int(has_k_consecutive_above(y_prob, tau, k=2))
    elif warn_rule == "30perc":
        y_patient_pred = int(np.mean(np.asarray(y_prob) >= tau) >= 0.30)  # ex.: 30% of sessions
    else:
        # pro-recall threshold
        y_patient_pred = int(p_patient >= tau) if np.isfinite(p_patient) else 0

    # patient-level recall (NaN for negatives → recall defined on positive class)
    rec_patient = float(1.0 if (y_patient_true == 1 and y_patient_pred == 1) else 0.0) if y_patient_true == 1 else float("nan")
    spec_patient = float(1.0 if (y_patient_true == 0 and y_patient_pred == 0) else 0.0) if y_patient_true == 0 else float("nan")

    # Precision and F1 (positive class) 
    if y_patient_pred == 1 and y_patient_true == 1:
        prec_patient = 1.0
        f1_patient   = 1.0
    elif y_patient_pred == 1 and y_patient_true == 0:
        prec_patient = 0.0
        f1_patient   = 0.0
    elif y_patient_pred == 0 and y_patient_true == 1:
        prec_patient = float("nan")  # undefined precision (0 positive predicted)
        f1_patient   = 0.0
    else:  # y_pred=0, y_true=0
        prec_patient = float("nan")  # undefined precision (0 positive predicted)
        f1_patient   = float("nan")

    sessions = test_df["Session"].astype(int).tolist()
    return {
        "Model": model_name.upper(),
        "Patient_ID": patient,
        "Session": sessions,
        "REC": rec_patient,          # sensitivity (NaN for negatives)
        "SPEC": spec_patient,        # specificity (NaN for positives) 
        "PREC": prec_patient,
        "F1": f1_patient,
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
        "p_patient": p_patient,
        "y_patient_true": y_patient_true,
        "y_patient_pred": y_patient_pred,
        "tau": float(tau),
        "dropout_session": dropout_session}


# In[12]:


def fbeta_from_probs(y_true, y_prob, beta=2.0, tau=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= tau).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    return (1+b2) * prec * rec / (b2 * prec + rec) if (prec + rec) else 0.0


# In[13]:


def calibrate_tau_f2(y_true_train, y_prob_train, beta=2.0):
    y_true_train = np.asarray(y_true_train).astype(int)
    y_prob_train = np.asarray(y_prob_train).astype(float)

    # 1) guard: no positives -> fallback
    if y_true_train.sum() == 0:
        return 0.5, 0.0

    taus = np.linspace(0.05, 0.95, 19)
    scores = [fbeta_from_probs(y_true_train, y_prob_train, beta=beta, tau=t) for t in taus]
    best = np.max(scores)

    # 2) tie-break: choose *smallest* tau among ties (favor recall)
    best_taus = taus[np.where(np.isclose(scores, best))[0]]
    best_tau = float(np.min(best_taus))
    return best_tau, float(best)


# In[ ]:


def calibrate_tau_quantile(y_prob_train, q=0.85):
    y_prob_train = np.asarray(y_prob_train, dtype=float)
    y_prob_train = y_prob_train[np.isfinite(y_prob_train)]
    if y_prob_train.size == 0:
        return 0.6
    return float(np.quantile(y_prob_train, q))


# In[ ]:


def has_k_consecutive_above(y_prob, tau, k=2):
    flags = (np.asarray(y_prob) >= tau).astype(int)
    run = 0
    for f in flags:
        run = run + 1 if f else 0
        if run >= k:
            return True
    return False

