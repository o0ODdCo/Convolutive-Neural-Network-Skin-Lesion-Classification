#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    _CUPY_AVAILABLE = False


# =====================================================================
# 0. BACKEND NUMERIQUE : NumPy CPU ou CuPy GPU
# =====================================================================

xp = np
_BACKEND_NAME = 'numpy'
_IM2COL_INDEX_CACHE = {}


def _cupy_has_usable_device():
    if not _CUPY_AVAILABLE:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def set_backend(name='auto', seed=None):
    """Selectionne le backend de calcul sans bibliotheque de deep learning.

    name peut valoir 'auto', 'cupy'/'cuda'/'gpu' ou 'cpu'/'numpy'.
    En mode 'auto', CuPy est utilise si un GPU CUDA est accessible ; sinon
    le programme reste en NumPy CPU.
    """
    global xp, _BACKEND_NAME, _IM2COL_INDEX_CACHE
    requested = str(name or 'auto').lower()
    want_gpu = requested in {'cupy', 'cuda', 'gpu'}
    want_cpu = requested in {'cpu', 'numpy'}

    if want_cpu:
        xp = np
        _BACKEND_NAME = 'numpy'
    else:
        usable = _cupy_has_usable_device()
        if usable:
            xp = cp
            _BACKEND_NAME = 'cupy'
            cp.cuda.Device().use()
            if seed is not None:
                cp.random.seed(int(seed))
        elif want_gpu:
            raise RuntimeError(
                "Backend GPU demande, mais CuPy ou un peripherique CUDA utilisable est indisponible. "
                "Installez une version de cupy adaptee a votre CUDA, ou utilisez --backend cpu."
            )
        else:
            xp = np
            _BACKEND_NAME = 'numpy'

    _IM2COL_INDEX_CACHE = {}
    return _BACKEND_NAME


def using_gpu():
    return _BACKEND_NAME == 'cupy'


def backend_info():
    if not using_gpu():
        return 'numpy/cpu'
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props.get('name', b'GPU')
        if isinstance(name, bytes):
            name = name.decode('utf-8', errors='replace')
        return f'cupy/cuda:{dev.id} ({name})'
    except Exception:
        return 'cupy/cuda'


def to_device(a, dtype=None):
    if using_gpu():
        return cp.asarray(a, dtype=dtype)
    return np.asarray(a, dtype=dtype)


def to_cpu(a):
    if _CUPY_AVAILABLE and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return a


def scalar_to_float(a):
    if _CUPY_AVAILABLE and isinstance(a, cp.ndarray):
        return float(a.get())
    try:
        return float(a.item())
    except AttributeError:
        return float(a)


def state_to_cpu(obj):
    if isinstance(obj, dict):
        return {k: state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(state_to_cpu(v) for v in obj)
    if _CUPY_AVAILABLE and isinstance(obj, cp.ndarray):
        return cp.asnumpy(obj).copy()
    if isinstance(obj, np.ndarray):
        return obj.copy()
    return obj


def synchronize_backend():
    if using_gpu():
        cp.cuda.Stream.null.synchronize()


# =====================================================================
# 1. CONSTANTES
# =====================================================================

CLASSES_HAM10000 = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_NAMES_FR = {
    'akiec': 'Keratoses actiniques',
    'bcc':   'Carcinome basocellulaire',
    'bkl':   'Keratoses benignes',
    'df':    'Dermatofibrome',
    'mel':   'Melanome',
    'nv':    'Naevi melanocytaires',
    'vasc':  'Lesions vasculaires',
}

NORM_STATS = {
    'imagenet': (
        np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1),
        np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1),
    ),
    'half': (
        np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1),
        np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1),
    ),
    'none': (
        np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(3, 1, 1),
        np.array([1.0, 1.0, 1.0], dtype=np.float32).reshape(3, 1, 1),
    ),
}


# =====================================================================
# 2. AFFICHAGE CONSOLE
# =====================================================================

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def _enable_ansi_on_windows():
    if os.name == 'nt':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


_TTY = sys.stdout.isatty()
if _TTY:
    _enable_ansi_on_windows()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')


class C:
    if _TTY:
        RESET = '\033[0m'; BOLD = '\033[1m'; DIM = '\033[2m'
        RED = '\033[31m'; GREEN = '\033[32m'; YELLOW = '\033[33m'
        BLUE = '\033[34m'; CYAN = '\033[36m'
        GRAY = '\033[90m'
    else:
        RESET = BOLD = DIM = RED = GREEN = YELLOW = BLUE = CYAN = GRAY = ''


def visible_len(s):
    return len(_ANSI_RE.sub('', str(s)))


def banner(title, width=88, color=None):
    color = color or C.CYAN
    bar = '=' * width
    pad = max(0, (width - visible_len(title)) // 2)
    print(f'\n{color}{bar}{C.RESET}')
    print(f'{color}{" " * pad}{C.BOLD}{title}{C.RESET}')
    print(f'{color}{bar}{C.RESET}')


def section(title, width=88):
    line = '-' * max(1, width - visible_len(title) - 5)
    print(f'\n{C.BOLD}{C.BLUE}-- {title} {line}{C.RESET}')


def fmt_table(headers, rows, align=None):
    if align is None:
        align = ['<'] * len(headers)
    str_rows = [[str(c) for c in r] for r in rows]
    widths = []
    for i, h in enumerate(headers):
        cells = [str(h)] + [r[i] for r in str_rows]
        widths.append(max(visible_len(c) for c in cells))

    def pad(s, w, a):
        diff = w - visible_len(s)
        if a == '<':
            return s + ' ' * diff
        if a == '>':
            return ' ' * diff + s
        return ' ' * (diff // 2) + s + ' ' * (diff - diff // 2)

    def line(cells):
        return '|' + '|'.join(f' {pad(c, w, a)} ' for c, a, w in zip(cells, align, widths)) + '|'

    bar = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    out = [bar, line(headers), bar]
    out.extend(line(r) for r in str_rows)
    out.append(bar)
    return '\n'.join(out)


class ProgressBar:
    def __init__(self, total, prefix='', width=24, min_interval=0.15):
        self.total = max(int(total), 1)
        self.prefix = prefix
        self.width = width
        self.min_interval = min_interval
        self.t0 = time.time()
        self._last = 0.0

    def update(self, current, **fields):
        now = time.time()
        is_last = current >= self.total
        if not is_last and now - self._last < self.min_interval:
            return
        self._last = now
        elapsed = now - self.t0
        pct = min(max(current / self.total, 0.0), 1.0)
        filled = int(self.width * pct)
        bar = '#' * filled + '.' * (self.width - filled)
        rate = current / max(elapsed, 1e-9)
        eta = (self.total - current) / max(rate, 1e-9)
        extras = '  '.join(f'{C.DIM}{k}{C.RESET}={v}' for k, v in fields.items())
        msg = (f'\r\033[K{C.CYAN}{self.prefix}{C.RESET} '
               f'[{C.GREEN}{bar}{C.RESET}] {current:4d}/{self.total}  {extras}  '
               f'{C.GRAY}{rate:5.1f} it/s | ETA {eta:5.1f}s{C.RESET}')
        sys.stdout.write(msg)
        sys.stdout.flush()
        if is_last:
            sys.stdout.write('\n')


def fmt_prob_bars(labels, probs, width=24):
    lines = []
    for label, p in zip(labels, probs):
        p = float(p)
        filled = int(round(width * p))
        bar = '#' * filled + '.' * (width - filled)
        color = C.GREEN if p > 0.5 else (C.YELLOW if p > 0.2 else C.GRAY)
        lines.append(f'  {label:6s}  {color}{bar}{C.RESET}  {p * 100:5.2f}%')
    return '\n'.join(lines)


def temps_humain(secondes):
    secondes = int(secondes)
    h = secondes // 3600
    m = (secondes % 3600) // 60
    s = secondes % 60
    if h:
        return f'{h:d}h {m:02d}m {s:02d}s'
    if m:
        return f'{m:d}m {s:02d}s'
    return f'{s:d}s'


# =====================================================================
# 3. METRIQUES
# =====================================================================

def per_class_metrics(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.float64)
    support_total = float(max(support.sum(), 1.0))

    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)

    micro_tp = float(tp.sum())
    micro_fp = float(fp.sum())
    micro_fn = float(fn.sum())
    micro_precision = micro_tp / max(micro_tp + micro_fp, 1e-12)
    micro_recall = micro_tp / max(micro_tp + micro_fn, 1e-12)
    micro_f1 = (2.0 * micro_precision * micro_recall /
                max(micro_precision + micro_recall, 1e-12))

    weighted_precision = float((prec * support).sum() / support_total)
    weighted_recall = float((rec * support).sum() / support_total)
    weighted_f1 = float((f1 * support).sum() / support_total)
    accuracy = float(tp.sum() / support_total)

    return {
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'support': support.astype(np.int64),
        'macro_precision': float(prec.mean()),
        'macro_recall': float(rec.mean()),
        'macro_f1': float(f1.mean()),
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'micro_precision': float(micro_precision),
        'micro_recall': float(micro_recall),
        'micro_f1': float(micro_f1),
        'balanced_accuracy': float(rec.mean()),
        'accuracy': accuracy,
        'error_rate': float(1.0 - accuracy),
    }


def color_metric(v):
    return C.GREEN if v > 0.7 else (C.YELLOW if v > 0.4 else C.RED)


def fmt_classification_report(cm, classes):
    m = per_class_metrics(cm)
    total_support = int(m['support'].sum())
    rows = []
    for i, c in enumerate(classes):
        rows.append([
            c,
            f'{color_metric(m["precision"][i])}{m["precision"][i]:.3f}{C.RESET}',
            f'{color_metric(m["recall"][i])}{m["recall"][i]:.3f}{C.RESET}',
            f'{color_metric(m["f1"][i])}{m["f1"][i]:.3f}{C.RESET}',
            str(int(m['support'][i])),
        ])

    rows.append(['------', '------', '------', '------', '------'])
    rows.append([
        f'{C.BOLD}macro{C.RESET}',
        f'{C.BOLD}{m["macro_precision"]:.3f}{C.RESET}',
        f'{C.BOLD}{m["macro_recall"]:.3f}{C.RESET}',
        f'{C.BOLD}{m["macro_f1"]:.3f}{C.RESET}',
        f'{C.BOLD}{total_support}{C.RESET}',
    ])
    rows.append([
        f'{C.BOLD}weighted{C.RESET}',
        f'{C.BOLD}{m["weighted_precision"]:.3f}{C.RESET}',
        f'{C.BOLD}{m["weighted_recall"]:.3f}{C.RESET}',
        f'{C.BOLD}{m["weighted_f1"]:.3f}{C.RESET}',
        f'{C.BOLD}{total_support}{C.RESET}',
    ])
    rows.append([
        f'{C.BOLD}micro{C.RESET}',
        f'{C.BOLD}{m["micro_precision"]:.3f}{C.RESET}',
        f'{C.BOLD}{m["micro_recall"]:.3f}{C.RESET}',
        f'{C.BOLD}{m["micro_f1"]:.3f}{C.RESET}',
        f'{C.BOLD}{total_support}{C.RESET}',
    ])
    return fmt_table(['classe', 'precision', 'recall', 'f1', 'support'], rows,
                     align=['<', '>', '>', '>', '>'])


def fmt_global_metrics(cm):
    m = per_class_metrics(cm)
    rows = [
        ['accuracy', f'{m["accuracy"]:.4f}'],
        ['balanced_accuracy', f'{m["balanced_accuracy"]:.4f}'],
        ['error_rate', f'{m["error_rate"]:.4f}'],
        ['macro_precision', f'{m["macro_precision"]:.4f}'],
        ['macro_recall', f'{m["macro_recall"]:.4f}'],
        ['macro_f1', f'{m["macro_f1"]:.4f}'],
        ['weighted_precision', f'{m["weighted_precision"]:.4f}'],
        ['weighted_recall', f'{m["weighted_recall"]:.4f}'],
        ['weighted_f1', f'{m["weighted_f1"]:.4f}'],
        ['micro_precision', f'{m["micro_precision"]:.4f}'],
        ['micro_recall', f'{m["micro_recall"]:.4f}'],
        ['micro_f1', f'{m["micro_f1"]:.4f}'],
    ]
    return fmt_table(['metrique globale', 'valeur'], rows, align=['<', '>'])

def fmt_confusion_matrix(cm, classes):
    headers = ['true \\ pred'] + list(classes)
    rows = []
    for i, c in enumerate(classes):
        row = [c]
        for j in range(len(classes)):
            v = int(cm[i, j])
            if v == 0:
                cell = f'{C.DIM}.{C.RESET}'
            elif i == j:
                cell = f'{C.GREEN}{C.BOLD}{v}{C.RESET}'
            else:
                cell = str(v)
            row.append(cell)
        rows.append(row)
    return fmt_table(headers, rows, align=['<'] + ['>'] * len(classes))


EPOCH_METRIC_KEYS = (
    'accuracy',
    'error_rate',
    'macro_precision',
    'macro_recall',
    'macro_f1',
    'weighted_precision',
    'weighted_recall',
    'weighted_f1',
    'micro_precision',
    'micro_recall',
    'micro_f1',
    'balanced_accuracy',
)


BATCH_PER_CLASS_KEYS = ('precision', 'recall', 'f1', 'support')


def csv_scalar(v):
    """Convertit proprement les valeurs numeriques vers un format CSV stable."""
    if v is None:
        return ''
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return '' if math.isnan(v) or math.isinf(v) else f'{v:.10g}'
    return v


def flatten_metrics(prefix, metrics, classes):
    """Aplati toutes les metriques globales et par classe pour une ligne CSV."""
    row = {}
    for key in EPOCH_METRIC_KEYS:
        row[f'{prefix}_{key}'] = csv_scalar(metrics[key])
    for key in BATCH_PER_CLASS_KEYS:
        values = metrics[key]
        for i, c in enumerate(classes):
            name = f'{prefix}_{key}_{c}'
            row[name] = int(values[i]) if key == 'support' else csv_scalar(values[i])
    return row


def batch_metric_fieldnames(classes):
    base = [
        'phase',
        'epoch',
        'batch',
        'num_batches',
        'global_batch',
        'samples_batch',
        'samples_seen_epoch',
        'samples_seen_total',
        'loss',
        'loss_mean_epoch',
        'grad_norm',
        'elapsed_epoch_s',
        'elapsed_total_s',
    ]
    metric_fields = []
    for prefix in ('batch', 'running'):
        metric_fields.extend(f'{prefix}_{key}' for key in EPOCH_METRIC_KEYS)
        for key in BATCH_PER_CLASS_KEYS:
            metric_fields.extend(f'{prefix}_{key}_{c}' for c in classes)
    return base + metric_fields


class BatchMetricLogger:
    """Journal CSV des metriques a chaque batch.

    Chaque ligne contient les metriques du batch courant et les metriques
    cumulees depuis le debut de l'epoch pour la phase concernee.
    """
    def __init__(self, path, classes, append=False, enabled=True):
        self.path = path
        self.classes = list(classes)
        self.enabled = bool(enabled and path)
        self._file = None
        self._writer = None
        self._fieldnames = batch_metric_fieldnames(self.classes)
        if not self.enabled:
            return

        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        mode = 'a' if append and file_exists else 'w'
        self._file = open(path, mode, newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames, extrasaction='ignore')
        if mode == 'w':
            self._writer.writeheader()
            self._file.flush()

    def log(self, phase, epoch, batch, num_batches, global_batch,
            samples_batch, samples_seen_epoch, samples_seen_total,
            loss, loss_mean_epoch, grad_norm,
            batch_metrics, running_metrics,
            elapsed_epoch_s, elapsed_total_s):
        if not self.enabled:
            return
        row = {
            'phase': phase,
            'epoch': int(epoch),
            'batch': int(batch),
            'num_batches': int(num_batches),
            'global_batch': int(global_batch),
            'samples_batch': int(samples_batch),
            'samples_seen_epoch': int(samples_seen_epoch),
            'samples_seen_total': int(samples_seen_total),
            'loss': csv_scalar(loss),
            'loss_mean_epoch': csv_scalar(loss_mean_epoch),
            'grad_norm': csv_scalar(grad_norm),
            'elapsed_epoch_s': csv_scalar(elapsed_epoch_s),
            'elapsed_total_s': csv_scalar(elapsed_total_s),
        }
        row.update(flatten_metrics('batch', batch_metrics, self.classes))
        row.update(flatten_metrics('running', running_metrics, self.classes))
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def fmt_epoch_metrics_table(train_loss, train_metrics, val_loss, val_metrics):
    rows = [['loss', f'{train_loss:.4f}', f'{val_loss:.4f}', f'{val_loss - train_loss:+.4f}']]
    for key in EPOCH_METRIC_KEYS:
        rows.append([
            key,
            f'{train_metrics[key]:.4f}',
            f'{val_metrics[key]:.4f}',
            f'{val_metrics[key] - train_metrics[key]:+.4f}',
        ])
    return fmt_table(['metrique', 'train', 'validation', 'val-train'], rows,
                     align=['<', '>', '>', '>'])

def _recent_slope(values, window=4):
    vals = np.asarray(values[-window:], dtype=np.float64)
    if vals.size < 2:
        return 0.0
    x = np.arange(vals.size, dtype=np.float64)
    x -= x.mean()
    denom = float(np.sum(x * x))
    if denom <= 0:
        return 0.0
    return float(np.sum(x * (vals - vals.mean())) / denom)


def _scale_channels(channels, factor):
    out = []
    for c in channels:
        v = max(4, int(round(float(c) * factor / 4.0)) * 4)
        out.append(v)
    return ','.join(str(v) for v in out)


def diagnose_fit_state(history, train_loss, train_metrics, val_loss, val_metrics, args):
    train_f1 = float(train_metrics['macro_f1'])
    val_f1 = float(val_metrics['macro_f1'])
    train_acc = float(train_metrics['accuracy'])
    val_acc = float(val_metrics['accuracy'])
    gap_f1 = train_f1 - val_f1
    gap_acc = train_acc - val_acc
    gap_loss = val_loss - train_loss
    slope_f1 = _recent_slope(history.get('val_macro_f1', []), window=4)

    channels = tuple(int(x) for x in str(args.channels).split(','))

    if (gap_f1 >= 0.10 and gap_loss >= 0.10) or (gap_acc >= 0.12 and train_f1 >= 0.45):
        state = 'sur-apprentissage probable'
        recs = [
            f'augmenter --dropout vers {min(args.dropout + 0.10, 0.70):.2f}',
            f'augmenter --weight-decay vers {max(args.weight_decay * 2.0, 1e-6):.2e}',
            'utiliser --augment-strength strong ou augmenter --max-augment-repeat',
            f'reduire --channels vers {_scale_channels(channels, 0.75)} si le sur-apprentissage persiste',
            'arreter plus tot si val_macro_f1 ne progresse plus',
        ]
    elif train_f1 < 0.45 and val_f1 < 0.45 and abs(gap_f1) < 0.08:
        state = 'sous-apprentissage probable'
        recs = [
            f'augmenter --channels vers {_scale_channels(channels, 1.25)}',
            f'diminuer --dropout vers {max(args.dropout - 0.10, 0.0):.2f}',
            f'diminuer --weight-decay vers {args.weight_decay / 2.0:.2e}',
            f'essayer --lr autour de {args.lr * 1.5:.2e} si la loss baisse trop lentement',
            f'augmenter --epochs vers {args.epochs + max(5, args.epochs // 3)}',
        ]
    elif slope_f1 < -0.015:
        state = 'instabilite ou degradation recente'
        recs = [
            f'diminuer --lr vers {args.lr / 2.0:.2e}',
            'conserver le checkpoint best.pkl car il est choisi sur val_macro_f1',
            'verifier que les augmentations ne sont pas trop fortes pour les petites lesions',
        ]
    else:
        state = 'apprentissage globalement equilibre'
        recs = [
            'continuer sans changer les hyperparametres principaux',
            'surveiller val_macro_f1 et le rappel des classes minoritaires',
        ]

    return {
        'state': state,
        'gap_f1': float(gap_f1),
        'gap_acc': float(gap_acc),
        'gap_loss': float(gap_loss),
        'slope_val_macro_f1': float(slope_f1),
        'recommendations': recs,
    }


def fmt_fit_diagnostic(diag):
    rows = [
        ['etat', diag['state']],
        ['ecart macro-F1 train-val', f'{diag["gap_f1"]:+.4f}'],
        ['ecart accuracy train-val', f'{diag["gap_acc"]:+.4f}'],
        ['ecart loss val-train', f'{diag["gap_loss"]:+.4f}'],
        ['pente recente val_macro_f1', f'{diag["slope_val_macro_f1"]:+.4f}'],
    ]
    for i, r in enumerate(diag['recommendations'], start=1):
        rows.append([f'reglage {i}', r])
    return fmt_table(['diagnostic', 'valeur / action'], rows, align=['<', '<'])


def make_empty_history():
    history = {
        'train_loss': [],
        'val_loss': [],
        'grad_norm': [],
        'epoch_time': [],
        'fit_state': [],
    }
    for prefix in ('train', 'val'):
        for key in EPOCH_METRIC_KEYS:
            history[f'{prefix}_{key}'] = []

    # Alias historiques conserves pour compatibilite avec les anciens checkpoints.
    history['train_acc'] = history['train_accuracy']
    history['val_acc'] = history['val_accuracy']
    return history


def history_append(history, key, value):
    history.setdefault(key, []).append(value)


def history_append_metrics(history, prefix, metrics):
    for key in EPOCH_METRIC_KEYS:
        history_append(history, f'{prefix}_{key}', metrics[key])
    if prefix in {'train', 'val'}:
        history[f'{prefix}_acc'] = history[f'{prefix}_accuracy']


# =====================================================================
# 4. IM2COL / COL2IM
# =====================================================================

def _im2col_indices(C_, H, W, KH, KW, padding, stride):
    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1
    key = (_BACKEND_NAME, int(C_), int(H), int(W), int(KH), int(KW), int(padding), int(stride))
    cached = _IM2COL_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    i0 = xp.repeat(xp.arange(KH), KW)
    i0 = xp.tile(i0, C_)
    i1 = stride * xp.repeat(xp.arange(H_out), W_out)
    j0 = xp.tile(xp.arange(KW), KH * C_)
    j1 = stride * xp.tile(xp.arange(W_out), H_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = xp.repeat(xp.arange(C_), KH * KW).reshape(-1, 1)
    out = (k, i, j, H_out, W_out)
    _IM2COL_INDEX_CACHE[key] = out
    return out


def im2col(x, KH, KW, padding, stride):
    N, C_, H, W = x.shape
    if padding > 0:
        x_padded = xp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
        x_padded = x
    k, i, j, H_out, W_out = _im2col_indices(C_, H, W, KH, KW, padding, stride)
    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(C_ * KH * KW, -1)
    return cols, H_out, W_out


def col2im(cols, x_shape, KH, KW, padding, stride):
    N, C_, H, W = x_shape
    H_p, W_p = H + 2 * padding, W + 2 * padding
    x_padded = xp.zeros((N, C_, H_p, W_p), dtype=cols.dtype)
    k, i, j, H_out, W_out = _im2col_indices(C_, H, W, KH, KW, padding, stride)
    cols_re = cols.reshape(C_ * KH * KW, -1, N).transpose(2, 0, 1)
    xp.add.at(x_padded, (slice(None), k, i, j), cols_re)
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


# =====================================================================
# 5. COUCHES
# =====================================================================

def he_init(shape, fan_in):
    return (xp.random.randn(*shape) * np.sqrt(2.0 / fan_in)).astype(xp.float32)


def glorot_init(shape, fan_in, fan_out):
    return (xp.random.randn(*shape) * np.sqrt(2.0 / (fan_in + fan_out))).astype(xp.float32)


class Layer:
    def forward(self, x, training=True):
        return x

    def backward(self, dout):
        return dout

    def params(self):
        return {}

    def grads(self):
        return {}

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


class Conv2D(Layer):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        self.Cin, self.Cout = in_ch, out_ch
        self.KH = self.KW = k
        self.stride, self.padding = s, p
        self.W = he_init((out_ch, in_ch, k, k), in_ch * k * k)
        self.b = xp.zeros(out_ch, dtype=xp.float32)
        self.dW = xp.zeros_like(self.W)
        self.db = xp.zeros_like(self.b)
        self._cache = None

    def forward(self, x, training=True):
        cols, H_out, W_out = im2col(x, self.KH, self.KW, self.padding, self.stride)
        out = self.W.reshape(self.Cout, -1) @ cols + self.b.reshape(-1, 1)
        N = x.shape[0]
        out = out.reshape(self.Cout, H_out, W_out, N).transpose(3, 0, 1, 2)
        if training:
            self._cache = (x.shape, cols)
        return out

    def backward(self, dout):
        x_shape, cols = self._cache
        dout_col = dout.transpose(1, 2, 3, 0).reshape(self.Cout, -1)
        self.db[...] = dout_col.sum(axis=1)
        self.dW[...] = (dout_col @ cols.T).reshape(self.W.shape)
        dcols = self.W.reshape(self.Cout, -1).T @ dout_col
        return col2im(dcols, x_shape, self.KH, self.KW, self.padding, self.stride)

    def params(self):
        return {'W': self.W, 'b': self.b}

    def grads(self):
        return {'W': self.dW, 'b': self.db}



class BatchNorm2D(Layer):
    """Normalisation de batch pour tenseurs image NCHW.

    Pendant l'entrainement, la moyenne et la variance sont calculees pour
    chaque canal sur les axes (batch, hauteur, largeur). Pendant l'inference,
    on utilise les moyennes mobiles running_mean et running_var.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)

        self.gamma = xp.ones((1, self.num_features, 1, 1), dtype=xp.float32)
        self.beta = xp.zeros((1, self.num_features, 1, 1), dtype=xp.float32)
        self.dgamma = xp.zeros_like(self.gamma)
        self.dbeta = xp.zeros_like(self.beta)

        self.running_mean = xp.zeros((1, self.num_features, 1, 1), dtype=xp.float32)
        self.running_var = xp.ones((1, self.num_features, 1, 1), dtype=xp.float32)

        self._cache = None

    def forward(self, x, training=True):
        axes = (0, 2, 3)
        if training:
            mean = x.mean(axis=axes, keepdims=True)
            var = x.var(axis=axes, keepdims=True)
            x_centered = x - mean
            inv_std = 1.0 / xp.sqrt(var + self.eps)
            x_hat = x_centered * inv_std

            self.running_mean = (
                self.momentum * self.running_mean
                + (1.0 - self.momentum) * mean
            ).astype(np.float32, copy=False)
            self.running_var = (
                self.momentum * self.running_var
                + (1.0 - self.momentum) * var
            ).astype(np.float32, copy=False)

            self._cache = (x_hat, x_centered, inv_std, var)
        else:
            x_hat = (x - self.running_mean) / xp.sqrt(self.running_var + self.eps)

        return (self.gamma * x_hat + self.beta).astype(xp.float32, copy=False)

    def backward(self, dout):
        x_hat, x_centered, inv_std, var = self._cache
        N, C, H, W = dout.shape
        M = N * H * W

        self.dgamma[...] = xp.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)
        self.dbeta[...] = xp.sum(dout, axis=(0, 2, 3), keepdims=True)

        dxhat = dout * self.gamma
        dvar = xp.sum(
            dxhat * x_centered * (-0.5) * (var + self.eps) ** (-1.5),
            axis=(0, 2, 3),
            keepdims=True,
        )
        dmean = (
            xp.sum(dxhat * (-inv_std), axis=(0, 2, 3), keepdims=True)
            + dvar * xp.mean(-2.0 * x_centered, axis=(0, 2, 3), keepdims=True)
        )
        dx = dxhat * inv_std + dvar * 2.0 * x_centered / M + dmean / M
        return dx.astype(xp.float32, copy=False)

    def params(self):
        return {'gamma': self.gamma, 'beta': self.beta}

    def grads(self):
        return {'gamma': self.dgamma, 'beta': self.dbeta}

    def state_dict(self):
        return {
            'running_mean': self.running_mean.copy(),
            'running_var': self.running_var.copy(),
        }

    def load_state_dict(self, state):
        self.running_mean[...] = to_device(state['running_mean'], dtype=xp.float32)
        self.running_var[...] = to_device(state['running_var'], dtype=xp.float32)


class MaxPool2D(Layer):
    def __init__(self, k=2, s=2):
        self.K, self.stride = k, s
        self._cache = None

    def forward(self, x, training=True):
        N, C_, H, W = x.shape
        K, s = self.K, self.stride
        H_out, W_out = (H - K) // s + 1, (W - K) // s + 1
        cols, _, _ = im2col(x.reshape(N * C_, 1, H, W), K, K, 0, s)
        max_idx = xp.argmax(cols, axis=0)
        out_col = cols[max_idx, xp.arange(cols.shape[1])]
        out = out_col.reshape(H_out, W_out, N, C_).transpose(2, 3, 0, 1)
        if training:
            self._cache = (x.shape, cols.shape, max_idx)
        return out

    def backward(self, dout):
        x_shape, cols_shape, max_idx = self._cache
        N, C_, H, W = x_shape
        K, s = self.K, self.stride
        dout_col = dout.transpose(2, 3, 0, 1).reshape(-1)
        dcols = xp.zeros(cols_shape, dtype=dout.dtype)
        dcols[max_idx, xp.arange(dcols.shape[1])] = dout_col
        return col2im(dcols, (N * C_, 1, H, W), K, K, 0, s).reshape(N, C_, H, W)


class ReLU(Layer):
    def __init__(self):
        self._mask = None

    def forward(self, x, training=True):
        m = x > 0
        if training:
            self._mask = m
        return x * m

    def backward(self, dout):
        return dout * self._mask


class GlobalAveragePool2D(Layer):
    def __init__(self):
        self._shape = None

    def forward(self, x, training=True):
        if training:
            self._shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, dout):
        N, C, H, W = self._shape
        return (xp.ones((N, C, H, W), dtype=dout.dtype)
                * dout[:, :, None, None] / float(H * W))


class Dense(Layer):
    def __init__(self, in_dim, out_dim):
        self.W = glorot_init((in_dim, out_dim), in_dim, out_dim)
        self.b = xp.zeros(out_dim, dtype=xp.float32)
        self.dW = xp.zeros_like(self.W)
        self.db = xp.zeros_like(self.b)
        self._x = None

    def forward(self, x, training=True):
        if training:
            self._x = x
        return x @ self.W + self.b

    def backward(self, dout):
        x = self._x
        self.dW[...] = x.T @ dout
        self.db[...] = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self):
        return {'W': self.W, 'b': self.b}

    def grads(self):
        return {'W': self.dW, 'b': self.db}


class Dropout(Layer):
    def __init__(self, rate=0.5):
        self.rate, self._mask = rate, None

    def forward(self, x, training=True):
        if training and self.rate > 0:
            self._mask = (xp.random.rand(*x.shape) > self.rate).astype(x.dtype) / (1.0 - self.rate)
            return x * self._mask
        self._mask = None
        return x

    def backward(self, dout):
        return dout if self._mask is None else dout * self._mask


class SoftmaxCrossEntropy:
    def __init__(self, class_weights=None):
        self.class_weights = None if class_weights is None else to_device(class_weights, dtype=xp.float32)
        self._cache = None

    def forward(self, logits, y):
        z = logits - logits.max(axis=1, keepdims=True)
        ez = xp.exp(z)
        probs = ez / ez.sum(axis=1, keepdims=True)
        N = logits.shape[0]
        rows = xp.arange(N)
        log_p = xp.log(probs[rows, y] + 1e-12)
        if self.class_weights is not None:
            w = self.class_weights[y]
            denom = max(scalar_to_float(w.sum()), 1e-12)
            loss = -scalar_to_float((log_p * w).sum() / denom)
            self._cache = (probs, y, w)
        else:
            loss = -scalar_to_float(log_p.mean())
            self._cache = (probs, y, None)
        return loss, probs

    def backward(self):
        probs, y, w = self._cache
        N = probs.shape[0]
        d = probs.copy()
        d[xp.arange(N), y] -= 1.0
        if w is not None:
            d = d * w.reshape(-1, 1) / max(scalar_to_float(w.sum()), 1e-12)
        else:
            d = d / N
        return d.astype(xp.float32)


# =====================================================================
# 6. MODELE
# =====================================================================

class CNNModel:
    def __init__(self, num_classes=7, in_ch=3, in_size=64,
                 channels=(32, 64, 128), dense_dim=256, dropout=0.3,
                 class_weights=None):
        c1, c2, c3 = channels
        self.in_size = in_size
        self.num_classes = num_classes
        self.channels = tuple(int(c) for c in channels)
        self.dense_dim = int(dense_dim)
        self.dropout = float(dropout)
        self.architecture_version = 'v7_two_conv_bn_gap'
        self.layers = [
            # Bloc 1 : deux convolutions normalisees, puis pooling
            Conv2D(in_ch, c1), BatchNorm2D(c1), ReLU(),
            Conv2D(c1, c1),    BatchNorm2D(c1), ReLU(),
            MaxPool2D(),

            # Bloc 2 : deux convolutions normalisees, puis pooling
            Conv2D(c1, c2), BatchNorm2D(c2), ReLU(),
            Conv2D(c2, c2), BatchNorm2D(c2), ReLU(),
            MaxPool2D(),

            # Bloc 3 : deux convolutions normalisees, puis pooling
            Conv2D(c2, c3), BatchNorm2D(c3), ReLU(),
            Conv2D(c3, c3), BatchNorm2D(c3), ReLU(),
            MaxPool2D(),

            # Remplace Flatten : (N, c3, H, W) -> (N, c3)
            GlobalAveragePool2D(),

            Dense(c3, dense_dim), ReLU(), Dropout(dropout),
            Dense(dense_dim, num_classes),
        ]
        self.loss = SoftmaxCrossEntropy(class_weights=class_weights)

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self):
        out = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.params().items():
                out[f'L{i}_{type(layer).__name__}_{k}'] = v
        return out

    def gradients(self):
        out = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.grads().items():
                out[f'L{i}_{type(layer).__name__}_{k}'] = v
        return out

    def layer_states(self):
        out = {}
        for i, layer in enumerate(self.layers):
            state = layer.state_dict()
            if state:
                out[f'L{i}_{type(layer).__name__}'] = state
        return out

    def load_layer_states(self, states):
        for i, layer in enumerate(self.layers):
            key = f'L{i}_{type(layer).__name__}'
            if key in states:
                layer.load_state_dict(states[key])

    def num_params(self):
        return sum(v.size for v in self.parameters().values())

    def architecture_config(self):
        return {
            'architecture_version': self.architecture_version,
            'in_size': self.in_size,
            'num_classes': self.num_classes,
            'channels': self.channels,
            'dense_dim': self.dense_dim,
            'dropout': self.dropout,
            'head': 'global_average_pooling',
            'conv_per_block': 2,
            'batchnorm_after_conv': True,
        }


# =====================================================================
# 7. OPTIMISEUR ADAM
# =====================================================================

class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = {k: xp.zeros_like(v) for k, v in params.items()}
        self.v = {k: xp.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for k, p in params.items():
            g = grads[k]
            if self.weight_decay > 0:
                g = g + self.weight_decay * p
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            p -= self.lr * (self.m[k] / bc1) / (xp.sqrt(self.v[k] / bc2) + self.eps)

    def state_dict(self):
        return {
            'type': 'Adam',
            't': self.t,
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'm': {k: state_to_cpu(v) for k, v in self.m.items()},
            'v': {k: state_to_cpu(v) for k, v in self.v.items()},
        }

    def load_state_dict(self, state):
        self.t = int(state['t'])
        self.lr = float(state['lr'])
        self.beta1 = float(state['beta1'])
        self.beta2 = float(state['beta2'])
        self.eps = float(state['eps'])
        self.weight_decay = float(state['weight_decay'])
        self.m = {k: to_device(v, dtype=xp.float32).copy() for k, v in state['m'].items()}
        self.v = {k: to_device(v, dtype=xp.float32).copy() for k, v in state['v'].items()}


# =====================================================================
# 8. CHECKPOINTS
# =====================================================================

def save_checkpoint(path, model, optimizer, epoch, best_val_f1,
                    class_to_idx, history=None, config=None, best_val_acc=0.0):
    path = str(path)
    state = {
        'epoch': int(epoch),
        'best_metric_name': 'val_macro_f1',
        'best_score': float(best_val_f1),
        'best_val_f1': float(best_val_f1),
        'best_val_acc': float(best_val_acc),
        'class_to_idx': dict(class_to_idx),
        'label_to_idx': dict(class_to_idx),  # compatibilite avec l'ancienne version
        'history': history or {},
        'config': config or {},
        'arch': model.architecture_config(),
        'model_params': {k: state_to_cpu(v) for k, v in model.parameters().items()},
        'layer_states': state_to_cpu(model.layer_states()),
        'optimizer': optimizer.state_dict(),
    }
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None):
    with open(path, 'rb') as f:
        state = pickle.load(f)
    params = model.parameters()
    saved = state['model_params']
    for k in params:
        params[k][...] = to_device(saved[k], dtype=params[k].dtype)
    if 'layer_states' in state:
        model.load_layer_states(state['layer_states'])
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    class_to_idx = state.get('class_to_idx', state.get('label_to_idx'))
    best_score = state.get('best_val_f1', state.get('best_score', state.get('best_val_acc', 0.0)))
    return (
        int(state['epoch']),
        float(best_score),
        dict(class_to_idx),
        state.get('history', {}),
        state.get('config', {}),
        state.get('arch', {}),
    )


def read_checkpoint_metadata(path):
    with open(path, 'rb') as f:
        state = pickle.load(f)
    best_score = state.get('best_val_f1', state.get('best_score', state.get('best_val_acc', 0.0)))
    return {
        'epoch': state.get('epoch'),
        'best_metric_name': state.get('best_metric_name', 'val_macro_f1'),
        'best_score': best_score,
        'best_val_f1': best_score,
        'best_val_acc': state.get('best_val_acc', 0.0),
        'class_to_idx': state.get('class_to_idx', state.get('label_to_idx')),
        'config': state.get('config', {}),
        'arch': state.get('arch', {}),
    }


# =====================================================================
# 9. DONNEES HAM10000
# =====================================================================

def index_images(*roots):
    """Construit image_id -> chemin image en explorant un ou plusieurs dossiers."""
    extensions = {'.jpg', '.jpeg', '.png'}
    out = {}
    seen_roots = set()
    for root in roots:
        if root is None or str(root).strip() == '':
            continue
        root = Path(root)
        if root.is_file():
            root = root.parent
        key = str(root.resolve())
        if key in seen_roots:
            continue
        seen_roots.add(key)
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in extensions:
                out[p.stem] = str(p)
    return out


def find_metadata_csv(data_dir, metadata_csv=None):
    if metadata_csv:
        return Path(metadata_csv)
    root = Path(data_dir)
    if root.is_file():
        return root
    direct = root / 'HAM10000_metadata.csv'
    if direct.exists():
        return direct
    return sorted(root.rglob('HAM10000_metadata.csv'))[0]


def read_metadata_csv(csv_path):
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['image_id']
            dx = row['dx']
            lesion_id = row.get('lesion_id') or image_id
            rows.append({'image_id': image_id, 'dx': dx, 'lesion_id': lesion_id})
    return rows


def filter_rows_with_images(rows, image_paths):
    kept = [r for r in rows if r['image_id'] in image_paths]
    missing = len(rows) - len(kept)
    return kept, missing


def make_class_mapping(rows, use_ham_order=True):
    present = sorted({r['dx'] for r in rows})
    if use_ham_order and all(c in CLASSES_HAM10000 for c in present):
        classes = [c for c in CLASSES_HAM10000 if c in present]
    else:
        classes = present
    return {c: i for i, c in enumerate(classes)}, classes


def split_train_val_grouped_stratified(rows, val_frac=0.15, seed=42):
    rng = np.random.RandomState(seed)
    groups = {}
    for r in rows:
        lid = r['lesion_id']
        if lid not in groups:
            groups[lid] = {'dx': r['dx'], 'rows': []}
        groups[lid]['rows'].append(r)

    by_class = defaultdict(list)
    for lid, g in groups.items():
        by_class[g['dx']].append(lid)

    train_lids, val_lids = set(), set()
    for dx, lids in by_class.items():
        lids = list(lids)
        rng.shuffle(lids)
        n = len(lids)
        if n == 1:
            train_lids.add(lids[0])
            continue
        n_val = int(round(n * val_frac))
        n_val = min(n - 1, max(1, n_val))
        val_lids.update(lids[:n_val])
        train_lids.update(lids[n_val:])

    train_rows, val_rows = [], []
    for lid, g in groups.items():
        if lid in val_lids:
            val_rows.extend(g['rows'])
        else:
            train_rows.extend(g['rows'])


    return train_rows, val_rows


def compute_class_weights(rows, class_to_idx):
    counts = np.zeros(len(class_to_idx), dtype=np.float32)
    for r in rows:
        counts[class_to_idx[r['dx']]] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return weights.astype(np.float32)


def expand_rows_by_class_augmentation(rows, class_to_idx, target='max', max_repeat=8, seed=42):
    """Sur-echantillonne les lignes d'entrainement en fonction de la rarete des classes.

    Les images ne sont pas copiees sur disque. On repete seulement les lignes du CSV ;
    comme les transformations sont stochastiques, une meme image peut produire des
    variantes differentes au fil des epochs.
    """
    rng = np.random.RandomState(seed)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r['dx']].append(r)

    counts = {c: len(by_class.get(c, [])) for c in class_to_idx}
    nonzero = [v for v in counts.values() if v > 0]
    if not nonzero:
        return list(rows), {}

    if str(target).lower() == 'max':
        target_count = int(max(nonzero))
    elif str(target).lower() == 'median':
        target_count = int(np.median(nonzero))
    elif str(target).lower() == 'mean':
        target_count = int(round(float(np.mean(nonzero))))
    else:
        target_count = int(target)
    target_count = max(1, target_count)
    max_repeat = max(1, int(max_repeat))

    expanded = []
    info = {}
    for c in class_to_idx:
        items = list(by_class.get(c, []))
        n = len(items)
        if n == 0:
            info[c] = {'original': 0, 'effective': 0, 'repeat_max': 0}
            continue
        desired = min(target_count, n * max_repeat)
        base_repeat = max(1, desired // n)
        remainder = max(0, desired - base_repeat * n)

        for rep in range(base_repeat):
            for r in items:
                rr = dict(r)
                rr['_augmented_repeat'] = rep
                expanded.append(rr)
        if remainder > 0:
            chosen = rng.choice(n, size=remainder, replace=remainder > n)
            for j in chosen:
                rr = dict(items[int(j)])
                rr['_augmented_repeat'] = base_repeat
                expanded.append(rr)

        info[c] = {
            'original': n,
            'effective': base_repeat * n + remainder,
            'repeat_max': max_repeat,
        }

    rng.shuffle(expanded)
    return expanded, info


AUGMENT_PRESETS = {
    'light': {
        'noise_std': 0.010,
        'brightness': 0.10,
        'contrast': 0.10,
        'saturation': 0.06,
        'translate': 0.04,
        'cutout': 0.08,
    },
    'medium': {
        'noise_std': 0.020,
        'brightness': 0.18,
        'contrast': 0.18,
        'saturation': 0.10,
        'translate': 0.06,
        'cutout': 0.12,
    },
    'strong': {
        'noise_std': 0.035,
        'brightness': 0.28,
        'contrast': 0.28,
        'saturation': 0.18,
        'translate': 0.10,
        'cutout': 0.18,
    },
}


class ImageTransform:
    def __init__(self, norm='imagenet', training=False, strength='medium',
                 noise_std=None, brightness=None, contrast=None,
                 saturation=None, translate=None, cutout=None):
        self.mean, self.std = NORM_STATS[norm]
        self.training = bool(training)
        preset = dict(AUGMENT_PRESETS.get(str(strength).lower(), AUGMENT_PRESETS['medium']))
        if noise_std is not None:
            preset['noise_std'] = float(noise_std)
        if brightness is not None:
            preset['brightness'] = float(brightness)
        if contrast is not None:
            preset['contrast'] = float(contrast)
        if saturation is not None:
            preset['saturation'] = float(saturation)
        if translate is not None:
            preset['translate'] = float(translate)
        if cutout is not None:
            preset['cutout'] = float(cutout)
        self.cfg = preset

    def __call__(self, img):
        img = img.astype(np.float32, copy=True)
        if self.training:
            img = self.augment(img)
        img = (img - self.mean) / self.std
        return np.ascontiguousarray(img, dtype=np.float32)

    def augment(self, img):
        if np.random.rand() < 0.5:
            img = img[:, :, ::-1]
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :]

        k = np.random.randint(4)
        if k > 0:
            img = np.rot90(img, k, axes=(1, 2))

        tr = float(self.cfg['translate'])
        if tr > 0 and np.random.rand() < 0.7:
            img = self._random_translate(img, tr)

        b = float(self.cfg['brightness'])
        if b > 0 and np.random.rand() < 0.8:
            factor = np.random.uniform(1.0 - b, 1.0 + b)
            img = img * factor

        c = float(self.cfg['contrast'])
        if c > 0 and np.random.rand() < 0.8:
            factor = np.random.uniform(1.0 - c, 1.0 + c)
            mean = img.mean(axis=(1, 2), keepdims=True)
            img = (img - mean) * factor + mean

        s = float(self.cfg['saturation'])
        if s > 0 and np.random.rand() < 0.6:
            factor = np.random.uniform(1.0 - s, 1.0 + s)
            gray = (0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3])
            img = gray + (img - gray) * factor

        ns = float(self.cfg['noise_std'])
        if ns > 0 and np.random.rand() < 0.7:
            img = img + np.random.normal(0.0, ns, size=img.shape).astype(np.float32)

        co = float(self.cfg['cutout'])
        if co > 0 and np.random.rand() < 0.45:
            img = self._random_cutout(img, co)

        return np.ascontiguousarray(np.clip(img, 0.0, 1.0), dtype=np.float32)

    @staticmethod
    def _random_translate(img, max_frac):
        _, H, W = img.shape
        max_shift_y = max(1, int(round(H * max_frac)))
        max_shift_x = max(1, int(round(W * max_frac)))
        dy = np.random.randint(-max_shift_y, max_shift_y + 1)
        dx = np.random.randint(-max_shift_x, max_shift_x + 1)
        padded = np.pad(img, ((0, 0), (max_shift_y, max_shift_y), (max_shift_x, max_shift_x)), mode='edge')
        y0 = max_shift_y + dy
        x0 = max_shift_x + dx
        return padded[:, y0:y0 + H, x0:x0 + W]

    @staticmethod
    def _random_cutout(img, max_frac):
        _, H, W = img.shape
        side_h = max(1, int(round(H * np.random.uniform(max_frac / 2.0, max_frac))))
        side_w = max(1, int(round(W * np.random.uniform(max_frac / 2.0, max_frac))))
        y0 = np.random.randint(0, max(1, H - side_h + 1))
        x0 = np.random.randint(0, max(1, W - side_w + 1))
        fill = img.mean(axis=(1, 2), keepdims=True)
        img[:, y0:y0 + side_h, x0:x0 + side_w] = fill
        return img


class HAM10000Dataset:
    def __init__(self, rows, image_paths, image_size=64, class_to_idx=None,
                 augment=False, cache=False, norm='imagenet', augment_strength='medium',
                 aug_noise_std=None, aug_brightness=None, aug_contrast=None,
                 aug_saturation=None, aug_translate=None, aug_cutout=None):
        from PIL import Image
        self._Image = Image
        self.rows = list(rows)
        self.image_paths = dict(image_paths)
        self.image_size = int(image_size)
        self.class_to_idx = dict(class_to_idx)
        self.augment = bool(augment)
        self.cache = bool(cache)
        self._mem = {}
        self.transform = ImageTransform(
            norm=norm, training=self.augment, strength=augment_strength,
            noise_std=aug_noise_std, brightness=aug_brightness, contrast=aug_contrast,
            saturation=aug_saturation, translate=aug_translate, cutout=aug_cutout,
        )

    def __len__(self):
        return len(self.rows)

    def _load_raw(self, image_id):
        if self.cache and image_id in self._mem:
            return self._mem[image_id]
        path = self.image_paths[image_id]
        with self._Image.open(path) as img:
            img = img.convert('RGB').resize((self.image_size, self.image_size), self._Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)
        if self.cache:
            self._mem[image_id] = arr
        return arr

    def get_batch(self, indices):
        imgs, labels = [], []
        for i in indices:
            row = self.rows[int(i)]
            img = self._load_raw(row['image_id'])
            img = self.transform(img)
            imgs.append(img)
            labels.append(self.class_to_idx[row['dx']])
        return np.stack(imgs).astype(np.float32, copy=False), np.array(labels, dtype=np.int64)


class BatchIterator:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, seed=None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        n = len(self.dataset)
        idx = np.arange(n)
        if self.shuffle:
            rng = np.random.RandomState(None if self.seed is None else self.seed + self.epoch)
            rng.shuffle(idx)
            self.epoch += 1
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            yield self.dataset.get_batch(idx[start:end])

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size


# =====================================================================
# 10. ENTRAINEMENT ET EVALUATION
# =====================================================================

def grad_norm(model):
    total = 0.0
    for v in model.gradients().values():
        total += scalar_to_float(xp.sum(v * v))
    return math.sqrt(total)


def evaluate(model, iterator, num_classes, show_progress=True,
             batch_logger=None, epoch=0, classes=None, total_t0=None):
    correct = total = n_batches = 0
    loss_sum = 0.0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    pb = ProgressBar(len(iterator), prefix='   val', width=18) if show_progress else None
    eval_t0 = time.time()
    total_t0 = eval_t0 if total_t0 is None else total_t0
    classes = list(classes) if classes is not None else [str(i) for i in range(num_classes)]

    for i, (x_cpu, y_cpu) in enumerate(iterator):
        x = to_device(x_cpu, dtype=xp.float32)
        y = to_device(y_cpu, dtype=xp.int64)
        logits = model.forward(x, training=False)
        loss, probs = model.loss.forward(logits, y)
        loss_sum += loss
        n_batches += 1
        preds_cpu = to_cpu(probs.argmax(axis=1)).astype(np.int64, copy=False)
        y_np = np.asarray(y_cpu, dtype=np.int64)
        correct += int((preds_cpu == y_np).sum())
        total += len(y_np)

        batch_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        np.add.at(batch_cm, (y_np, preds_cpu), 1)
        np.add.at(cm, (y_np, preds_cpu), 1)
        running_metrics = per_class_metrics(cm)

        if batch_logger is not None:
            batch_logger.log(
                phase='val',
                epoch=epoch,
                batch=i + 1,
                num_batches=len(iterator),
                global_batch=int(epoch) * len(iterator) + i + 1,
                samples_batch=len(y_np),
                samples_seen_epoch=total,
                samples_seen_total=int(epoch) * len(iterator.dataset) + total,
                loss=loss,
                loss_mean_epoch=loss_sum / max(n_batches, 1),
                grad_norm=None,
                batch_metrics=per_class_metrics(batch_cm),
                running_metrics=running_metrics,
                elapsed_epoch_s=time.time() - eval_t0,
                elapsed_total_s=time.time() - total_t0,
            )

        if pb:
            pb.update(i + 1, loss=f'{loss_sum / n_batches:.3f}',
                      acc=f'{correct / max(total, 1):5.1%}', f1=f'{running_metrics["macro_f1"]:.3f}')
    synchronize_backend()
    return loss_sum / max(n_batches, 1), correct / max(total, 1), cm


def preprocess_single_image(path, image_size, norm):
    from PIL import Image
    mean, std = NORM_STATS[norm]
    with Image.open(path) as img:
        img = img.convert('RGB').resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = ((arr.transpose(2, 0, 1) - mean) / std)[np.newaxis]
    return arr.astype(np.float32)


def train(args):
    set_backend(args.backend, seed=args.seed)
    banner('HAM10000 - CNN CuPy/NumPy v7 : 2 conv/bloc + BatchNorm + GAP')

    section('Configuration')
    cfg = [(k, str(v)) for k, v in sorted(vars(args).items()) if k != 'cmd']
    print(fmt_table(['parametre', 'valeur'], cfg))

    np.random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f'  backend de calcul              : {backend_info()}')

    section('Chargement des metadonnees et indexation des images')
    metadata_path = find_metadata_csv(args.data_dir, args.metadata_csv)
    rows = read_metadata_csv(metadata_path)

    image_roots = []
    if args.images_dir:
        image_roots.append(args.images_dir)
    image_roots.append(args.data_dir)
    image_roots.append(metadata_path.parent)
    image_paths = index_images(*image_roots)

    rows, n_missing = filter_rows_with_images(rows, image_paths)
    print(f'  fichier metadata              : {metadata_path}')
    class_to_idx, classes = make_class_mapping(rows, use_ham_order=True)
    print(f'  images referencees dans le CSV : {len(rows) + n_missing}')
    print(f'  images trouvees et retenues    : {len(rows)}')
    print(f'  images ignorees car absentes   : {n_missing}')
    print(f'  classes                        : {classes}')

    section('Split stratifie et groupe par lesion_id')
    train_rows, val_rows = split_train_val_grouped_stratified(rows, val_frac=args.val_frac, seed=args.seed)
    train_dist = Counter(r['dx'] for r in train_rows)
    val_dist = Counter(r['dx'] for r in val_rows)
    class_weights = None if args.no_class_weights else compute_class_weights(train_rows, class_to_idx)

    train_rows_effective = list(train_rows)
    aug_info = {}
    if (not args.no_augment) and args.balance_augment:
        train_rows_effective, aug_info = expand_rows_by_class_augmentation(
            train_rows, class_to_idx, target=args.augment_target,
            max_repeat=args.max_augment_repeat, seed=args.seed,
        )

    effective_dist = Counter(r['dx'] for r in train_rows_effective)
    dist_rows = []
    for c in classes:
        tr = train_dist.get(c, 0)
        tre = effective_dist.get(c, 0)
        va = val_dist.get(c, 0)
        pct = tr * 100.0 / max(len(train_rows), 1)
        cw = '' if class_weights is None else f'{class_weights[class_to_idx[c]]:.3f}'
        repeat = ''
        if c in aug_info and aug_info[c]['original'] > 0:
            repeat = f'x{tre / max(tr, 1):.2f}'
        dist_rows.append([c, CLASS_NAMES_FR.get(c, c), tr, tre, va, f'{pct:5.2f}%', cw, repeat])
    dist_rows.append(['', f'{C.BOLD}TOTAL{C.RESET}', f'{C.BOLD}{len(train_rows)}{C.RESET}',
                      f'{C.BOLD}{len(train_rows_effective)}{C.RESET}',
                      f'{C.BOLD}{len(val_rows)}{C.RESET}', '100.00%', '', ''])
    print(fmt_table(['code', 'classe', 'train reel', 'train effectif', 'val', '%train', 'poids loss', 'repeat'],
                    dist_rows, align=['<', '<', '>', '>', '>', '>', '>', '>']))

    train_ds = HAM10000Dataset(
        train_rows_effective, image_paths, image_size=args.image_size, class_to_idx=class_to_idx,
        augment=not args.no_augment, cache=args.cache, norm=args.norm,
        augment_strength=args.augment_strength, aug_noise_std=args.aug_noise_std,
        aug_brightness=args.aug_brightness, aug_contrast=args.aug_contrast,
        aug_saturation=args.aug_saturation, aug_translate=args.aug_translate,
        aug_cutout=args.aug_cutout,
    )
    val_ds = HAM10000Dataset(
        val_rows, image_paths, image_size=args.image_size, class_to_idx=class_to_idx,
        augment=False, cache=args.cache, norm=args.norm,
        augment_strength='light',
    )
    train_iter = BatchIterator(train_ds, args.batch_size, shuffle=True, seed=args.seed)
    val_iter = BatchIterator(val_ds, args.batch_size, shuffle=False)

    section('Construction du modele')
    channels = tuple(int(x) for x in args.channels.split(','))
    model = CNNModel(
        num_classes=len(classes), in_ch=3, in_size=args.image_size,
        channels=channels, dense_dim=args.dense_dim, dropout=args.dropout,
        class_weights=class_weights,
    )
    params_cache = model.parameters()
    grads_cache = model.gradients()
    optimizer = Adam(params_cache, lr=args.lr, weight_decay=args.weight_decay)

    layer_rows = []
    for i, layer in enumerate(model.layers):
        n_p = sum(v.size for v in layer.params().values())
        layer_rows.append([i, type(layer).__name__, f'{n_p:,}'])
    layer_rows.append(['', f'{C.BOLD}TOTAL{C.RESET}', f'{C.BOLD}{model.num_params():,}{C.RESET}'])
    print(fmt_table(['#', 'couche', 'parametres'], layer_rows, align=['>', '<', '>']))

    start_epoch = 0
    best_val_f1 = 0.0
    best_val_acc = 0.0
    history = make_empty_history()

    config = vars(args).copy()
    config['classes'] = classes
    config['class_to_idx'] = class_to_idx
    config['channels_tuple'] = channels

    if args.resume:
        section(f'Reprise depuis {args.resume}')
        last_epoch, best_val_f1, ckpt_class_to_idx, hist, ckpt_config, _ = load_checkpoint(args.resume, model, optimizer)
        if ckpt_class_to_idx != class_to_idx:
            raise ValueError(
                "Le mapping des classes du checkpoint ne correspond pas au dataset actuel. "
                "Utilisez le meme CSV, le meme ordre de classes, ou recommencez un entrainement."
            )
        start_epoch = last_epoch + 1
        if hist:
            history.update(hist)
            if 'val_acc' in hist and 'val_accuracy' not in hist:
                history['val_accuracy'] = history['val_acc']
            if 'train_acc' in hist and 'train_accuracy' not in hist:
                history['train_accuracy'] = history['train_acc']
            if history.get('val_acc'):
                best_val_acc = max(history.get('val_acc', [0.0]))
        print(f'  reprise epoch {start_epoch}, meilleur val_macro_f1 precedent {best_val_f1:.4f}')
        if ckpt_config:
            print(f'  configuration checkpoint chargee: image_size={ckpt_config.get("image_size")}, norm={ckpt_config.get("norm")}')

    last_path = os.path.join(args.checkpoint_dir, 'last.pkl')
    best_path = os.path.join(args.checkpoint_dir, 'best.pkl')

    batch_log_path = None if args.no_batch_log else (
        args.batch_log_csv or os.path.join(args.checkpoint_dir, 'batch_metrics.csv')
    )
    batch_logger = BatchMetricLogger(
        batch_log_path,
        classes,
        append=bool(args.resume),
        enabled=not args.no_batch_log,
    )
    if batch_logger.enabled:
        print(f'  log batch CSV                 : {batch_log_path}')

    section("Boucle d'entrainement")
    train_t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.time()
        loss_sum = gn_sum = 0.0
        correct = total = n_batches = gn_n = 0
        train_cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
        pb = ProgressBar(len(train_iter), prefix=f' epoch {epoch:02d}', width=24)

        for batch_idx, (x_cpu, y_cpu) in enumerate(train_iter):
            x = to_device(x_cpu, dtype=xp.float32)
            y = to_device(y_cpu, dtype=xp.int64)
            logits = model.forward(x, training=True)
            loss, probs = model.loss.forward(logits, y)
            dlogits = model.loss.backward()
            model.backward(dlogits)
            compute_gn = args.grad_norm_every > 0 and ((batch_idx + 1) % args.grad_norm_every == 0 or batch_idx + 1 == len(train_iter))
            gn = grad_norm(model) if compute_gn else float('nan')
            optimizer.step(params_cache, grads_cache)

            loss_sum += loss
            n_batches += 1
            if compute_gn:
                gn_sum += gn
                gn_n += 1
            preds_cpu = to_cpu(probs.argmax(axis=1)).astype(np.int64, copy=False)
            y_np = np.asarray(y_cpu, dtype=np.int64)
            correct += int((preds_cpu == y_np).sum())
            total += len(y_np)

            batch_cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
            np.add.at(batch_cm, (y_np, preds_cpu), 1)
            np.add.at(train_cm, (y_np, preds_cpu), 1)
            train_running_metrics = per_class_metrics(train_cm)

            if batch_logger.enabled:
                batch_logger.log(
                    phase='train',
                    epoch=epoch,
                    batch=batch_idx + 1,
                    num_batches=len(train_iter),
                    global_batch=epoch * len(train_iter) + batch_idx + 1,
                    samples_batch=len(y_np),
                    samples_seen_epoch=total,
                    samples_seen_total=epoch * len(train_iter.dataset) + total,
                    loss=loss,
                    loss_mean_epoch=loss_sum / max(n_batches, 1),
                    grad_norm=gn,
                    batch_metrics=per_class_metrics(batch_cm),
                    running_metrics=train_running_metrics,
                    elapsed_epoch_s=time.time() - epoch_t0,
                    elapsed_total_s=time.time() - train_t0,
                )

            train_macro_f1_live = train_running_metrics['macro_f1']
            gn_display = '-' if math.isnan(gn) else f'{gn:5.2f}'
            pb.update(batch_idx + 1, loss=f'{loss:.3f}',
                      acc=f'{correct / max(total, 1):5.1%}', f1=f'{train_macro_f1_live:.3f}', gn=gn_display)

        train_loss = loss_sum / max(n_batches, 1)
        train_acc = correct / max(total, 1)
        gn_mean = gn_sum / gn_n if gn_n > 0 else float('nan')
        train_metrics = per_class_metrics(train_cm)
        val_loss, val_acc, val_cm = evaluate(
            model,
            val_iter,
            len(classes),
            show_progress=True,
            batch_logger=batch_logger if args.batch_log_val else None,
            epoch=epoch,
            classes=classes,
            total_t0=train_t0,
        )
        val_metrics = per_class_metrics(val_cm)
        macro_f1 = val_metrics['macro_f1']
        epoch_time = time.time() - epoch_t0

        history_append(history, 'train_loss', train_loss)
        history_append_metrics(history, 'train', train_metrics)
        history_append(history, 'val_loss', val_loss)
        history_append_metrics(history, 'val', val_metrics)
        history_append(history, 'grad_norm', gn_mean)
        history_append(history, 'epoch_time', epoch_time)

        diag = diagnose_fit_state(history, train_loss, train_metrics, val_loss, val_metrics, args)
        history_append(history, 'fit_state', diag['state'])

        improved = macro_f1 > best_val_f1
        if improved:
            best_val_f1 = macro_f1
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        prev_f1 = history['val_macro_f1'][-2] if len(history['val_macro_f1']) > 1 else macro_f1
        delta = macro_f1 - prev_f1
        delta_color = C.GREEN if delta >= 0 else C.RED
        delta_str = f'{delta_color}{"+" if delta >= 0 else ""}{delta:.4f}{C.RESET}'
        f1_color = C.GREEN if improved else C.YELLOW
        elapsed_total = time.time() - train_t0
        epochs_done = epoch - start_epoch + 1
        eta_total = (args.epochs - epoch - 1) * (elapsed_total / max(epochs_done, 1))

        print(
            f'  epoch {epoch:02d}: '
            f'train[loss={train_loss:.4f} acc={train_acc:5.1%} macroF1={train_metrics["macro_f1"]:.3f}] '
            f'val[loss={val_loss:.4f} acc={val_acc:5.1%} '
            f'macroF1={f1_color}{macro_f1:.3f}{C.RESET} ({delta_str}) '
            f'wF1={val_metrics["weighted_f1"]:.3f} balAcc={val_metrics["balanced_accuracy"]:.3f}] '
            f'|grad|={gn_mean:.2f}  '
            f'{C.DIM}{temps_humain(epoch_time)} | ETA total {temps_humain(eta_total)}{C.RESET}'
        )
        print('  ' + fmt_epoch_metrics_table(train_loss, train_metrics, val_loss, val_metrics).replace('\n', '\n  '))
        print('  ' + fmt_fit_diagnostic(diag).replace('\n', '\n  '))

        if args.report_every > 0 and ((epoch + 1) % args.report_every == 0 or epoch == args.epochs - 1):
            print(f'\n  {C.BOLD}Rapport classification (val) - epoch {epoch}{C.RESET}')
            print('  ' + fmt_classification_report(val_cm, classes).replace('\n', '\n  '))
            print(f'\n  {C.BOLD}Metriques globales (val){C.RESET}')
            print('  ' + fmt_global_metrics(val_cm).replace('\n', '\n  '))
            print(f'\n  {C.BOLD}Matrice de confusion{C.RESET}')
            print('  ' + fmt_confusion_matrix(val_cm, classes).replace('\n', '\n  '))
            print()

        save_checkpoint(last_path, model, optimizer, epoch, best_val_f1, class_to_idx, history, config, best_val_acc=best_val_acc)
        print(f'  checkpoint last -> {last_path}')
        if improved:
            save_checkpoint(best_path, model, optimizer, epoch, best_val_f1, class_to_idx, history, config, best_val_acc=best_val_acc)
            print(f'  {C.GREEN}[+] nouveau meilleur val_macro_f1={macro_f1:.4f} -> {best_path}{C.RESET}')

    batch_logger.close()

    banner('Entrainement termine', color=C.GREEN)
    print(f'  meilleur val_macro_f1: {C.GREEN}{C.BOLD}{best_val_f1:.4f}{C.RESET}')
    print(f'  meilleur val_acc observe: {C.GREEN}{C.BOLD}{best_val_acc:.4f}{C.RESET}')
    print(f'  checkpoints:      {best_path}, {last_path}')
    print(f'  duree totale:     {temps_humain(time.time() - train_t0)}')


def predict(args):
    set_backend(args.backend, seed=None)
    meta = read_checkpoint_metadata(args.checkpoint)
    class_to_idx = dict(meta['class_to_idx'])
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    arch = meta.get('arch', {})
    config = meta.get('config', {})
    image_size = int(args.image_size or arch.get('in_size') or config.get('image_size') or 64)
    channels = tuple(arch.get('channels') or config.get('channels_tuple') or (32, 64, 128))
    dense_dim = int(arch.get('dense_dim') or config.get('dense_dim') or 256)
    norm = args.norm or config.get('norm') or 'imagenet'

    model = CNNModel(
        num_classes=len(class_to_idx), in_size=image_size, channels=channels,
        dense_dim=dense_dim, dropout=0.0, class_weights=None,
    )
    epoch, best_val_f1, _, hist, _, _ = load_checkpoint(args.checkpoint, model, optimizer=None)

    x = to_device(preprocess_single_image(args.image, image_size=image_size, norm=norm), dtype=xp.float32)
    logits = model.forward(x, training=False)
    z = logits - logits.max(axis=1, keepdims=True)
    ez = xp.exp(z)
    probs = to_cpu((ez / ez.sum(axis=1, keepdims=True))[0])

    top_k = min(max(1, int(args.top_k)), len(probs))
    top = np.argsort(-probs)[:top_k]

    banner('HAM10000 - Inference NumPy')
    print(f'  checkpoint:      {args.checkpoint}')
    print(f'  image:           {args.image}')
    print(f'  modele charge:   epoch {epoch}, best_val_macro_f1={best_val_f1:.4f}, epochs entrainees={len(hist.get("val_macro_f1", []))}')
    print(f'  preprocessing:   image_size={image_size}, norm={norm}')
    print(f'  backend:         {backend_info()}')

    section(f'Top-{top_k} probabilites')
    labels = [idx_to_class[int(i)] for i in top]
    p_vals = [float(probs[int(i)]) for i in top]
    print(fmt_prob_bars(labels, p_vals, width=24))

    pred = idx_to_class[int(top[0])]
    confidence = p_vals[0]
    if confidence > 0.7:
        badge = f'{C.GREEN}HAUTE{C.RESET}'
    elif confidence > 0.4:
        badge = f'{C.YELLOW}MOYENNE{C.RESET}'
    else:
        badge = f'{C.RED}FAIBLE{C.RESET}'
    print(f'\n  Prediction:  {C.BOLD}{pred}{C.RESET} ({CLASS_NAMES_FR.get(pred, "")})')
    print(f'  Confiance:   {confidence * 100:.2f}%  fiabilite={badge}')


# =====================================================================
# 11. CLI
# =====================================================================


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    pt = sub.add_parser('train', help='entrainer le CNN NumPy sur HAM10000')
    pt.add_argument('--data-dir', type=str, required=True,
                    help='dossier du dataset ; peut aussi etre le chemin direct vers HAM10000_metadata.csv')
    pt.add_argument('--metadata-csv', type=str, default=None,
                    help='chemin explicite vers HAM10000_metadata.csv si --data-dir ne suffit pas')
    pt.add_argument('--images-dir', type=str, default=None,
                    help='dossier explicite contenant les images si elles sont separees du CSV')
    pt.add_argument('--image-size', type=int, default=64)
    pt.add_argument('--batch-size', type=int, default=32)
    pt.add_argument('--epochs', type=int, default=30)
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--weight-decay', type=float, default=1e-4)
    pt.add_argument('--dropout', type=float, default=0.3)
    pt.add_argument('--channels', type=str, default='32,64,128', help='exemple: 32,64,128')
    pt.add_argument('--dense-dim', type=int, default=256)
    pt.add_argument('--val-frac', type=float, default=0.15)
    pt.add_argument('--checkpoint-dir', type=str, default='checkpoints_numpy')
    pt.add_argument('--resume', type=str, default=None)
    pt.add_argument('--cache', action='store_true', help='garder les images brutes redimensionnees en RAM')
    pt.add_argument('--no-augment', action='store_true')
    pt.set_defaults(balance_augment=True)
    pt.add_argument('--balance-augment', dest='balance_augment', action='store_true',
                    help='sur-echantillonner les classes rares par repetitions avec transformations aleatoires')
    pt.add_argument('--no-balance-augment', dest='balance_augment', action='store_false',
                    help='desactiver le sur-echantillonnage pondere par classe')
    pt.add_argument('--augment-target', type=str, default='max',
                    help='cible du train effectif par classe: max, median, mean ou entier')
    pt.add_argument('--max-augment-repeat', type=int, default=8,
                    help='borne le nombre de repetitions par image pour eviter un dataset artificiel trop grand')
    pt.add_argument('--augment-strength', type=str, default='medium', choices=['light', 'medium', 'strong'])
    pt.add_argument('--aug-noise-std', type=float, default=None,
                    help='ecart-type du bruit gaussien applique avant normalisation')
    pt.add_argument('--aug-brightness', type=float, default=None,
                    help='amplitude de luminosite, ex: 0.18 pour plus ou moins 18 pour cent')
    pt.add_argument('--aug-contrast', type=float, default=None,
                    help='amplitude de contraste, ex: 0.18 pour plus ou moins 18 pour cent')
    pt.add_argument('--aug-saturation', type=float, default=None,
                    help='amplitude de saturation, ex: 0.10 pour plus ou moins 10 pour cent')
    pt.add_argument('--aug-translate', type=float, default=None,
                    help='translation maximale en fraction de taille image')
    pt.add_argument('--aug-cutout', type=float, default=None,
                    help='taille maximale de l effacement aleatoire en fraction de taille image')
    pt.add_argument('--no-class-weights', action='store_true')
    pt.add_argument('--norm', type=str, default='imagenet')
    pt.add_argument('--report-every', type=int, default=5)
    pt.add_argument('--grad-norm-every', type=int, default=0,
                    help='frequence de calcul de la norme du gradient; 0 la desactive pour limiter les synchronisations GPU')
    pt.add_argument('--batch-log-csv', type=str, default=None,
                    help='chemin du CSV de suivi batch par batch; par defaut: checkpoint-dir/batch_metrics.csv')
    pt.add_argument('--no-batch-log', action='store_true',
                    help='desactiver le journal CSV des metriques de chaque batch')
    pt.set_defaults(batch_log_val=True)
    pt.add_argument('--batch-log-val', dest='batch_log_val', action='store_true',
                    help='inclure les batchs de validation dans le journal CSV')
    pt.add_argument('--no-batch-log-val', dest='batch_log_val', action='store_false',
                    help='ne journaliser que les batchs d entrainement')
    pt.add_argument('--seed', type=int, default=42)
    pt.add_argument('--backend', type=str, default='auto', choices=['auto', 'cupy', 'cuda', 'gpu', 'cpu', 'numpy'],
                    help='backend de calcul: auto utilise CuPy si CUDA est disponible, sinon NumPy CPU')

    pp = sub.add_parser('predict', help='inference top-k sur une image')
    pp.add_argument('--checkpoint', type=str, required=True)
    pp.add_argument('--image', type=str, required=True)
    pp.add_argument('--image-size', type=int, default=None, help='par defaut: valeur du checkpoint')
    pp.add_argument('--norm', type=str, default=None, help='par defaut: valeur du checkpoint')
    pp.add_argument('--top-k', type=int, default=3)
    pp.add_argument('--backend', type=str, default='auto', choices=['auto', 'cupy', 'cuda', 'gpu', 'cpu', 'numpy'],
                    help='backend de calcul: auto utilise CuPy si CUDA est disponible, sinon NumPy CPU')

    return p


def main():
    args = build_parser().parse_args()
    {'train': train, 'predict': predict}[args.cmd](args)


if __name__ == '__main__':
    main()
