# src/metrics.py
# Metrik evaluasi sesuai Subbab 2.8 proposal:
#   utama     : EER            (Persamaan 2.14 - 2.16)
#   pendukung : AUC, F1, balanced accuracy (Persamaan 2.17 - 2.18), min t-DCF
#
# Konvensi label: bona fide = 0, spoof = 1. fake_score = skor kelas spoof.
import os
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, balanced_accuracy_score


# ---------------------------------------------------------------- EER
def compute_eer(y_true, fake_score):
    y_true = np.asarray(y_true).astype(int)
    fake_score = np.asarray(fake_score).astype(float)
    fpr, tpr, thresholds = roc_curve(y_true, fake_score, pos_label=1)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thresholds[i])


def compute_metrics(y_true, fake_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    fake_score = np.asarray(fake_score).astype(float)
    eer, eer_threshold = compute_eer(y_true, fake_score)
    pred = (fake_score >= threshold).astype(int)
    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": float(roc_auc_score(y_true, fake_score)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }


# ---------------------------------------------------------------- min t-DCF
# Model biaya resmi ASVspoof 2019 (Kinnunen et al., 2018).
COST_MODEL_2019 = {
    "Pspoof": 0.05,
    "Ptar": (1 - 0.05) * 0.99,
    "Pnon": (1 - 0.05) * 0.01,
    "Cmiss_asv": 1.0,
    "Cfa_asv": 10.0,
    "Cmiss_cm": 1.0,
    "Cfa_cm": 10.0,
}


def _det_curve(target, nontarget):
    """Kembalikan (frr, far, thresholds) untuk dua himpunan skor."""
    n_t, n_n = len(target), len(nontarget)
    all_scores = np.concatenate((target, nontarget))
    labels = np.concatenate((np.ones(n_t), np.zeros(n_n)))
    idx = np.argsort(all_scores, kind="mergesort")
    labels = labels[idx]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = n_n - (np.arange(1, n_t + n_n + 1) - tar_trial_sums)
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / n_t))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / n_n))
    thresholds = np.concatenate((np.atleast_1d(all_scores[idx[0]] - 1e-8), all_scores[idx]))
    return frr, far, thresholds


def asv_error_rates(tar_asv, non_asv, spoof_asv):
    """Laju galat sistem ASV pada ambang EER-nya sendiri.
    Kembalikan (Pfa_asv, Pmiss_asv, Pmiss_spoof_asv)."""
    frr, far, thr = _det_curve(np.asarray(tar_asv), np.asarray(non_asv))
    i = int(np.nanargmin(np.abs(frr - far)))
    tau = thr[i]
    Pmiss_asv = float(np.sum(np.asarray(tar_asv) < tau) / len(tar_asv))
    Pfa_asv = float(np.sum(np.asarray(non_asv) >= tau) / len(non_asv))
    Pmiss_spoof_asv = float(np.sum(np.asarray(spoof_asv) < tau) / len(spoof_asv))
    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_min_tdcf(bonafide_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,
                     cost_model=None):
    """min t-DCF ternormalisasi (formulasi ASVspoof 2019).

    bonafide_cm / spoof_cm: skor CM dengan konvensi 'makin besar makin bona fide'.
    """
    c = cost_model or COST_MODEL_2019
    bonafide_cm = np.asarray(bonafide_cm, dtype=float)
    spoof_cm = np.asarray(spoof_cm, dtype=float)

    C1 = (c["Ptar"] * (c["Cmiss_cm"] - c["Cmiss_asv"] * Pmiss_asv)
          - c["Pnon"] * c["Cfa_asv"] * Pfa_asv)
    C2 = c["Cfa_cm"] * c["Pspoof"] * (1 - Pmiss_spoof_asv)
    if C1 <= 0 or C2 <= 0:
        return float("nan"), float("nan")

    frr_cm, far_cm, thr = _det_curve(bonafide_cm, spoof_cm)
    tdcf = C1 * frr_cm + C2 * far_cm
    tdcf_norm = tdcf / min(C1, C2)
    i = int(np.argmin(tdcf_norm))
    return float(tdcf_norm[i]), float(thr[i])


def load_asv_scores(path):
    """Baca berkas skor ASV resmi ASVspoof 2019.
    Format tiap baris: <sistem_serangan> <jenis_trial> <skor>"""
    tar, non, spoof = [], [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            kind, score = p[1], float(p[2])
            if kind == "target":
                tar.append(score)
            elif kind == "nontarget":
                non.append(score)
            elif kind == "spoof":
                spoof.append(score)
    return np.array(tar), np.array(non), np.array(spoof)


def min_tdcf_from_manifest(y_true, fake_score, asv_score_file):
    """Jalan pintas: hitung min t-DCF dari label + skor CM + berkas skor ASV resmi.
    Mengembalikan nan bila berkas ASV tidak tersedia."""
    if not asv_score_file or not os.path.exists(asv_score_file):
        return float("nan")
    y_true = np.asarray(y_true).astype(int)
    fake_score = np.asarray(fake_score, dtype=float)
    cm = -fake_score                      # ubah ke orientasi 'besar = bona fide'
    tar, non, spoof = load_asv_scores(asv_score_file)
    if len(tar) == 0 or len(non) == 0 or len(spoof) == 0:
        return float("nan")
    Pfa, Pmiss, Pmiss_spoof = asv_error_rates(tar, non, spoof)
    val, _ = compute_min_tdcf(cm[y_true == 0], cm[y_true == 1], Pfa, Pmiss, Pmiss_spoof)
    return val


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(500), np.ones(2000)]).astype(int)
    s = np.concatenate([rng.normal(0.25, 0.15, 500), rng.normal(0.75, 0.15, 2000)])
    m = compute_metrics(y, s)
    print({k: round(v, 4) for k, v in m.items()})
    asv = ("data/raw/LA/ASVspoof2019_LA_asv_scores/"
           "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt")
    print("min t-DCF:", round(min_tdcf_from_manifest(y, s, asv), 5))
