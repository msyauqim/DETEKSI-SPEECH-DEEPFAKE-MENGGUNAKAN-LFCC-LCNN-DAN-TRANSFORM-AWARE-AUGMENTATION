# src/train.py
# Berkas UTAMA pelatihan LFCC-LCNN, sesuai Tabel 3.5 proposal.
#
#   python src/train.py --config configs/baseline.yaml --seed 2026
#   python src/train.py --config configs/baseline.yaml --augment --seed 2026
#   python src/train.py --config configs/baseline.yaml --subset 400 --epochs 3   # uji cepat
#   # ablation komponen augmentasi:
#   python src/train.py --config configs/baseline.yaml --augment --disable reverb --tag no_reverb
#   # ablation komponen fitur:
#   python src/train.py --config configs/baseline.yaml --components delta --tag static_delta
import os, sys, json, time, math, argparse, yaml
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from device import DEVICE
from reproducibility import set_seed
from dataset import ASVDataset, collate
from features_lfcc import LFCCFeature
from model_lcnn import LCNN
from metrics import compute_metrics


def build_lfcc(cfg, components=None):
    lf = cfg["lfcc"]
    stats = None
    p = lf.get("norm_stats")
    comp = components or lf["components"]
    if p and os.path.exists(p):
        s = json.load(open(p))
        if s.get("components") == comp:
            stats = s
        else:
            print(f"  ! norm_stats dilewati: dihitung untuk '{s.get('components')}', "
                  f"sedangkan berjalan pada '{comp}'")
    elif p:
        print(f"  ! {p} belum ada; jalankan src/compute_norm_stats.py agar CMVN aktif")
    return LFCCFeature(sample_rate=cfg["audio"]["target_sr"], n_filter=lf["n_filter"],
                       n_lfcc=lf["n_lfcc"], n_fft=lf["n_fft"], win_length=lf["win_length"],
                       hop_length=lf["hop_length"], log_lf=lf["log_lf"],
                       window=lf.get("window", "hamming"),
                       components=comp, norm_stats=stats)


def make_scheduler(opt, cfg, total_steps):
    """Cosine annealing dengan warmup linear (Tabel 3.5)."""
    if str(cfg["train"].get("scheduler", "none")).lower() != "cosine":
        return None
    warm = int(cfg["train"].get("warmup_steps", 0))

    def lr_lambda(step):
        if warm > 0 and step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def class_weights(df, mode):
    """Bobot berbanding terbalik terhadap jumlah sampel tiap kelas (Tabel 3.5)."""
    if str(mode).lower() != "balanced":
        return None
    n = len(df)
    counts = df.label.value_counts().reindex([0, 1]).fillna(0).values.astype(float)
    w = n / (2.0 * np.clip(counts, 1, None))
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def cm_score(logits):
    """Skor detektor sesuai Subbab 2.4.2: log p(bona fide | x) - log p(spoof | x).
    Nilai makin besar berarti makin condong ke kelas bona fide."""
    lp = logits.log_softmax(dim=1)
    return (lp[:, 0] - lp[:, 1]).detach().cpu().numpy()


def run_inference(model, lfcc, loader):
    """Kembalikan (label, skor_spoof). skor_spoof = -cm_score agar berorientasi
    'makin besar makin spoof', sesuai konvensi label spoof = 1."""
    model.eval()
    ys, scores = [], []
    with torch.no_grad():
        for wav, label, _, _ in loader:
            logits, _ = model(lfcc(wav.to(DEVICE)))
            scores.append(-cm_score(logits))
            ys.append(label.numpy())
    return np.concatenate(ys), np.concatenate(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None, help="subset kecil untuk uji cepat")
    ap.add_argument("--val-subset", type=int, default=None,
                    help="jumlah berkas dev untuk validasi tiap epoch (hemat waktu); "
                         "0 atau kosong = seluruh partisi dev")
    ap.add_argument("--workers", type=int, default=None,
                    help="default dari config; 0 paling cepat pada Apple Silicon")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--components", choices=["static", "delta", "delta2"], default=None,
                    help="ablation komponen fitur LFCC")
    ap.add_argument("--disable", nargs="*", default=None,
                    help="ablation komponen augmentasi, mis. --disable reverb noise")
    ap.add_argument("--tag", default=None, help="nama berkas checkpoint, otomatis bila kosong")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tr = cfg["train"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 2026)
    augment = args.augment or cfg.get("augment", {}).get("enabled", False)
    epochs = args.epochs if args.epochs is not None else tr["max_epochs"]
    bs = args.batch_size if args.batch_size is not None else tr["batch_size"]
    patience = args.patience if args.patience is not None else tr.get("patience", 10)
    comp = args.components or cfg["lfcc"]["components"]
    sr, seconds = cfg["audio"]["target_sr"], cfg["audio"]["seconds"]
    workers = args.workers if args.workers is not None else tr.get("workers", 0)
    val_subset = args.val_subset if args.val_subset is not None else tr.get("val_subset")
    val_subset = val_subset or None

    aug_cfg = dict(cfg.get("augment", {}))
    if args.disable is not None:
        aug_cfg["disable"] = args.disable

    set_seed(seed)
    base = "augmented" if augment else "baseline"
    tag = args.tag or f"{base}_seed{seed}"
    if args.tag:
        tag = f"{args.tag}_seed{seed}"

    print(f"== TRAIN {tag} | device={DEVICE} | epoch<= {epochs} | batch={bs} "
          f"| augment={augment} | fitur={comp} ==")
    if augment and aug_cfg.get("disable"):
        print(f"   ablation augmentasi, dimatikan: {aug_cfg['disable']}")

    # ---- data ----
    train_ds = ASVDataset(cfg["data"]["train_manifest"], training=True, augment=augment,
                          augment_cfg=aug_cfg, sr=sr, seconds=seconds,
                          subset=args.subset, seed=seed)
    val_ds = ASVDataset(cfg["data"].get("dev_manifest", "manifests/dev_protocol.csv"),
                        training=False, augment=False, sr=sr, seconds=seconds,
                        subset=args.subset or val_subset, seed=seed)
    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers,
                          collate_fn=collate, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=workers,
                        collate_fn=collate)
    print(f"train={len(train_ds):,} | val={len(val_ds):,}"
          + (f" (subset dev untuk hemat waktu)" if val_subset and not args.subset else ""))

    # ---- model ----
    lfcc = build_lfcc(cfg, comp).to(DEVICE)
    model = LCNN(n_feat=lfcc.out_dim, embedding_dim=cfg["model"]["embedding_dim"],
                 dropout=cfg["model"].get("dropout", 0.7)).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    w = class_weights(train_ds.df, tr.get("class_weight", "none"))
    print(f"parameter model {n_par:,} | dim fitur {lfcc.out_dim}"
          + (f" | bobot kelas {w.tolist()}" if w is not None else " | tanpa bobot kelas"))

    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = make_scheduler(opt, cfg, total_steps=epochs * max(1, len(train_ld)))
    grad_clip = tr["grad_clip"]

    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["paths"]["checkpoints_dir"], f"{tag}.pt")

    best_eer, best_epoch, bad = 1.0, -1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, running, nb = time.time(), 0.0, 0
        n_batch = len(train_ld)
        lapor = max(1, n_batch // 10)          # indikator progres tiap 10%
        for wav, label, _, _ in train_ld:
            wav, label = wav.to(DEVICE), label.to(DEVICE)
            logits, _ = model(lfcc(wav))
            loss = cross_entropy(logits, label, weight=w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            running += loss.item(); nb += 1
            if nb % lapor == 0 or nb == n_batch:
                laju = nb / max(time.time() - t0, 1e-9)
                sisa = (n_batch - nb) / max(laju, 1e-9)
                print(f"   epoch {epoch:3d} [{nb:4d}/{n_batch}] {100*nb//n_batch:3d}% "
                      f"| loss {running/nb:.4f} | {laju*bs:5.1f} berkas/s "
                      f"| sisa ~{sisa/60:4.1f} m", flush=True)
        print(f"   epoch {epoch:3d} validasi ...", flush=True)
        y, s = run_inference(model, lfcc, val_ld)
        m = compute_metrics(y, s)
        print(f"epoch {epoch:3d} | loss {running/max(nb,1):.4f} | val_EER {m['eer']*100:5.2f}% "
              f"| val_AUC {m['auc']:.3f} | lr {opt.param_groups[0]['lr']:.2e} "
              f"| {time.time()-t0:.0f}s")

        if m["eer"] < best_eer:
            best_eer, best_epoch, bad = m["eer"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                        "epoch": epoch, "val_eer": best_eer,
                        # ambang tetap untuk F1 dan balanced accuracy (Subbab 2.8.2),
                        # ditentukan dari partisi validasi, bukan dari data uji
                        "val_threshold": m["eer_threshold"],
                        "config": cfg, "seed": seed,
                        "augment": augment, "components": comp,
                        "augment_disable": aug_cfg.get("disable", []),
                        "n_feat": lfcc.out_dim, "tag": tag}, ckpt_path)
            print(f"   checkpoint tersimpan (EER {best_eer*100:.2f}%) -> {ckpt_path}")
        else:
            bad += 1
            if bad >= patience:
                print(f"early stopping (patience {patience}) pada epoch {epoch}")
                break

    summary = os.path.join(cfg["paths"]["results_dir"], "run_summary.csv")
    head = not os.path.exists(summary)
    with open(summary, "a") as f:
        if head:
            f.write("model,seed,augment,components,augment_disable,best_epoch,validation_eer,test_eer\n")
        f.write(f"{tag},{seed},{augment},{comp},"
                f"\"{'|'.join(aug_cfg.get('disable', []))}\",{best_epoch},{best_eer:.5f},\n")
    print(f"SELESAI {tag} | val_EER terbaik {best_eer*100:.2f}% @ epoch {best_epoch} -> {summary}")


if __name__ == "__main__":
    main()
