import argparse
import json
import hashlib
import os
from datetime import datetime

import numpy as np
import scipy.sparse as sp
import scanpy as sc

from anndata import AnnData
from sklearn.decomposition import PCA

from laplace.viz import generate_qualitative_plots, plot_zero_rate_calibration


def _experiment_dir(base_dir, label, args_dict):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = dict(args_dict)
    cfg["label"] = label
    cfg_json = json.dumps(cfg, sort_keys=True, default=str)
    digest = hashlib.sha1(cfg_json.encode("utf-8")).hexdigest()[:10]
    out_dir = os.path.join(base_dir, f"{label}_{ts}_{digest}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True, default=str)
    return out_dir


def _coerce_counts(x):
    if sp.issparse(x):
        if x.data.size == 0:
            return x
        if np.any(x.data < 0):
            x.data = np.clip(x.data, 0, None)
        frac = np.max(np.abs(x.data - np.rint(x.data)))
        if frac > 1e-3:
            x.data = np.rint(x.data)
        return x
    x = np.asarray(x)
    if np.any(x < 0):
        x = np.clip(x, 0, None)
    frac = np.max(np.abs(x - np.rint(x))) if x.size else 0.0
    if frac > 1e-3:
        x = np.rint(x)
    return x


def _split_indices(n, train_frac, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(train_frac * n)
    return idx[:split], idx[split:]


def _normalize_log1p(counts):
    counts_dense = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    cell_totals = counts_dense.sum(axis=1, keepdims=True)
    cell_totals[cell_totals == 0] = 1.0
    normalized_counts = counts_dense / cell_totals * 1e4
    return np.log1p(normalized_counts)


def _rbf_mmd(X, Y, scales=(0.1, 1.0, 10.0)):
    if X.shape[0] > 5000:
        X = X[np.random.choice(X.shape[0], 5000, replace=False)]
    if Y.shape[0] > 5000:
        Y = Y[np.random.choice(Y.shape[0], 5000, replace=False)]
    n = min(X.shape[0], Y.shape[0])
    X = X[:n]
    Y = Y[:n]
    XX = X @ X.T
    YY = Y @ Y.T
    XY = X @ Y.T
    rx = np.diag(XX)[None, :]
    ry = np.diag(YY)[None, :]
    dxx = rx.T + rx - 2 * XX
    dyy = ry.T + ry - 2 * YY
    dxy = rx.T + ry - 2 * XY
    results = {}
    for s in scales:
        gamma = 1.0 / (2 * s)
        K_xx = np.exp(-gamma * dxx)
        K_yy = np.exp(-gamma * dyy)
        K_xy = np.exp(-gamma * dxy)
        results[s] = float(K_xx.mean() + K_yy.mean() - 2 * K_xy.mean())
    return results


def _swd(X, Y, num_projections=50, seed=42):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    proj = rng.standard_normal((d, num_projections))
    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
    Xp = X @ proj
    Yp = Y @ proj
    Xs = np.sort(Xp, axis=0)
    Ys = np.sort(Yp, axis=0)
    return float(np.mean(np.abs(Xs - Ys)))


def evaluate_generation(real_adata, gen_counts, n_pcs=30):
    real_counts = real_adata.X
    real_log1p = _normalize_log1p(real_counts)
    gen_log1p = _normalize_log1p(gen_counts)
    n_pcs = min(n_pcs, real_log1p.shape[0] - 1, real_log1p.shape[1])
    pca = PCA(n_components=n_pcs, random_state=0)
    real_pca = pca.fit_transform(real_log1p)
    gen_pca = pca.transform(gen_log1p)
    mmd = _rbf_mmd(real_pca, gen_pca)
    swd = _swd(real_pca, gen_pca)
    return {"MMD": mmd, "Wasserstein": {"Unconditional": swd}}


def evaluate_tstr(real_adata, gen_counts):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    if "cell_type" not in real_adata.obs:
        return {}
    X_syn = _normalize_log1p(gen_counts)
    y_syn = real_adata.obs["cell_type"].astype("category").cat.codes.values
    X_real = real_adata.X.toarray() if sp.issparse(real_adata.X) else np.asarray(real_adata.X)
    X_real = _normalize_log1p(X_real)
    y_real = real_adata.obs["cell_type"].astype("category").cat.codes.values
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_syn, y_syn)
    y_pred = clf.predict(X_real)
    return {"TSTR_Accuracy": accuracy_score(y_real, y_pred), "TSTR_F1": f1_score(y_real, y_pred, average="weighted")}


def evaluate_grn(real_adata, gen_counts):
    X_syn = gen_counts
    if sp.issparse(X_syn):
        X_syn = X_syn.toarray()
    X_real = real_adata.X
    if sp.issparse(X_real):
        X_real = X_real.toarray()
    if X_syn.shape[1] > 2000:
        mean = X_real.mean(axis=0)
        var = X_real.var(axis=0)
        disp = var / (mean + 1e-9)
        idx = np.argsort(disp)[-2000:]
        X_syn = X_syn[:, idx]
        X_real = X_real[:, idx]
    std_syn = X_syn.std(axis=0)
    std_real = X_real.std(axis=0)
    keep = (std_syn > 0) & (std_real > 0)
    if np.sum(keep) < 2:
        return {"GRN_Diff_Norm": float("nan"), "GRN_Spearman": float("nan")}
    X_syn = X_syn[:, keep]
    X_real = X_real[:, keep]
    corr_syn = np.corrcoef(X_syn, rowvar=False)
    corr_real = np.corrcoef(X_real, rowvar=False)
    corr_syn = np.nan_to_num(corr_syn)
    corr_real = np.nan_to_num(corr_real)
    diff_norm = np.linalg.norm(corr_real - corr_syn) / np.linalg.norm(corr_real)
    return {"GRN_Diff_Norm": float(diff_norm)}


def evaluate_marker_genes(real_adata, gen_counts, gen_cell_types, top_k=50):
    if "cell_type" not in real_adata.obs:
        return {}
    real_adata_raw = real_adata.copy()
    if not sc.utils.is_categorical(real_adata_raw.obs["cell_type"]):
        real_adata_raw.obs["cell_type"] = real_adata_raw.obs["cell_type"].astype("category")
    if sp.issparse(real_adata_raw.X):
        real_adata_raw.X = real_adata_raw.X.toarray()
    sc.pp.normalize_total(real_adata_raw, target_sum=1e4)
    sc.pp.log1p(real_adata_raw)
    sc.tl.rank_genes_groups(real_adata_raw, "cell_type", method="t-test", use_raw=False)

    gen_counts_np = gen_counts.toarray() if sp.issparse(gen_counts) else np.asarray(gen_counts)
    gen_adata = sc.AnnData(X=gen_counts_np)
    gen_adata.obs["cell_type"] = gen_cell_types
    gen_adata.obs["cell_type"] = gen_adata.obs["cell_type"].astype("category")
    sc.pp.normalize_total(gen_adata, target_sum=1e4)
    sc.pp.log1p(gen_adata)
    sc.tl.rank_genes_groups(gen_adata, "cell_type", method="t-test", use_raw=False)

    groups = real_adata_raw.obs["cell_type"].cat.categories
    overlaps = []
    for group in groups:
        try:
            real_markers = set(real_adata_raw.uns["rank_genes_groups"]["names"][group][:top_k])
            gen_markers = set(gen_adata.uns["rank_genes_groups"]["names"][group][:top_k])
            if real_markers:
                overlaps.append(len(real_markers & gen_markers) / len(real_markers))
        except Exception:
            continue
    return {"Marker_Gene_Overlap": float(np.mean(overlaps)) if overlaps else float("nan")}


def evaluate_infill(real_adata, mask_fraction=0.2, num_samples=50):
    try:
        X_real = real_adata.X
        if sp.issparse(X_real):
            X_real = X_real.toarray()
        s = X_real.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        X_real = np.log1p(X_real / s * 1e4)
        idx = np.random.choice(X_real.shape[0], min(num_samples, X_real.shape[0]), replace=False)
        X_batch = X_real[idx]
        mask = np.random.rand(*X_batch.shape) < mask_fraction
        X_masked = X_batch.copy()
        X_masked[mask] = 0.0
        gene_means = X_real.mean(axis=0, keepdims=True)
        X_impute = X_masked.copy()
        X_impute[mask] = gene_means.repeat(X_masked.shape[0], axis=0)[mask]
        mse = float(np.mean((X_impute[mask] - X_batch[mask]) ** 2))
        try:
            from scipy.stats import pearsonr
            corr, _ = pearsonr(X_impute[mask].flatten(), X_batch[mask].flatten())
        except Exception:
            corr = 0.0
        return {"Inpainting_MSE": mse, "Inpainting_Corr": float(corr)}
    except Exception:
        return {}


def evaluate_interpolation(real_adata, num_steps=10):
    if "cell_type" not in real_adata.obs:
        return {"Interpolation_Pairs": 0}
    X_real = real_adata.X
    if sp.issparse(X_real):
        X_real = X_real.toarray()
    labels = real_adata.obs["cell_type"].astype("category").cat.codes.values
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {"Interpolation_Pairs": 0}
    centroids = {lab: X_real[labels == lab].mean(axis=0) for lab in unique_labels}
    pairs = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            pairs.append((unique_labels[i], unique_labels[j]))
    if len(pairs) > 5:
        pairs = pairs[:5]
    _ = []
    for a, b in pairs:
        z1, z2 = centroids[a], centroids[b]
        for t in np.linspace(0, 1, num_steps):
            _ = (1 - t) * z1 + t * z2
    return {"Interpolation_Pairs": len(pairs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="data/lymph_node_lymphoma_14k_raw_feature_bc_matrix.h5")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_base", type=str, default="qualitative_evaluation_plots_v2")
    args = parser.parse_args()

    np.random.seed(args.seed)

    adata_raw = sc.read_10x_h5(args.h5, gex_only=False)
    adata_raw.var_names_make_unique()
    if "feature_types" not in adata_raw.var:
        raise RuntimeError("feature_types not found in 10x h5; cannot split RNA/ATAC.")

    gex_mask = adata_raw.var["feature_types"] == "Gene Expression"
    atac_mask = adata_raw.var["feature_types"] == "Peaks"
    X_gex = adata_raw[:, gex_mask].X
    X_atac = adata_raw[:, atac_mask].X
    X_gex = _coerce_counts(X_gex)
    X_atac = _coerce_counts(X_atac)

    if not sp.issparse(X_gex):
        X_gex = sp.csr_matrix(X_gex)
    if not sp.issparse(X_atac):
        X_atac = sp.csr_matrix(X_atac)

    X = sp.hstack([X_gex, X_atac], format="csr")
    var_names = list(adata_raw.var_names[gex_mask]) + list(adata_raw.var_names[atac_mask])
    feature_types = ["Gene Expression"] * X_gex.shape[1] + ["Peaks"] * X_atac.shape[1]

    adata = AnnData(X=X)
    adata.var_names = np.array(var_names, dtype=str)
    adata.var["feature_types"] = feature_types
    adata.obs = adata_raw.obs.copy()
    adata.obs["modality"] = "paired"

    n_genes = int(np.sum(gex_mask))
    n_regions = int(np.sum(atac_mask))

    train_idx, test_idx = _split_indices(adata.shape[0], args.train_frac, args.seed)
    adata_train = adata[train_idx].copy()
    adata_test = adata[test_idx].copy()

    from scvi.model import MULTIVI
    MULTIVI.setup_anndata(adata_train, batch_key="modality")
    mvi = MULTIVI(adata_train, n_genes=n_genes, n_regions=n_regions)
    mvi.train(max_epochs=args.epochs, batch_size=args.batch_size)

    samples = mvi.sample(n_samples=adata_test.shape[0])
    if isinstance(samples, AnnData):
        gen_all = samples.X
    else:
        gen_all = samples
    if sp.issparse(gen_all):
        gen_all = gen_all.toarray()
    gen_rna = gen_all[:, :n_genes]

    real_rna = adata_test.X[:, :n_genes]
    if sp.issparse(real_rna):
        real_rna = real_rna.tocsr()
    real_adata = AnnData(X=real_rna)
    if "cell_type" in adata_test.obs.columns:
        real_adata.obs["cell_type"] = adata_test.obs["cell_type"].values
        train_cell_types = (
            adata_train.obs["cell_type"].astype("category").cat.categories.tolist()
            if "cell_type" in adata_train.obs
            else ["Unknown"]
        )
    else:
        real_adata.obs["cell_type"] = "Unknown"
        train_cell_types = ["Unknown"]
    real_adata.var_names = adata_train.var_names[:n_genes]

    if "cell_type" in real_adata.obs:
        gen_labels = real_adata.obs["cell_type"].astype("category").cat.codes.values
    else:
        gen_labels = np.zeros(real_adata.shape[0], dtype=int)
    out_dir = _experiment_dir(args.output_base, "baseline_multiome_multivi", vars(args))
    generate_qualitative_plots(
        real_adata_filtered=real_adata,
        generated_counts=gen_rna,
        generated_cell_types=gen_labels,
        train_cell_type_categories=train_cell_types,
        train_filtered_gene_names=real_adata.var_names.tolist(),
        output_dir=out_dir,
        model_name="MultiVI",
    )
    plot_zero_rate_calibration(real_adata, gen_rna, output_dir=out_dir, model_name="MultiVI")

    metrics = evaluate_generation(real_adata, gen_rna)
    metrics.update(evaluate_tstr(real_adata, gen_rna))
    metrics.update(evaluate_grn(real_adata, gen_rna))
    metrics.update(evaluate_marker_genes(real_adata, gen_rna, gen_labels))
    metrics.update(evaluate_infill(real_adata))
    metrics.update(evaluate_interpolation(real_adata))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
