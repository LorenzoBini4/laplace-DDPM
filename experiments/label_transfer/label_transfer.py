import argparse
import os
import numpy as np
import scanpy as sc
import scvi

def _read_query(path):
    if path.endswith(".h5ad"):
        return sc.read_h5ad(path)
    return sc.read_10x_h5(path)

def _harmonize_gene_ids(ref, query):
    # Prefer gene symbols when available in reference
    if "feature_name" in ref.var.columns:
        ref.var_names = ref.var["feature_name"].astype(str)
    ref.var_names_make_unique()
    # Query var_names are usually gene symbols; make unique defensively
    query.var_names_make_unique()

def _subsample(adata, n):
    if n <= 0 or adata.n_obs <= n:
        return adata
    idx = np.random.choice(adata.n_obs, size=n, replace=False)
    return adata[idx].copy()

def main():
    parser = argparse.ArgumentParser(description="Transfer cell-type labels using scVI/SCANVI.")
    parser.add_argument("--reference_h5ad", type=str, required=True)
    parser.add_argument("--query_path", type=str, required=True)
    parser.add_argument("--label_key", type=str, default="cell_type")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_tsv", action="store_true", default=False)
    parser.add_argument("--reference_ncells", type=int, default=100000)
    parser.add_argument("--n_latent", type=int, default=30)
    parser.add_argument("--max_epochs_ref", type=int, default=50)
    parser.add_argument("--max_epochs_scanvi", type=int, default=20)
    parser.add_argument("--max_epochs_query", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()

    scvi.settings.seed = args.seed

    ref = sc.read_h5ad(args.reference_h5ad)
    if args.label_key not in ref.obs.columns:
        raise ValueError(f"label_key '{args.label_key}' not found in reference obs.")
    ref = _subsample(ref, args.reference_ncells)

    query = _read_query(args.query_path)

    _harmonize_gene_ids(ref, query)

    common_genes = ref.var_names.intersection(query.var_names)
    if common_genes.size == 0:
        raise ValueError("No overlapping genes between reference and query.")
    ref = ref[:, common_genes].copy()
    query = query[:, common_genes].copy()

    scvi.model.SCVI.setup_anndata(ref, labels_key=args.label_key)
    scvi_model = scvi.model.SCVI(ref, n_latent=args.n_latent)
    # scvi-tools version compatibility: some versions don't accept data_loader_kwargs
    scvi.settings.dl_num_workers = args.num_workers
    scvi.settings.dl_pin_memory = args.pin_memory
    scvi_model.train(max_epochs=args.max_epochs_ref, batch_size=args.batch_size)

    # Handle scvi-tools API differences across versions
    try:
        scanvi_model = scvi.model.SCANVI.from_scvi_model(
            scvi_model, labels_key=args.label_key, unlabeled_category="Unknown"
        )
    except TypeError:
        # Older versions expect labels_key in setup_anndata instead of constructor
        scvi.model.SCANVI.setup_anndata(ref, labels_key=args.label_key)
        scanvi_model = scvi.model.SCANVI.from_scvi_model(
            scvi_model, unlabeled_category="Unknown"
        )
    scanvi_model.train(max_epochs=args.max_epochs_scanvi, batch_size=args.batch_size)

    # Ensure label column exists in query for scvi transfer
    if args.label_key not in query.obs.columns:
        query.obs[args.label_key] = "Unknown"
        query.obs[args.label_key] = query.obs[args.label_key].astype("category")

    scanvi_query = scvi.model.SCANVI.load_query_data(query, scanvi_model)
    scanvi_query.train(max_epochs=args.max_epochs_query, batch_size=args.batch_size)
    pred = scanvi_query.predict(query)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    sep = "\t" if args.output_tsv else ","
    with open(args.output_csv, "w", encoding="utf-8") as f:
        f.write(f"barcode{sep}cell_type\n")
        for bc, ct in zip(query.obs_names.astype(str), pred.astype(str)):
            f.write(f"{bc}{sep}{ct}\n")

    print(f"Saved labels to {args.output_csv}")


if __name__ == "__main__":
    main()
