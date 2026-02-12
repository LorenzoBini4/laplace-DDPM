import os
import runpy
import sys

_BOOL_ARGS = {
    "bio_positional",
    "disable_kl",
    "dual_graph",
    "early_stop_gene_stats",
    "eval_only",
    "force_reprocess",
    "gen_calibrate_means",
    "gen_calibrate_zero",
    "gen_match_zero_per_cell",
    "scvi_train",
    "simple_mode",
    "stage2_freeze_decoder",
    "use_pseudo_labels",
    "use_scgen_context",
    "use_scvi_context",
    "use_wandb",
    "viz",
}

def _normalize_bool_args(argv):
    normalized = []
    for arg in argv:
        if not arg.startswith("--") or "=" not in arg:
            normalized.append(arg)
            continue
        key, val = arg[2:].split("=", 1)
        if key not in _BOOL_ARGS:
            normalized.append(arg)
            continue
        v = val.strip().lower()
        if v in ("true", "1", "yes", "y", "t"):
            normalized.append(f"--{key}")
        elif v in ("false", "0", "no", "n", "f", ""):
            os.environ[f"BOOL_OVERRIDE_{key}"] = "false"
        else:
            normalized.append(arg)
    return normalized

def _resolve_target(repo_root: str) -> str:
    # Select experiment via CLI: --experiment=spatial|multiome
    # or env: SWEEP_EXPERIMENT=spatial|multiome
    exp = None
    for arg in list(sys.argv[1:]):
        if arg.startswith("--experiment="):
            exp = arg.split("=", 1)[1].strip().lower()
            sys.argv.remove(arg)
            break
    if exp is None:
        exp = os.environ.get("SWEEP_EXPERIMENT", "spatial").strip().lower()
    if exp not in ("spatial", "multiome"):
        raise ValueError(f"Unsupported experiment '{exp}'. Use 'spatial' or 'multiome'.")
    if exp == "multiome":
        return os.path.join(repo_root, "experiments", "multiome", "main_multiome.py")
    return os.path.join(repo_root, "experiments", "spatial", "main_spatial.py")

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sys.argv[1:] = _normalize_bool_args(sys.argv[1:])
    target = _resolve_target(repo_root)
    if not os.path.exists(target):
        raise FileNotFoundError(f"Expected training entrypoint not found: {target}")
    runpy.run_path(target, run_name="__main__")

if __name__ == "__main__":
    main()
