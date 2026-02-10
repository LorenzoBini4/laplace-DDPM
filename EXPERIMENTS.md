# Experiments Guide

This document lists the main experiment entrypoints and where outputs are stored after the repo restructure.

## Folder Map

- `experiments/spatial/` — spatial (Visium) training, generation, logs
- `experiments/multiome/` — multiome training, generation, logs
- `experiments/baselines/` — scVI/scANVI and MultiVI baselines
- `experiments/label_transfer/` — label transfer (HLCA → datasets)
- `experiments/sweeps/` — hyperparameter sweeps (SLURM arrays)
- `experiments/unimodal/` — PBMC3K / Dentate scripts
- `experiments/qualitative_evaluation_plots_v2/` — qualitative plots

## Spatial (Visium)

Train:
```
sbatch experiments/spatial/slurm/train_spatial.slurm
```

Generate/eval only:
```
sbatch experiments/spatial/slurm/generate_spatial.slurm
```

Outputs:
- Logs: `experiments/spatial/logs/`
- Plots: `experiments/qualitative_evaluation_plots_v2/`
- Processed data: `data/processed/spatial/`

## Multiome (10x 14k)

Train:
```
sbatch experiments/multiome/slurm/train_multiome.slurm
```

Generate/eval only:
```
sbatch experiments/multiome/slurm/generate_multiome.slurm
```

Outputs:
- Logs: `experiments/multiome/logs/`
- Plots: `experiments/qualitative_evaluation_plots_v2/`
- Processed data: `data/processed/multiome/`

## Label Transfer (Cell Types)

```
sbatch experiments/label_transfer/slurm/label_transfer_spatial.slurm
sbatch experiments/label_transfer/slurm/label_transfer_multiome.slurm
```

Outputs:
- `data/labels/spatial_cell_types.tsv`
- `data/labels/multiome_cell_types.tsv`

## Baselines

```
sbatch experiments/baselines/spatial/slurm/baseline_spatial_scvi_scanvi.slurm
sbatch experiments/baselines/multiome/slurm/baseline_multiome_multivi.slurm
```

Outputs:
- Logs: `experiments/baselines/*/logs/`
- Plots: `experiments/qualitative_evaluation_plots_v2/`

## Sweeps (Arrays)

```
sbatch experiments/sweeps/spatial/slurm/sweep_spatial.slurm
sbatch experiments/sweeps/multiome/slurm/sweep_multiome.slurm
```

Each array task saves plots tagged by the config in
`experiments/qualitative_evaluation_plots_v2/`.

## Unimodal (PBMC3K / Dentate)

```
sbatch experiments/unimodal/slurm/main.slurm
```
