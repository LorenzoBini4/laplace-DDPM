import random
import numpy as np
import torch

torch.backends.cudnn.benchmark = True
# Attempt to set device, fall back to CPU if CUDA is not available or fails
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except Exception as e:
    print(f"Error setting up CUDA device: {e}. Falling back to CPU.")
    device = torch.device('cpu')

print(f"Using device: {device}")
def set_seed(seed):
    """Set random seeds for reproducibility."""
    if seed is not None:
        print(f"Setting random seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        print("No seed provided, using default random state.")
        # No seed set, will use default random state