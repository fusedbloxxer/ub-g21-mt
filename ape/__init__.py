import os
import random
import torch
from torch import Generator, device
import numpy as np
import pathlib as pb
import mlflow


# Configure paths
ROOT_DIR = pb.Path('.')
CACHE_DIR = ROOT_DIR / '..' / 'cache'
HF_CACHE_DIR = CACHE_DIR / 'huggingface'
DATA_DIR = ROOT_DIR / '..' / 'data'

# Explicitly cache HuggingFace at desired path
os.environ['HF_HOME'] = str(HF_CACHE_DIR)
os.environ['HF_HUB_CACHE'] = str(HF_CACHE_DIR)

# Environment
SEED: int = 80
DEVICE: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PREFETCH_FACTOR = None
BATCH_SIZE = 4
WORKERS = 0

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_float32_matmul_precision('medium')
gen_torch = Generator().manual_seed(SEED)

# For custom hardware
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# MLFlow Local Logging
MLFLOW_TRACKING_URI = uri='http://127.0.0.1:8080'
