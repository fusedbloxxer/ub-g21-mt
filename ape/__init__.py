import os
import pathlib as pb


# Configure paths
ROOT_DIR = pb.Path('.')
CACHE_DIR = ROOT_DIR / '..' / 'cache'
HF_CACHE_DIR = CACHE_DIR / 'huggingface'
DATA_DIR = ROOT_DIR / '..' / 'data'

# Explicitly cache HuggingFace at desired path
os.environ['HF_HOME'] = str(HF_CACHE_DIR)
os.environ['HF_HUB_CACHE'] = str(HF_CACHE_DIR)
