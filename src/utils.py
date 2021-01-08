from contextlib import contextmanager
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


@contextmanager
def timer(message: str):
    print(f'[{message}] start.')
    t0 = time.time()
    yield
    elapsed_time = time.time() - t0
    print(f'[{message}] done in {elapsed_time / 60:.1f} min.')


def set_seed(seed: int = 1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
