
def set_deterministic_training(seed: int, rank: int = 0):
    import random
    import numpy as np
    import torch
    import os
    rseed = seed+rank
    # Python 随机种子
    random.seed(rseed)
    # Numpy 随机种子
    np.random.seed(rseed)
    # Pytorch
    torch.manual_seed(rseed)
    torch.cuda.manual_seed_all(rseed)
    # Env
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


