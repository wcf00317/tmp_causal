import torch
import random
import numpy as np
import os


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure that computations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', filename='checkpoint.pth.tar'):
    """
    Saves a model checkpoint.
    Args:
        state (dict): Contains model's state_dict, optimizer's state_dict, epoch, etc.
        is_best (bool): If true, saves a copy as 'model_best.pth.tar'.
        checkpoint_dir (str): Directory to save checkpoints.
        filename (str): The name of the checkpoint file.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)
        print(f"==> Saved new best model to {best_filepath}")