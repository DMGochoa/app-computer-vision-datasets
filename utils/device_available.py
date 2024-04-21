import torch

def device_available():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"