import torch


def torch_get_device():
    if torch.cuda.is_available():
        print("Cuda device available.. Using Cuda as primary device")
        return torch.device("cuda")
    else:
        print("Cuda device unavailable.. Using Cpu as primary device")
        return torch.device("cpu")
