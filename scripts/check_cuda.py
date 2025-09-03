import torch

def check_cuda():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU device name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected or CUDA not available.")

if __name__ == "__main__":
    check_cuda()
