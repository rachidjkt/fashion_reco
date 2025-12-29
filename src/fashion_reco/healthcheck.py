import sys
import torch
from fashion_reco.config import DATA_DIR

def main():
    print("Python:", sys.version.split()[0])
    print("Torch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    print("Data dir:", DATA_DIR.resolve())

if __name__ == "__main__":
    main()