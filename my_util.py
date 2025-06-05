from torch import cuda
from torch.backends import mps

DEVICE = "cuda" if cuda.is_available() else \
          "mps" if mps.is_available() else "cpu"
