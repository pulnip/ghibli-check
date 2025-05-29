from torch import cuda
from torch.backends import mps
import sys

DEVICE = "cuda" if cuda.is_available() else \
          "mps" if mps.is_available() else "cpu"

def get_argv(index: int, else_val):
    return sys.argv[index] if len(sys.argv) > index else else_val

def get_argvs(start: int, end: int=None):
    end = len(sys.argv) if end==None else end
    end = len(sys.argv) if end > len(sys.argv) else end
    assert start < end

    argvs = []
    for i in range(start, end):
        argvs.append(sys.argv[i])

    return argvs
