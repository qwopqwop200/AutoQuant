import math
import scipy
import time

import torch
import torch.nn as nn
import transformers

from auto_quant.quant.gptq.quantize.quantizer import Quantizer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class RTN:
    def __init__(self, layer):
        self.layer = layer
        self.quantizer = Quantizer()

    def fasterquant(self,):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)
        
        Q = self.quantizer.quantize(W).to(self.layer.weight.data.dtype)
        
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q
        return self.quantizer.scale, self.quantizer.zero, None