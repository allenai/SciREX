from allennlp.nn.activations import Activation
import torch

@Activation.register("gelu")
class GELU(Activation) :
    def __call__(self, tensor) :
        return torch.nn.functional.gelu(tensor)