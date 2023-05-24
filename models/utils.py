import torch

class RegNet(torch.nn.Sequential):
    def __init__(self, dimensions, activation, input_dim=1, output_dim=1,
                        dtype=torch.float64, device="cpu"):
        super(RegNet, self).__init__()
        self.dimensions = [input_dim, *dimensions, output_dim]
        for i in range(len(self.dimensions) - 1):
            self.add_module('linear%d' % i, torch.nn.Linear(
                self.dimensions[i], self.dimensions[i + 1], dtype=dtype, device=device)
            )
            if i < len(self.dimensions) - 2:
                if activation == "tanh":
                    self.add_module('tanh%d' % i, torch.nn.Tanh())
                elif activation == "relu":
                    self.add_module('relu%d' % i, torch.nn.ReLU(inplace=True))
                else:
                    raise NotImplementedError("Activation type %s is not supported" % activation)