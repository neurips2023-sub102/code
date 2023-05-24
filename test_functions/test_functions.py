import os
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
from botorch.test_functions import SyntheticTestFunction
from botorch.test_functions.base import MultiObjectiveTestProblem
from models.meta_module import *
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from torch import Tensor


class Toy(SyntheticTestFunction):

    dim = 1
    _optimal_value = 1.37552
    _bounds = [[-4.5, 4.5]]
    _optimizers = [-3.25962]

    def f(self, x):
        return (1.4 - 3 * x) * torch.sin(1.5 * x) / 8

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.stack([self.f(x).squeeze() for x in X])


class NonStationary(SyntheticTestFunction):

    dim = 1
    _optimal_value = 0.9232
    _bounds = [[-1.0, 1.0]]
    _optimizers = [-1.7677]

    def obj1(self, x):
        x = x * 5
        x = 1.5 * x
        y = (1.4 - 3 * x) * torch.sin(6 * x) / 8 + 0.25
        return y

    def obj2(self, x):
        x = x * 5
        y = 0.5 * torch.cos(x) + torch.sin(1.5 * x) / 8 + torch.sin(10 * x) / 4  + 0.25#-(1.4 - 3 * x) * torch.sin(1.5 * x) / 8 + torch.sin(10 * x) / 4 * x
        return y

    def obj(self, x, sig=0.01):
        if x > -2/5 and x < 1/5:
            y = -self.obj1(x)
        else:
            y = -self.obj2(x)
        return y + torch.randn((1,)).to(x) * sig

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.stack([self.obj(x).squeeze() for x in X])



class MetaRegNet(MetaSequential):
    def __init__(self, dimensions, activation, input_dim=1, output_dim=1, 
                        dtype=torch.float64, device="cpu"):
        super(MetaRegNet, self).__init__()
        self.dimensions = [input_dim, *dimensions, output_dim]
        for i in range(len(self.dimensions) - 1):
            self.add_module('linear%d' % i, MetaLinear(
                self.dimensions[i], self.dimensions[i + 1], dtype=dtype, device=device)
            )
            if i < len(self.dimensions) - 2:
                if activation == "tanh":
                    self.add_module('tanh%d' % i, torch.nn.Tanh())
                elif activation == "relu":
                    self.add_module('relu%d' % i, torch.nn.ReLU())
                else:
                    raise NotImplementedError("Activation type %s is not supported" % activation)


class BnnDraw(MultiObjectiveTestProblem):

    def __init__(
        self,
        input_dim,
        output_dim,
        seed,
        noise_std = None,
        negate: bool = False
    ) -> None:
        self.dim = input_dim
        self.num_objectives = output_dim
        self._bounds = np.repeat([[0, 1]], input_dim, axis=0)
        self._ref_point = np.ones(output_dim) * -2
        super().__init__(negate=negate, noise_std=noise_std)

        dimensions = [256, 256]
        activation = "tanh"
        self.model = MetaRegNet(dimensions=dimensions,
                                activation=activation,
                                input_dim=input_dim,
                                output_dim=output_dim,
                                dtype=torch.float64)
        path = "%s/bnn_params/%d_%d_%d_%s" % (os.getcwd(), input_dim, output_dim, seed, activation)
        for d in dimensions:
            path = path + "_%d" % d
        path = path + ".pt"
        print(path)
        if True and os.path.exists(path):
            self.params = torch.load(path)
            self.vector_to_parameters(self.params, self.model)
        else:
            param_size = len(torch.nn.utils.parameters_to_vector(self.model.state_dict().values()).detach())

            self.params = torch.distributions.Normal(torch.zeros(param_size), 1.0).sample()
            self.vector_to_parameters(self.params, self.model)
            torch.save(self.params, path)

    def vector_to_parameters(self, params, model):
        pointer = 0

        def get_module_by_name(module,
                            access_string: str):
            """Retrieve a module nested in another by its access string.

            Works even when there is a Sequential in the module.
            """
            names = access_string.split(sep='.')
            return reduce(getattr, names, module)
        
        for name, param in model.named_parameters():
            old_params = get_module_by_name(model, name)
            old_params = old_params + 1
            # The length of the parameter
            num_param = param.numel()

            new_params = params[pointer:pointer + num_param].view_as(param)
            exec("model." + name + " = new_params")
            
            # Increment the pointer
            pointer += num_param


    def f(self, x):
        return self.model(x)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self.model(X.cpu()).squeeze(-1).to(X)


class PolyDraw(SyntheticTestFunction):

    def __init__(
        self,
        input_dim,
        seed,
        noise_std = None,
        negate: bool = False
    ) -> None:
        self.dim = input_dim
        self.num_objectives = 1
        self._bounds = np.repeat([[0, 1]], input_dim, axis=0)
        super().__init__(negate=negate, noise_std=noise_std)

        path = "%s/bnn_params/poly_%d_%d.pt" % (os.getcwd(), input_dim, seed)
        if True and os.path.exists(path):
            self.coef = torch.load(path)
        else:
            self.coef = torch.rand((input_dim,)) * 2 - 1
            torch.save(self.coef, path)

    def f(self, x):
        return torch.pow(x, torch.range(len(x)))

    def evaluate_true(self, X: Tensor) -> Tensor:
        return (torch.pow(X, self.dim) * self.coef.to(X)).sum(-1)


class NNParams(SyntheticTestFunction):

    def __init__(
        self,
        noise_std = None,
        negate: bool = False
    ) -> None:
        
        param_path = "%s/nn_params.pt" % (os.getcwd())
        x_path = "%s/nn_params_x.pt" % (os.getcwd())
        model = MetaRegNet(dimensions=[32, 16],
                        activation="relu",
                        input_dim=2,
                        output_dim=1,
                        dtype=torch.float64)
        param_size = len(torch.nn.utils.parameters_to_vector(model.state_dict().values()))

        self.dim = param_size
        self.num_objectives = 1
        self._bounds = np.repeat([[0, 1]], self.dim, axis=0)
        super().__init__(negate=negate, noise_std=noise_std)

        self.model = model
        self.true_model = MetaRegNet(dimensions=[32, 16],
                            activation="relu",
                            input_dim=2,
                            output_dim=1,
                            dtype=torch.float64)
        
        if os.path.exists(param_path):
            params = torch.load(param_path)
            self.x = torch.load(x_path)
        else:
            params = torch.rand((param_size,))
            torch.save(params, param_path)

            self.x = torch.rand((100, 2))
            torch.save(self.x, x_path)

        self.vector_to_parameters(params, self.true_model)

    def vector_to_parameters(self, params, model):
        pointer = 0

        def get_module_by_name(module,
                            access_string: str):
            """Retrieve a module nested in another by its access string.

            Works even when there is a Sequential in the module.
            """
            names = access_string.split(sep='.')
            return reduce(getattr, names, module)
        
        for name, param in model.named_parameters():
            old_params = get_module_by_name(model, name)
            old_params = old_params + 1
            # The length of the parameter
            num_param = param.numel()

            new_params = params[pointer:pointer + num_param].view_as(param)
            exec("model." + name + " = new_params")
            
            # Increment the pointer
            pointer += num_param

    def f(self, params):
        self.vector_to_parameters(params, self.model)

        model_output = self.model(self.x)
        true_output = self.true_model(self.x)
        return nn.MSELoss()(model_output, true_output)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.stack([self.f(x.cpu()) for x in X]).to(X)