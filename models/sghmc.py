from typing import Any, Callable, List, Optional

import numpy as np
import torch
from botorch.models.model import Model
from botorch.posteriors import Posterior
from torch import Tensor

from pybnn.bohamiann import Bohamiann
from pybnn.util.layers import AppendLayer

from .utils import RegNet


class BohamiannPosterior(Posterior):
    def __init__(self, X, model):
        super().__init__()
        _, _, samples = model.predict(X, return_individual_predictions=True)
        self.preds = samples

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
    ) -> Tensor:
        n = len(self.preds)
        rand_ints = np.random.randint(n, size=sample_shape)
        return self.preds[rand_ints]

    @property
    def mean(self) -> Tensor:
        r"""The posterior mean."""
        return self.preds.mean(axis=0)

    @property
    def variance(self) -> Tensor:
        r"""The posterior variance."""
        return self.preds.var(axis=0)

    def _extended_shape(
        self, sample_shape: torch.Size = torch.Size()
    ) -> torch.Size:
        r"""Returns the shape of the samples produced by the posterior with
        the given `sample_shape`.
        """
        return sample_shape + self.preds.shape

    @property
    def device(self) -> torch.device:
        return self.preds.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the distribution."""
        return self.preds.dtype



class BohamiannModel(Model):

    def __init__(self, model_args, input_dim, output_dim, device):
        super().__init__()
        self.gp = None
        self.output_dim = output_dim
        self.device = device
        self.model_args = model_args

        self.input_dim = input_dim
        self.problem_output_dim = output_dim

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        return BohamiannPosterior(X, self.model)

    @property
    def num_outputs(self) -> int:
        return self.output_dim

    def fit_and_save(self, train_x, train_y, save_dir):
        def get_network(input_dimensionality, output_dim, device) -> torch.nn.Module:
            regnet = RegNet(dimensions=self.model_args["regnet_dims"],
                        activation=self.model_args["regnet_activation"],
                        input_dim=input_dimensionality,
                        output_dim=output_dim,
                        dtype=torch.float64,
                        device=device)
            last_layer = AppendLayer(noise=1e-3, device=device)
            return torch.nn.Sequential(regnet, last_layer)
        
        self.model = Bohamiann(get_network, print_every_n_steps=1000, device=self.device)
        self.model.train(train_x, train_y, num_steps=2000, num_burn_in_steps=1000, keep_every=5, lr=1e-2, batch_size=5, verbose=False)