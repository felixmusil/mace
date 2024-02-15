from e3nn.util.jit import compile_mode
import numpy as np
import torch
from torch import nn

class BesselBasis(torch.nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


@compile_mode("script")
class GaussianBasis(torch.nn.Module):
    """
    Gaussian basis functions
    """

    def __init__(self, r_max: float, num_basis=128, trainable=False):
        super().__init__()
        gaussian_weights = torch.linspace(
            start=0.0, end=r_max, steps=num_basis, dtype=torch.get_default_dtype()
        )
        if trainable:
            self.gaussian_weights = torch.nn.Parameter(
                gaussian_weights, requires_grad=True
            )
        else:
            self.register_buffer("gaussian_weights", gaussian_weights)
        self.coeff = -0.5 / (r_max / (num_basis - 1)) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        x = x - self.gaussian_weights
        return torch.exp(self.coeff * torch.pow(x, 2))


class ExpNormalBasis(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_basis: int = 50,
        trainable: bool = False,
    ):
        super().__init__()

        self.r_max = r_max

        self.num_basis = num_basis
        self.trainable = trainable
        self.alpha = 5.0 / self.r_max

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        r"""Method for initializing the basis function parameters, as described in
        https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181 .
        """

        start_value = torch.exp(
            torch.scalar_tensor(
                -self.r_max
            )
        )
        means = torch.linspace(start_value, 1, self.num_basis)
        betas = torch.tensor(
            [(2 / self.num_basis * (1 - start_value)) ** -2] * self.num_basis
        )
        return means, betas

    def reset_parameters(self):
        r"""Method to reset the parameters of the basis functions to their
        initial values.
        """
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.betas)}, "
            f"trainable={self.betas.requires_grad})"
        )

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        r"""Expansion of distances through the radial basis function set.
        Parameters
        ----------
        dist: torch.Tensor
            Input pairwise distances of shape (total_num_edges)
        Return
        ------
        expanded_distances: torch.Tensor
            Distances expanded in the radial basis with shape (total_num_edges, num_basis)
        """

        dist = dist.unsqueeze(-1)
        return torch.exp(
            -self.betas
            * (
                torch.exp(self.alpha * (-dist))
                - self.means
            )
            ** 2
        ) # [..., num_basis]


@compile_mode("script")
class PolynomialCutoff(torch.nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (8)
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # yapf: disable
        envelope = (
                1.0
                - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
                + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
                - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
        )
        # yapf: enable

        # noinspection PyUnresolvedReferences
        return envelope * (x < self.r_max).type(torch.get_default_dtype())

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"

class CosineCutoff(torch.nn.Module):

    def __init__(self, r_max: float):
        super().__init__()
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cutoffs = 0.5 * (
            torch.cos(x * torch.pi / self.r_max) + 1.0
        )
        # remove contributions beyond the cutoff radius
        cutoffs = cutoffs * (x < self.r_max).to(
            x.dtype
        )
        return cutoffs

    def __repr__(self):
        return f"{self.__class__.__name__}(r_max={self.r_max})"
