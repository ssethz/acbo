import torch
from botorch import fit_gpytorch_model
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.polynomial_kernel import PolynomialKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.additive_structure_kernel import AdditiveStructureKernel
import pickle 


def fit_gp_model(X, Y):
    if Y.ndim == 1:
        Y = Y.unsqueeze(dim=-1)
    kernel = ScaleKernel(MaternKernel())
    
    model = SingleTaskGP(
        X, Y,
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=X.shape[-1]), covar_module = kernel
    )
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    return model
