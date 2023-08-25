import torch
from torch import Tensor
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.optim import optimize_acqf
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)
import wandb

from mcbo.acquisition_function_optimization.custom_acqf_optimizer import (
    custom_optimize_acqf,
)

import itertools

def update_mw(
    acq_func,
    X_at,
    mw_actions,
    mw_weights,
    discrete_map,
    eta,
) -> Tensor:
    r'''
    Updates the MW weights using the acquisition function
    '''
    # to the end of all mw_actions, add the current X_at
    mw_actions = mw_actions.unsqueeze(1)
    # repeat X_at so that the shape is the same as mw_actions
    X_at = X_at.repeat(mw_actions.shape[0], mw_actions.shape[1], 1)
    cont_actions = discrete_map(mw_actions)
    cont_actions = torch.cat((cont_actions, X_at), dim=-1)
    values = acq_func(cont_actions)
    values = acq_func.model.outcome_transform.untransform(values)[0].squeeze(0)
    mw_weights = mw_weights * torch.exp(- eta * (1.0- torch.min(Tensor([1.0]), values)))
    mw_weights = mw_weights / torch.sum(mw_weights)
    return mw_weights


def optimize_discrete_acqf_and_get_best_point(
    acq_func,
    discrete: int,
    discrete_map,
    width: int,
) -> Tensor:
    r'''
    Exhaustively tries all discrete points in the domain and returns the best on for this acqisition function. 
    Width is the size of the input tensor.
    '''
    # iterate through all vectors of length width that have values less than discrete
    # and then map them to the discrete space
    
    discrete_points = torch.tensor(list(itertools.product(range(discrete), repeat=width)))
    discrete_points = discrete_points.unsqueeze(1)
    cont_points = discrete_map(discrete_points)
    
    #TODO: make sure acq_fun is actually averaging over the aleotoric uncertainty
    values = acq_func(cont_points)
    # get the value of the best point
    best_index = torch.argmax(values)
    best_out = values[torch.argmax(values)]
    best_point = discrete_points[best_index]

    return best_point, best_out

def optimize_acqf_and_get_suggested_point(
    acq_func,
    bounds,
    batch_size,
    posterior_mean=None,
) -> Tensor:
    """Optimizes the acquisition function, and returns a new candidate."""
    input_dim = bounds.shape[1]
    num_restarts = 10 * input_dim
    raw_samples = 100 * input_dim

    ic_gen = (
        gen_one_shot_kg_initial_conditions
        if isinstance(acq_func, qKnowledgeGradient)
        else gen_batch_initial_conditions
    )

    # create a set of initialization conditions
    batch_initial_conditions = ic_gen(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": num_restarts},
    )

    # if posterior_mean function specified, adds another initial condition
    if posterior_mean is not None:
        baseline_candidate, _ = optimize_acqf(
            acq_function=posterior_mean,
            bounds=bounds,
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 5},
        )

        if isinstance(acq_func, qKnowledgeGradient):
            augmented_q_batch_size = acq_func.get_augmented_q_batch_size(batch_size)
            baseline_candidate = baseline_candidate.detach().repeat(
                1, augmented_q_batch_size, 1
            )
        else:
            baseline_candidate = baseline_candidate.detach().view(
                torch.Size([1, batch_size, input_dim])
            )

        batch_initial_conditions = torch.cat(
            [batch_initial_conditions, baseline_candidate], 0
        )
        num_restarts += 1
    else:
        baseline_candidate = None

    # starting from initial conditions create a set of candidate solutions
    # adds to the initial conditions the expected best option across the posterior
    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        options={"batch_limit": 2},
        # options={'disp': True, 'iprint': 101},
    )

    if baseline_candidate is not None:
        baseline_acq_value = acq_func.forward(baseline_candidate)[0].detach()
        picked_greedy = int(baseline_acq_value >= acq_value)
        # don't log this for now because not that meaningful, but when you do put everything in one step.
        if baseline_acq_value >= acq_value:
            print("Baseline candidate was best found.")
            candidate = baseline_candidate
            acq_value = baseline_acq_value

    wandb.log({"acq_value": acq_value}, commit=False)

    new_x = candidate.detach().view([batch_size, input_dim])
    return new_x, acq_value
