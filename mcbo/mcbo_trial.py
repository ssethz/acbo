import numpy as np
import os
import sys
import time
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    qExpectedImprovement,
    qKnowledgeGradient,
    UpperConfidenceBound,
    qSimpleRegret,
)
from botorch.acquisition import PosteriorMean as GPPosteriorMean
from botorch.sampling import SobolQMCNormalSampler

from torch import Tensor
from typing import Callable, List, Optional

from mcbo.acquisition_function_optimization.optimize_acqf import (
    optimize_acqf_and_get_suggested_point,
    optimize_discrete_acqf_and_get_best_point,
    update_mw,
)
import mcbo
from mcbo.utils.dag import DAG
from mcbo.utils.initial_design import generate_initial_design
from mcbo.utils.fit_gp_model import fit_gp_model
from mcbo.models.gp_network import GaussianProcessNetwork
from mcbo.utils.posterior_mean import PosteriorMean
from mcbo.models import eta_network

import itertools
import wandb

def obj_mean(
    X: Tensor, X_a : Tensor,
    function_network: Callable, network_to_objective_transform: Callable,
    adversarial: bool = False
) -> Tensor:
    '''
    Estimates the mean value of a noisy objective by computing it a lot of times and 
    averaging.
    '''
    X = X[None]  # create new 0th dim
    if adversarial:
        Ys = function_network(X.repeat(100000, 1, 1), X_a.repeat(100000, 1, 1))
    else:
        Ys = function_network(X.repeat(100000, 1, 1), None) 
    return torch.mean(network_to_objective_transform(Ys), dim=0)

def adversary_action_list(algo_profile: dict, env_profile: dict, method="Random"):
    r'''
    Returns a list of actions the adversary will take. 
    '''

    total_rounds = algo_profile['n_bo_iter']+1 + algo_profile['n_init_evals']
    # if no adversary, return a list of functions outputting None
    if env_profile['input_dim_a'] == 0:
        adversary_actions = []
        for i in range(total_rounds):
            adversary_actions.append(lambda i, mw_actions, mw_weights, function_network, objective: None)
        return adversary_actions

    adversary_actions = []
    if method=="Random":
        def rand_action(i, mw_actions, mw_weights, function_network, objective):
            return torch.randint(0, algo_profile['discrete'], [1, env_profile['input_dim_a']])
        for i in range(total_rounds):
            adversary_actions.append(rand_action)
    elif method=="Adversary":
        # The adversary plays randomly with probability 'jitter', else it plays a best response to the agent's mixed strategy.
        jitter = algo_profile['adversary_jitter']
        def adversary_action(i, mw_actions, mw_weights, function_network, objective):
            # if random number less than jitter just pick a random action
            if torch.rand(1) < jitter:
                return torch.randint(0, algo_profile['discrete'], [1, env_profile['input_dim_a']])
            # cycle through the actions and weights, then pick an action minimizing the objective
            options = list(itertools.product(range(algo_profile['discrete']), repeat=env_profile['input_dim_a']))
            best_val = -torch.Tensor([float('inf')])
            best_action = None
            for a in options:
                a = torch.Tensor(a).long()
                # repeat the option to length matches mw_actions
                a_repeat = a.repeat(len(mw_actions), 1)
                val = -torch.dot(objective(function_network(mw_actions, a_repeat)), mw_weights)
                if val > best_val:
                    best_action = a
                    best_val = val
            return best_action.unsqueeze(0)
        for i in range(total_rounds):
            adversary_actions.append(adversary_action)
    else:
        raise ValueError("Adversary method not implemented.")

    return adversary_actions

def no_regret_action(algo_profile, mw_actions, X_a, function_network, objective):
    """
    Takes in the adversary actions and returns the best fixed action that maximizes the cumulative reward.
    Used for computing regret. 
    """
    # repeat adversary_action until the lenth matches mw_actions
    # iterate over the rounds (use length of X_as)
    action_vals = torch.zeros(len(mw_actions))
    for i in range(len(X_a)):
        X_ai = X_a[i].repeat(len(mw_actions), 1)
        val = objective(function_network(mw_actions, X_ai))
        action_vals += val
    best_action_index = torch.argmax(action_vals)
    return mw_actions[best_action_index].unsqueeze(0)

def mcbo_trial(
    algo_profile: dict,
    env_profile: dict,
    function_network: Callable,
    network_to_objective_transform: Callable,
) -> None:
    r'''
    Interacts with the environment specified by env_profile and function_network for 
    algo_profile['n_bo_iters'] rounds using the algorithm specified in algo_profile. 
    Logs to wandb the average and best score across rounds. 
    '''
    # set the random seed for all libraries
    torch.manual_seed(algo_profile['seed'])
    np.random.seed(algo_profile['seed'])
    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    # results_folder = script_dir + "/results/" + problem + "/" + algo + "/"

    all_actions = list(itertools.product(range(algo_profile['discrete']), repeat=env_profile['input_dim']))
    mw_actions = torch.tensor(all_actions)
    mw_weights = torch.ones([len(all_actions)])

    adversary_actions = adversary_action_list(algo_profile, env_profile, method=algo_profile["adversary_method"])

    # Initial evaluations from randomly sampling agent and adversary actions
    X, X_a = generate_initial_design(algo_profile, env_profile)

    # Flag that indicates whether the environment is adversarial.
    adversarial_env = env_profile['input_dim_a'] > 0

    if algo_profile['algo'].endswith("-MW"):
        mw_algo = True
    else:
        mw_algo = False
    
    mean_at_X = obj_mean(X, X_a, function_network, network_to_objective_transform, adversarial_env)
    network_observation_at_X = function_network(X, X_a)
    observation_at_X = network_to_objective_transform(network_observation_at_X)
    # Current best objective value.
    best_obs_val = observation_at_X.max().item()

    # Historical best observed objective values and running times.
    hist_best_obs_vals = [best_obs_val]
    runtimes = []

    old_nets = []  # only used by NMCBO to reuse old computation
    
    for iteration in range(1, algo_profile["n_bo_iter"] + 1):
        print("Sampling policy: " + algo_profile["algo"])
        print("Iteration: " + str(iteration))

        # New suggested point
        t0 = time.time()
        if mw_algo:
            if iteration == 1:
                # sample a random point from MW
                new_x = mw_actions[torch.randint(0, len(mw_actions), [1])]
                new_net = None
            else:
                new_x, mw_weights = get_new_suggested_point_mw(
                    X=X,
                    X_a = X_a,
                    network_observation_at_X=network_observation_at_X,
                    observation_at_X=observation_at_X,
                    algo_profile=algo_profile,
                    env_profile=env_profile,
                    function_network=function_network,
                    network_to_objective_transform=network_to_objective_transform,
                    old_nets=old_nets,
                    mw_actions=mw_actions,
                    mw_weights=mw_weights,
                )
                new_net = None
        else:
                new_x, new_net = get_new_suggested_point(
                    X=X,
                    network_observation_at_X=network_observation_at_X,
                    observation_at_X=observation_at_X,
                    algo_profile=algo_profile,
                    env_profile=env_profile,
                    function_network=function_network,
                    network_to_objective_transform=network_to_objective_transform,
                    old_nets=old_nets,
                )
                if algo_profile['algo'] == 'random':
                    mw_weights = torch.ones([len(all_actions)]) / len(all_actions)
                else:
                    mw_weights = torch.zeros([len(all_actions)])
                    mw_weights[all_actions.index(tuple(new_x.tolist()[0]))] = 1

        new_x_a = adversary_actions.pop(0)(iteration+algo_profile['n_init_evals'], mw_actions, mw_weights, function_network, network_to_objective_transform)
        
        if new_net is not None:
            old_nets.append(new_net)
        t1 = time.time() #TODO: this time also includes time to generate the adversary right now
        runtimes.append(t1 - t0)
        # Evalaute network at new point
        network_observation_at_new_x = function_network(new_x, new_x_a)

        # The mean value of the new action. 
        mean_at_new_x = obj_mean(
            new_x, new_x_a, function_network, network_to_objective_transform, adversarial_env
        )
        if mean_at_X is None:
            mean_at_X = mean_at_new_x
        else:
            mean_at_X = torch.cat([mean_at_X, mean_at_new_x], 0)

        # Evaluate objective at new point
        observation_at_new_x = network_to_objective_transform(
            network_observation_at_new_x
        )

        # Update training data
        X = torch.cat([X, new_x], 0)
        if adversarial_env:
            X_a = torch.cat([X_a, new_x_a], 0)
        network_observation_at_X = torch.cat(
            [network_observation_at_X, network_observation_at_new_x], 0
        )
        observation_at_X = torch.cat([observation_at_X, observation_at_new_x], 0)

        # Update historical best observed objective values
        best_obs_val = observation_at_X[-iteration:].max().item()
        hist_best_obs_vals.append(best_obs_val)
        print("Best value found so far: " + str(best_obs_val))
        # average_score includes the random exploration runs at the init.

        # compute the regret at this iteration
        no_regret_action_hindsight = no_regret_action(algo_profile, mw_actions, X_a[-iteration:], function_network, network_to_objective_transform)
        # compute the regret of every past action individually (best fixed response may have changed), then sum them
        repeat_no_regret = no_regret_action_hindsight.repeat(iteration, 1)
        regret = obj_mean(repeat_no_regret, X_a[-iteration:], function_network, network_to_objective_transform, adversarial_env) - mean_at_X[-iteration:]
        cumulative_regret = regret.sum()
        wandb.log(
            {
                "score": mean_at_new_x,
                "best_score": torch.max(mean_at_X[-iteration:]),
                "average_score": torch.mean(mean_at_X[-iteration:]),
                "X": new_x,
                "X_a": new_x_a,
                "regret": cumulative_regret,
            }
        )

    # final log of regret
    wandb.log(
        {
            "final_regret": cumulative_regret,
            "final_average_score": torch.mean(mean_at_X[-iteration:]),
        }
    )
def get_model(
    X: Tensor,
    network_observation_at_X: Tensor,
    observation_at_X: Tensor,
    algo_profile: dict,
    env_profile: dict,
):
    input_dim = env_profile["input_dim"]
    algo = algo_profile["algo"]
    if algo == "EIFN":
        model = GaussianProcessNetwork(
            train_X=X,
            train_Y=network_observation_at_X,
            algo_profile=algo_profile,
            env_profile=env_profile,

        )
    elif algo == "EICF":
        model = fit_gp_model(X=X, Y=network_observation_at_X)
    elif algo == "EI":
        model = fit_gp_model(X=X, Y=observation_at_X)
    elif algo == "KG":
        model = fit_gp_model(X=X, Y=observation_at_X)
    elif algo in ["UCB", "GP-MW", "D-GP-MW"]:
        model = fit_gp_model(X=X, Y=observation_at_X)
    elif algo == "MCBO":
        model = GaussianProcessNetwork(
            train_X=X,
            train_Y=network_observation_at_X,
            algo_profile=algo_profile,
            env_profile=env_profile,
        )
        # Make the input dimension bigger to account for the hallucination inputs.
        input_dim = input_dim + env_profile["dag"].get_n_nodes()
    elif algo in ["NMCBO", "CBO-MW", "D-CBO-MW"]:
        model = GaussianProcessNetwork(
            train_X=X,
            train_Y=network_observation_at_X,
            algo_profile=algo_profile,
            env_profile=env_profile,
        )
    else:
        raise ValueError("No algorithm of this name implemented: " + algo)
    # Remove the binary target inputs for interventions (used in MCBO but not here since we don't have hard interventions)
    if env_profile["interventional"]:
        input_dim = input_dim - env_profile["dag"].get_n_nodes()
    return model, input_dim

def get_acq_fun(
    model, network_to_objective_transform, observation_at_X, algo_profile, env_profile
):
    algo = algo_profile["algo"]
    if algo == "EIFN":
        # Sampler
        qmc_sampler = SobolQMCNormalSampler(num_samples=128)
        # Acquisition function
        acquisition_function = qExpectedImprovement(
            model=model,
            best_f=observation_at_X.max().item(),
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
    elif algo == "EICF":
        qmc_sampler = SobolQMCNormalSampler(num_samples=128)
        acquisition_function = qExpectedImprovement(
            model=model,
            best_f=observation_at_X.max().item(),
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
    elif algo == "EI":
        acquisition_function = ExpectedImprovement(
            model=model, best_f=observation_at_X.max().item()
        )
        posterior_mean_function = GPPosteriorMean(model=model)
    elif algo == "KG":
        acquisition_function = qKnowledgeGradient(model=model, num_fantasies=8)
        posterior_mean_function = GPPosteriorMean(model=model)
    elif algo in ["UCB", "GP-MW", "D-GP-MW"]:
        acquisition_function = UpperConfidenceBound(
            model=model, beta=algo_profile["beta"]
        )
        posterior_mean_function = None
    elif algo == "MCBO":
        qmc_sampler = SobolQMCNormalSampler(num_samples=128)
        # Acquisition function
        acquisition_function = qSimpleRegret(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )

        posterior_mean_function = None

    return acquisition_function, posterior_mean_function

def get_new_suggested_point_random(env_profile, algo_profile) -> Tensor:
    r"""Returns a new suggested point randomly."""
    if env_profile["interventional"]:
        # For interventional random = random targets, random values
        target = np.random.choice(env_profile["valid_targets"])
        return mcbo.utils.initial_design.random_causal(target, env_profile), None
    else:
        if algo_profile["discrete"] != -1:
            return torch.randint(
                0, algo_profile["discrete"], [1, env_profile["input_dim"]]
            ), None
        return torch.rand([1, env_profile["input_dim"]]), None

def get_new_suggested_point_mw(
    X: Tensor,
    X_a: Tensor,
    network_observation_at_X: Tensor,
    observation_at_X: Tensor,
    algo_profile: dict,
    env_profile: dict,
    function_network: Callable,
    network_to_objective_transform: Callable,
    old_nets: List,
    mw_actions: Tensor,
    mw_weights: Tensor,
):
    r"""Returns a new suggested point using the MW algorithm."""

    # sample from the MW distibution
    algo = algo_profile["algo"]
    
    algos_no_etas = ["UCB", "KG", "EI", "EICF", "GP-MW"]

    #If environment is discrete, map all the discrete X to the continuous space
    X = env_profile["discrete_map"](X)
    X_a = env_profile["discrete_map_adversary"](X_a)
    #concat X_a onto X
    X = torch.cat((X, X_a), 1)

    env_profile['action_input'] = env_profile['full_input'] 
    env_profile["active_input_indices"] = env_profile["action_input"].active_input_indices
    
    model, acq_fun_input_dim = get_model(
        X, network_observation_at_X, observation_at_X, algo_profile, env_profile
    )

    #get just the most recent X_a
    X_at = X_a[-1, :].reshape(1, -1)

    if algo in algos_no_etas:
        acquisition_function, posterior_mean_function = get_acq_fun(
            model,
            network_to_objective_transform,
            observation_at_X,
            algo_profile,
            env_profile,
        )
        mw_weights = update_mw(acquisition_function, X_at, mw_actions, mw_weights, env_profile['discrete_map'], algo_profile['eta'])
    elif algo in ["NMCBO", "CBO-MW"]:
        
        mw_weights = eta_network.train_mw(
            model,
            network_to_objective_transform,
            env_profile,
            discrete_map=env_profile["discrete_map"],
            input_dim=acq_fun_input_dim,
            mw_actions=mw_actions,
            mw_weights = mw_weights,
            X_at = X_at,
            eta=algo_profile["eta"],
            lr = algo_profile["lr"],
        )
    else:
        raise ValueError("This algorithm is not implemented for the adverarial setting")
    choice = np.random.choice(mw_actions.shape[0], p=mw_weights.detach().numpy())
    new_suggested_point = mw_actions[choice].unsqueeze(0)
    return new_suggested_point, mw_weights

def get_new_suggested_point(
    X: Tensor,
    network_observation_at_X: Tensor,
    observation_at_X: Tensor,
    algo_profile: dict,
    env_profile: dict,
    function_network: Callable,
    network_to_objective_transform: Callable,
    old_nets: List
) -> Tensor:

    algo = algo_profile["algo"]
    
    algos_interventions_not_implemented = ["UCB", "KG", "EI", "EICF"]

    if env_profile["interventional"] and algo in algos_interventions_not_implemented:
        raise ValueError(
            "This algorithm is not implemented for the interventional setting"
        )

    if algo == "Random":
        return get_new_suggested_point_random(env_profile, algo_profile)

    #If environment is discrete, map all the discrete X to the continuous space
    if algo_profile["discrete"] != -1:
        X = env_profile["discrete_map"](X)

    model, acq_fun_input_dim = get_model(
        X, network_observation_at_X, observation_at_X, algo_profile, env_profile
    )

    if algo in algos_interventions_not_implemented:
        acquisition_function, posterior_mean_function = get_acq_fun(
            model,
            network_to_objective_transform,
            observation_at_X,
            algo_profile,
            env_profile,
        )
        if algo_profile['discrete'] != -1:
            new_x, new_score = optimize_discrete_acqf_and_get_best_point(
                acq_func=acquisition_function,
                discrete=algo_profile['discrete'],
                discrete_map=env_profile['discrete_map'],
                width=acq_fun_input_dim
            )
            return new_x, None
        else:
            new_x, new_score = optimize_acqf_and_get_suggested_point(
                acq_func=acquisition_function,
                bounds=torch.tensor(
                    [
                        [0.0 for i in range(acq_fun_input_dim)],
                        [1.0 for i in range(acq_fun_input_dim)],
                    ]
                ),
                batch_size=1,
                posterior_mean=posterior_mean_function,
            )
            return new_x, None

    r"""
    The approach is general for both causal BO environments (interventional=True) and 
    function network environments. For both we loop of the possible intervention targets.
    The 'target' for function networks makes no difference to the output and only a 
    single set of targets is contained in valid_targets. 
    """

    best_x = None
    best_score = -torch.inf
    best_target = None

    for target in env_profile["valid_targets"]:
        try:
            model.set_target(target)
        except:
            raise ValueError("Model class isn't able to have interventional targets")
        ## Use a different training procedure if noisy MCBO is used.
        if algo == "NMCBO":
            if algo_profile["discrete"] != -1:
                new_x, new_score, new_net = eta_network.train_discrete(
                    model,
                    network_to_objective_transform,
                    env_profile,
                    discrete_map=env_profile["discrete_map"],
                    discrete=algo_profile["discrete"],
                    input_dim=acq_fun_input_dim,
                    lr=algo_profile["lr"],
                )
            else:
                new_x, new_score, new_net = eta_network.train(
                    model,
                    network_to_objective_transform,
                    env_profile,
                    acq_fun_input_dim,
                    old_nets,
                    batch_size=algo_profile["batch_size"],
                    lr=algo_profile["lr"],
                )
        else:
            acquisition_function, posterior_mean_function = get_acq_fun(
                model,
                network_to_objective_transform,
                observation_at_X,
                algo_profile,
                env_profile,
            )

            if algo_profile['discrete'] != -1:
                new_x, new_score = optimize_discrete_acqf_and_get_best_point(
                    acq_func=acquisition_function,
                    discrete=algo_profile['discrete'],
                    discrete_map=env_profile['discrete_map'],
                    width=acq_fun_input_dim,
                )
            else:
                new_x, new_score = optimize_acqf_and_get_suggested_point(
                    acq_func=acquisition_function,
                    bounds=torch.tensor(
                        [
                            [0.0 for i in range(acq_fun_input_dim)],
                            [1.0 for i in range(acq_fun_input_dim)],
                        ]
                    ),
                    batch_size=1,
                    posterior_mean=posterior_mean_function,
                )
        if new_score > best_score:
            best_score = new_score
            best_x = new_x
            best_target = torch.tensor(target)

    r"""
    For algorithms that hallucinate inputs, we need to remove these parts of the action
    because they are not used in the real environment. 
    """

    if algo in ["NMCBO", "MCBO"]:
        if env_profile["interventional"]:
            r"""
            For causal BO we also ignore the target dimension which the env_profile
            counts in the input_dim.
            """
            X_dim = env_profile["input_dim"] - env_profile["dag"].get_n_nodes()
            best_x = best_x[:, 0:X_dim]
        else:
            X_dim = env_profile["input_dim"]
            best_x = best_x[:, 0:X_dim]

    # If we're in a causal BO setting to need to prepend the best targets to our action.
    if env_profile["interventional"]:
        best_x = torch.cat([best_target.unsqueeze(0), best_x], dim=-1)

    # Only NMCBO stores the action and eta networks previously used.
    if algo != "NMCBO":
        new_net = None

    return best_x, new_net

