import os
import torch
import botorch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from torch import Tensor
import wandb
import argparse
from typing import Callable, List, Optional
from mcbo.models import eta_network
import time
import copy
import bikes_map_plotter

from bikes import Bikes, BikesSparse

import warnings

warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float64)
debug._set_state(True)

from mcbo.mcbo_trial import get_model, get_acq_fun
from mcbo.utils import runner_utils

import numpy as np
import sys
import itertools

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def generate_initial_design_bikes(algo_profile, env_profile, get_z):
    """
    Simulate algo_profile['n_init_evals observations'] taking random actions in the bikes env specified by env_profile.
    """
    n_init_evals = algo_profile["n_init_evals"]
    # X will be a trucks by n_init_evals matrix where each entry is drawn uniform categorical from 0 to env_profile.centroids
    X = np.random.randint(
        low=0, high=env_profile["centroids"], size=(n_init_evals, env_profile["trucks"])
    ).astype(int)

    # X_a is of length n_init_evals where each row is obtained by running get_z with 0 through n_init_evals as the t input
    X_a = np.zeros((n_init_evals, env_profile["z_shape"]))
    for i in range(n_init_evals):
        X_a[i, :] = get_z(i+1)

    return X, X_a

def bikes_trial(
    algo_profile: dict,
    env_profile: dict,
    function_network,
    network_to_objective_transform,
    get_z,
    evaluate_with_trip_coords = None,
    plot_maps = False
):
    """
    Run a full trial of the algo in algo_dict on the bikes env specified in env_profile.
    Logs statistics on the algorithm performance to wandb and creates visualizations 
    of the bike allocations. 
    """
    torch.manual_seed(algo_profile['seed'])
    np.random.seed(algo_profile['seed'])

    # Initial random evaluations
    X, X_a = generate_initial_design_bikes(algo_profile, env_profile, get_z)

    network_observation_at_X = None # A tensor containing observations of all nodes for each time point
    observation_at_X = None # A tensor containing the objective value for each time point
    
    all_actions = list(range(env_profile['centroids']))
    # The actions we will maintain weights over
    mw_actions = torch.tensor(all_actions, requires_grad=False)
    mw_weights = torch.ones([env_profile["trucks"], len(all_actions)], requires_grad=False)

    for t in range(algo_profile["n_init_evals"]):
        if network_observation_at_X is None:
            network_observation_at_X = torch.tensor(function_network(X[t, :], X_a[t, :], t+1)).unsqueeze(0)
        else:
            network_observation_at_X = torch.cat(
                (network_observation_at_X, torch.tensor(function_network(X[t, :], X_a[t, :], t+1)).unsqueeze(0))
            )
        if observation_at_X is None:
            observation_at_X = network_to_objective_transform(network_observation_at_X[t, :]).unsqueeze(0)
        else:
            observation_at_X = torch.cat(
                (observation_at_X, network_to_objective_transform(network_observation_at_X[t, :].unsqueeze(0)))
            )

    #Store everything in Tensors even though we'll use np arrays to send inputs to the env
    X = torch.Tensor(X)
    X_a = torch.Tensor(X_a)

    # Now iterate through remaining timepoints where the algorithm picks the bike allocation
    for t in range(algo_profile["n_init_evals"], algo_profile["n_bo_iter"]):
        print("Time step: ", t)
        print("______________________")
        #same z_t
        z_t = get_z(t+1)
        # get the next action
        mw_weights = bikes_get_next_point_mw(X, X_a, network_observation_at_X, observation_at_X, algo_profile, env_profile, network_to_objective_transform, mw_actions, mw_weights)
        
        # for each truck, sample from mw_weights to get the next action. mw_weights has a set of weights per row.
        X_next = np.zeros((env_profile["trucks"])).astype(int)
        mw_numpy = mw_weights.detach().numpy()
        for i in range(env_profile["trucks"]):
            X_next[i] = np.random.choice(mw_actions, p=(mw_numpy[i, :]/np.sum(mw_numpy[i, :])))
        print("Bike locations: ", X_next)
        network_observation_at_X_next = torch.tensor(function_network(X_next, z_t, t+1))
        observation_at_X_next = network_to_objective_transform(network_observation_at_X_next)
        print("Reward: ", observation_at_X_next)
        # update X and X_a
        X = torch.cat((X, torch.Tensor(X_next).unsqueeze(0)))
        X_a = torch.cat((X_a, torch.Tensor(z_t).unsqueeze(0)))
        # update network_observation_at_X
        network_observation_at_X = torch.cat((network_observation_at_X, network_observation_at_X_next.unsqueeze(0)))
        # update observation_at_X
        observation_at_X = torch.cat((observation_at_X, observation_at_X_next.unsqueeze(0)))

        if plot_maps:
            if t%10 == 0:
                # a str used for saving plots
                algo_str = str(algo_profile["n_init_evals"]) + '/' + algo_profile["algo"] + '/' + str(algo_profile["eta"]) + '_' + str(algo_profile["beta"]) + '_' + str(algo_profile["seed"])

                # for each truck, plot the multiplicative weights across centroids
                for i in range(env_profile["trucks"]):
                    bikes_map_plotter.plot_weights_timestamp(mw_weights[i, :], env_profile['centroid_coords'], t=t, algo_str=algo_str, truck_num=i)
                # Do a seperate evaluation that gives the trip coordinates. There is no noise so it will be identical to the evaluation used above. 
                if evaluate_with_trip_coords is not None:
                    _, met, unmet, bikes = evaluate_with_trip_coords(X_next, z_t, t+1)
                    # Plot the bike locations and a visualization of the met and unmet trips. 
                    bikes_map_plotter.plot_bikes_timestep(X[t,:], env_profile['centroid_coords'], t=t, algo_str=algo_str, met=met, unmet=unmet)
                    # Plot the locations of the bikes at each time chunk we measure trips in. For sparseGraph the depth is 1 so we just get the starting locations.
                    bikes_map_plotter.plot_bike_locations(bikes, env_profile['centroid_coords'], algo_str, t=t, met=met, unmet=unmet)
                else:
                    bikes_map_plotter.plot_bikes_timestep(X[t,:], env_profile['centroid_coords'], t=t, algo_str=algo_str)

        wandb.log({"reward": observation_at_X_next.item(), "average_reward": torch.mean(observation_at_X[algo_profile["n_init_evals"]:]).item(), "X": X_next})


def truck_targets_to_bike_vector(X, env_profile):
    """ 
    X is a vector of length env_profile["trucks"]
    returns a vector of length env_profile["centroids"]
    """
    X = X.long()
    X_centroids = torch.zeros((X.shape[0], env_profile["centroids"]))
    for i in range(X.shape[0]):
        for j in X[i]:
            X_centroids[i,j] += env_profile["bikes_per_truck"] / env_profile["n_bikes"]
    return X_centroids

def bikes_update_mw_ucb(
    i,
    acq_func,
    X_t,
    X_at,
    mw_actions,
    mw_weights,
    eta,
    env_profile
) -> Tensor:
    r'''
    Updates the MW weights for the UCB acquisition function. 
    '''
    
    mw_actions = mw_actions.unsqueeze(1)
    # repeat X_at so that the shape is the same as mw_actions
    X_at = X_at.repeat(mw_actions.shape[0], mw_actions.shape[1], 1)
    X_t = X_t.repeat(mw_actions.shape[0], mw_actions.shape[1], 1)
    # replace the ith truck in X_t with mw_actions
    X_t[:, :, i] = mw_actions
    X_t = X_t.reshape(-1, X_t.shape[-1])
    X_t = truck_targets_to_bike_vector(X_t, env_profile)
    X_t = X_t.reshape(mw_actions.shape[0], mw_actions.shape[1], -1)
    # to the end of all mw_actions, add the current X_at
    cont_actions = torch.cat((X_t, X_at), dim=-1)
    values = acq_func(cont_actions)
    values = values.squeeze(0)
    
    values = torch.min(Tensor([1.0]), torch.max(Tensor([0.0]), values))
    mw_weights = mw_weights * torch.exp(- eta * (1.0- values))
    mw_weights = mw_weights / torch.sum(mw_weights)
    return mw_weights
      
def bikes_get_next_point_mw(X: Tensor,
    X_a: Tensor,
    network_observation_at_X: Tensor,
    observation_at_X: Tensor,
    algo_profile: dict,
    env_profile: dict,
    network_to_objective_transform: Callable,
    mw_actions: Tensor,
    mw_weights: Tensor
) -> Tensor:
    
    env_profile['action_input'] = env_profile['full_input'] 
    env_profile["active_input_indices"] = env_profile["action_input"].active_input_indices
    if algo_profile["algo"] == "Random":
        # just return the weights as is since they already represent a uniform distribution
        return mw_weights
    elif algo_profile['algo'].startswith("Fixed"):
        # extract fixed off the start of the algo name, rest contains the index of the depot we set assign bikes to
        fixed = int(algo_profile['algo'][5:])
        # set all mw weight to 0 except this index
        mw_weights = torch.zeros_like(mw_weights)
        mw_weights[:, fixed] = 1.0
        return mw_weights
    elif algo_profile["algo"] == "D-GP-MW":
        for i in range(env_profile["trucks"]):
            X_centroids = truck_targets_to_bike_vector(X, env_profile)

            # cat X and X_a into the model input
            Xs = torch.cat((X_centroids, X_a), dim=1)
            model, _ = get_model(Xs, network_observation_at_X, observation_at_X, algo_profile, env_profile)

            # extract the most recent X_a and X
            X_at = X_a[-1, :]
            X_t = X[-1, :]

            acquisition_function, _ = get_acq_fun(
                model,
                network_to_objective_transform,
                observation_at_X,
                algo_profile,
                env_profile,
            )
            
            #Iterate over all the actions that could have been taken instead at the ith index, then update the weights based on the UCB
            
            mw_weights[i, :] = bikes_update_mw_ucb(
                i,
                acquisition_function,
                X_t,
                X_at,
                mw_actions,
                mw_weights[i, :],
                algo_profile["eta"],
                env_profile
            )
        return mw_weights
    elif algo_profile["algo"] == "D-CBO-MW":
        X_centroids = truck_targets_to_bike_vector(X, env_profile)
        Xs = torch.cat((X_centroids, X_a), dim=1)
        
        model, acq_fun_input_dim = get_model(Xs, network_observation_at_X, observation_at_X, algo_profile, env_profile)
        
        X_at = X_a[-1, :]
        X_t = X[-1, :]
        for i in range(env_profile["trucks"]):
            
            def input_map(X):
                return truck_targets_to_bike_vector(X, env_profile)
            
            # if the graph is simple there are no etas to learn anyway- all 1s. But if we predict bikes we need etas. 
            
            if algo_profile["bike_env"] == "sparse":
                epochs = 1
                restarts = 1
            else:
                epochs = 10
                restarts = 3

            mw_weights = eta_network.train_mw_bikes(
                model,
                network_to_objective_transform,
                env_profile,
                discrete_map=env_profile["discrete_map"],
                input_dim=acq_fun_input_dim,
                mw_actions=mw_actions,
                mw_weights = mw_weights,
                X_t = X_t,
                i = i,
                X_at = X_at,
                input_map = input_map,
                eta=algo_profile["eta"],
                lr = algo_profile["lr"],
                epochs = epochs,
                restarts = restarts
            )

        return mw_weights
    else:
        raise NotImplementedError

def gen_env(bikes_depth, alpha, split_reward_by_centroid, walk_distance_max,bike_env , norm_reward=False):
    """
    Given a set of parameters return the appropriate environment.
    """
    if bike_env == "sparse":
        return BikesSparse(depth = bikes_depth, alpha=alpha, split_reward_by_centroid=split_reward_by_centroid, walk_distance_max=walk_distance_max, norm_reward=norm_reward)
    return Bikes(depth = bikes_depth, alpha=alpha, split_reward_by_centroid=split_reward_by_centroid, walk_distance_max=walk_distance_max)

def launch_exp(config_dict):
    """
    Given a config dict, run a trial of the algorithm and environment specified in this config dict.
    """
    env = gen_env(config_dict['bikes_depth'], config_dict['alpha'], config_dict['split_reward_by_centroid'], config_dict["walk_distance_max"], config_dict["bike_env"], norm_reward=config_dict["norm_reward"])
    def function_network(X: Tensor, X_a: Tensor, t:int):
        return env.evaluate(X=X, X_a=X_a, t=t)
    
    def get_z(t:int):
        return env.get_z_t(t=t)
    
    # Function that maps the network output to the objective value
    network_to_objective_transform = lambda Y: Y[..., -1]
    # For sparse, rewrite network to objective transform so that it takes the last values of Y except the final value and sums them
    # this is because the reward is just the sum of the trips taken
    if config_dict['bike_env'] == "sparse":
        network_to_objective_transform = lambda Y: torch.sum(Y[..., 0:-1], dim=-1)
    
    network_to_objective_transform = GenericMCObjective(network_to_objective_transform)
    batch_size = config_dict["batch_size"]
    env_profile = env.get_env_profile()
    n_bo_iter = 246 # all weekdays with no weekends
    algo_profile = {
        "algo": config_dict["algo"],
        "seed": config_dict["seed"],
        "n_init_evals": config_dict["n_init_evals"],
        "n_bo_iter": n_bo_iter,
        "beta": config_dict["beta"],
        "batch_size": batch_size,
        "eta": config_dict["eta"],
        "lr": config_dict["lr"],
        "split_reward_by_centroid": config_dict["split_reward_by_centroid"],
        "bike_env": config_dict["bike_env"],
        "norm_reward": config_dict["norm_reward"]
    }

    bikes_trial(
        algo_profile=algo_profile,
        env_profile=env_profile,
        function_network=function_network,
        network_to_objective_transform=network_to_objective_transform,
        get_z=get_z,
        evaluate_with_trip_coords=env.evaluate_with_trip_coords,
        plot_maps=config_dict['plot_maps']
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument(
        "-a", "--algo", type=str, default="Random", help="Which algorithm to use"
    )

    parser.add_argument(
        "--kernel", type=str, default="matern", help="Which kernel to use"
    )

    parser.add_argument(
        "-b",
        "--beta",
        type=float, #runner_utils.check_nonnegative,
        default=5.0,
        help="Value of beta in UCB-based algorithms",
    )

    parser.add_argument(
        "--scratch",
        type=bool,
        default=True,
        help="True if saving wandb logs to a scratch folder",
    )

    parser.add_argument(
        "--n_init_evals",
        type=int,
        default=10,
        choices=range(1, 300),
        help="If causal, number of observational samples.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, choices=range(1, 100), help="Batch size of NMCBO in noisy settings"
    )  

    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="Value of eta in MW algorithms",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate of any gradient-based algorithms",
    )

    parser.add_argument(
        "--bikes_depth",
        type=int,
        default=4,
        help="Number of timesteps to break the day into for the bikes",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Value of alpha in specifying the graph.",
    )

    parser.add_argument(
        "--split_reward_by_centroid",
        type=bool,
        default=True,
        help="Whether to compute rewards for every centroid individually",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--bike_env",
        type=str,
        default="sparse",
        help="Which bike environment to use",
    )

    parser.add_argument(
        "--walk_distance_max",
        type=float,
        default=1.0,
        help="Maximum distance a person is willing to walk for a bike",
    )

    parser.add_argument(
        "--norm_reward",
        type=bool,
        default=False,
        help="Whether to normalize all trips based on just possible trips for that day.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--wandb",
        type=bool,
        default=False,
        help="Whether to set wandb to online.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--plot_maps",
        type=bool,
        default=False,
        help="Whether to plot maps showing how the algorithm is acting at various timepoints.",
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()
    if args.wandb:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "offline"

    # if saving to a scratch directory, tell wandb to save there
    WANDB_ENTITY = ""
    WANDB_GROUP = ""
    WANDB_PROJECT = ""
    if args.scratch:
        wandb.init(
            project=WANDB_PROJECT,
            dir=os.environ.get("SCRATCH"),
            entity= WANDB_ENTITY,
            group= WANDB_GROUP,
        )
    else:
        wandb.init(project= WANDB_PROJECT, entity= WANDB_ENTITY, group= WANDB_GROUP)

    wandb.config.update(args)
    config_dict = wandb.config
    print(f"Using {device} device")
    launch_exp(config_dict)