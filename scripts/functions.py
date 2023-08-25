r"""
Defines the environments that BO agents take actions in. This module contains
all the synthetic function networks used in the paper.

References:

.. [astudilloBayesianOptimizationFunction2021]
    Astudillo, Raul, and Peter Frazier. "Bayesian optimization of function networks." 
    Advances in Neural Information Processing Systems 34 (2021): 14463-14475.
"""

import torch
import math
import numpy as np

from mcbo.utils.dag import DAG, FNActionInput
from mcbo.utils import functions_utils
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
import itertools
from copy import deepcopy

class Env:
    """
    Abstract class for environments.
    """

    def evaluate(self, X):
        r"""
        For action input X returns the observation of all network nodes.
        """
        raise NotImplementedError

    def check_input(self, X):
        """
        Checks if the input to evaluate matches the correct input dim. 
        """
        if X.shape[-1] != self.input_dim:
            raise ValueError("Input dimension does not fit for this function")  

    def check_input_adversary(self, X_a, input_dim_a):
        """
        Checks if the input to evaluate matches the correct input dim. 
        """ 
        if X_a.shape[-1] != input_dim_a:
            raise ValueError("Input dimension of adversary does not fit for this function")

    def check_discrete(self, X, discrete):
        """
        For discrete inputs, checks that X is discrete and within the correct range.
        """
        if discrete != -1:
            if X.dtype != torch.long:
                raise ValueError("Input must be discrete")
            if torch.any(X < 0) or torch.any(X >= discrete):
                raise ValueError("Input out of range")

    def get_causal_quantities(self):
        """
        Outputs a dict of environment details.
        """
        raise NotImplementedError
    
    def discrete_map(self, X):
        """
        Maps continuous input to discrete input. 
        """
        raise NotImplementedError
    
    def discrete_map_adversary(self, X_a):
        """
        Maps continuous input to discrete input. 
        """
        raise NotImplementedError

    def get_base_profile(self):
        """
        Outputs a dict of environment details that can be computed similarly for all
        types of environments. 
        """
        return {
            "additive_noise_dists": self.additive_noise_dists,
            "interventional": self.interventional,
            "dag": self.dag,
            "input_dim": self.input_dim,
            "input_dim_a": self.input_dim_a,
            "discrete_map": self.discrete_map,
            "discrete_map_adversary": self.discrete_map_adversary,
            "full_input_dim": self.full_input_dim,
            "full_input": self.full_input,
        }

    def get_env_profile(self):
        """
        Outputs a dictionary containing all details of the environment needed for 
        BO experiments. 
        """
        return {**self.get_base_profile(), **self.get_causal_quantities()}

class FNEnv(Env):
    """
    Abstract class for function network environments. All function network environments
    implemented as subclasses first appeared in 
    [astudilloBayesianOptimizationFunction2021] but have been modified for the adversarial setting.
    """

    def __init__(self, dag: DAG, action_input: FNActionInput, discrete = -1, input_dim_a = 0, full_input=None):
        self.dag = dag
        self.input_dim = action_input.get_input_dim()
        self.action_input = action_input
        # The full_input is the combined action and adversary inputs
        if full_input is None:
            self.full_input_dim = self.input_dim
            self.full_input = action_input
        else:
            self.full_input_dim = full_input.get_input_dim()
            self.full_input = full_input
        self.interventional = False # Unused: in MCBO was used for hard interventions
        # -1 if actions should be continuous (unused), 1 or more indicates the number of discrete actions
        self.discrete = discrete
        # The number of adversary actions
        self.input_dim_a = input_dim_a

        if input_dim_a > 0 and discrete == -1:
            print(discrete)
            raise ValueError("Adversary only works for discrete actions")

        # make sure discrete is at least 2 else it is pointless
        if discrete != -1 and discrete < 2:
            raise ValueError("Discrete actions must be cardinality at least 2")
        
        self.output_range_min = None # Output range needs to be set by the subclass if you normalize outputs
        self.output_range_max = None # Output range needs to be set by the subclass if you normalize outputs
        self.output_range_set = False

    def discrete_map_adversary(self, X_a):
        """
        Maps discrete adversary input to continuous output.
        In a subclass you need to set this.
        """
        raise NotImplementedError

    def get_causal_quantities(self):
        # valid targets is empty because in FNEnv actions are added nodes (soft interventions), not do-interventions.
        valid_targets = [torch.zeros(self.dag.get_n_nodes())]
        do_map = None
        active_input_indices = self.action_input.get_active_input_indices()
        return {
            "valid_targets": valid_targets,
            "do_map": do_map,
            "active_input_indices": active_input_indices,
        }
    
    def get_output_range(self) -> (torch.Tensor, torch.Tensor):
        """
        Returns the range of the output of the network (for all nodes) by testing all possible inputs for player and adversary. 
        """
        # iterate over all X and X_a
        X = itertools.product(*[range(self.discrete)] * self.input_dim)
        X_a = itertools.product(*[range(self.discrete)] * self.input_dim_a)
        #iterate over all combinations
        all_X = itertools.product(X, X_a)
        # get the output range
        # output_range_max has same dim as number of nodes
        output_range_max = torch.zeros(self.dag.get_n_nodes()) - float("inf")
        output_range_min = torch.zeros(self.dag.get_n_nodes()) + float("inf")
        for X, X_a in all_X:
            X = torch.tensor(X)
            X_a = torch.tensor(X_a)
            # repeat X, X_a 1000 times
            X = X.repeat(1000, 1)
            X_a = X_a.repeat(1000, 1)
            Y = self.evaluate(X, X_a)
            Y_max = torch.max(Y, dim=0)[0]
            Y_min = torch.min(Y, dim=0)[0]
            output_range_max = torch.max(output_range_max, Y_max)
            output_range_min = torch.min(output_range_min, Y_min)
                                         
        return output_range_min, output_range_max
    
    def opt_for_each_opponent(self):
        """
        Returns the optimal action for each action the opponent could take, returns it nicely formatted. 
        This is mostly for sanity-checking environments. 
        """
        # iterate over all X and X_a
        Xs = itertools.product(*[range(self.discrete)] * self.input_dim)
        X_as = itertools.product(*[range(self.discrete)] * self.input_dim_a)
        opt_actions = {}
        index_id = 0

        for X_a in X_as:
            # use opt_player_response
            best_action_index, best_X, best_Y = self.opt_player_response(X_a)
            # get the action
            action = self.discrete_map_adversary(torch.tensor(X_a))
            # get the optimal response for this action
            opt_actions[index_id] = {
                "action": action,
                "opt_action": best_action_index,
                "opt_X": best_X,
                "opt_Y": best_Y
            }
            index_id += 1
        
        print(opt_actions)
        # print number of unique best_action_index
        best_action_indices = [opt_actions[i]["opt_action"] for i in opt_actions]
        print("Number of unique best actions: ", len(set(best_action_indices)))
    
        return opt_actions
    
    def opt_player_response(self, X_a):
        """
        Returns the optimal response (index, X, Y) for a given vector of adversary actions.
        """
        Xs = itertools.product(*[range(self.discrete)] * self.input_dim)

        X_a = torch.tensor(X_a)
        # repeat X_a 1000 times
        X_a = X_a.repeat(1000, 1)
        # get the optimal action for each X
        best_X = None
        best_Y = None
        action_index = 0
        best_action_index = None
        for X in deepcopy(Xs):
            X = torch.tensor(X)
            # repeat X 1000 times
            X_repeat = X.repeat(1000, 1)
            Y = self.evaluate(X_repeat, X_a)[..., -1].mean()
            if best_Y is None or Y > best_Y:
                best_X = X
                best_Y = Y
                best_action_index = action_index
            action_index += 1
        return best_action_index, best_X, best_Y
    
    def opt_opponent_response(self, X):
        """
        Returns the optimal adversary action for each action the player could take. 
        """
        X_as = itertools.product(*[range(self.discrete)] * self.input_dim_a)
        opt_actions = {}
        index_id = 0

        X = torch.tensor(X)
        # repeat X 1000 times
        X = X.repeat(1000, 1)
        # get the optimal action for each X_a
        best_X_a = None
        best_Y = None
        action_index = 0
        best_action_index = None
        for X_a in deepcopy(X_as):
            X_a = torch.tensor(X_a)
            # repeat X_a 1000 times
            X_a_repeat = X_a.repeat(1000, 1)
            Y = self.evaluate(X, X_a_repeat)[..., -1].mean()
            if best_Y is None or Y < best_Y:
                best_X_a = X_a
                best_Y = Y
                best_action_index = action_index
            action_index += 1
        opt_actions[str(index_id)] = best_action_index

        return best_action_index, best_X_a, best_Y
    
    def regret_of_all_X(self):
        """
        For every X, computes the regret that action obtains against an adversary that plays optimally.
        """
        Xs = itertools.product(*[range(self.discrete)] * self.input_dim)
        for X in Xs:
            X = torch.tensor(X)
            best_action_index, X_a, Y = self.opt_opponent_response(X)
            # find the best response to X_a
            best_action_index_opponent, X_opt, Y_opt = self.opt_player_response(X_a)
            # compute the regret
            regret = Y_opt - Y
            print("Regret of X = ", X, " is ", regret, " when opponent picks ", X_a)
        return

    def standardize_output(self, Y):
        """
        Standardizes the output of the network.
        """
        # All values in Y should be normalized to [-1,1] except last dimension which is normalized to [0, 1]
        Y = (Y - self.output_range_min) / (self.output_range_max - self.output_range_min)
        Y[..., :-1] = 2 * Y[..., :-1] - 1
        return Y
    
    def discrete_map(self, X):
        # here X is a vector of ints in range [0, discrete), and we want to map to a vector with each entry in [0, 1]
        return X / (self.discrete -1)

class FNEnvPenny(FNEnv):
    """
    Abstract function network class for the adversarial penny game.
    """
    
    def __init__(self, dag: DAG, action_input: FNActionInput, discrete = -1, input_dim_a = 0, full_input=None):
        super().__init__(dag, action_input, discrete, input_dim_a, full_input)
    
    def discrete_map_adversary(self, X_a):
        """
        Overrides discrete_map_adversary to give outputs in [0, 1] but never = 0.5. 
        Having 0,5 results in the adversary 0'ing outputs in pennies which we don't want. 
        """
        epsilon = 0.005
        # if D is odd
        if self.discrete % 2 == 0:
            return (X_a) / (self.discrete -1) * (1 - 2 * epsilon) + epsilon
        # else if even 
        return (X_a+0.5) / (self.discrete -1) * (1 - 2 * epsilon) + epsilon 

class FNEnvPerturb(FNEnv):
    """
    Abstract function network class for the adversarial perturbation game.
    """
    
    def __init__(self, dag: DAG, action_input: FNActionInput, discrete = -1, input_dim_a = 0, full_input=None):
        super().__init__(dag, action_input, discrete, input_dim_a, full_input)
    
    def discrete_map_adversary(self, X_a):
        return (X_a) / (self.discrete -1)

class Dropwave(FNEnvPerturb):
    """
    A modification of the classic Drop-Wave test function to the Function Networks
    setting.An adversary can perturb the inputs. 

    We provide fairly descriptive comments for this function to show how environments are constructed. 
    The appendix gives DAGs and SEMs for all environments. 
    """
    def __init__(self, noise_scales=0.0, discrete = -1):
        print(discrete)
        parent_nodes = [[], [0]] # For each node we give a list of the indices of its parents
        dag = DAG(parent_nodes) # Create a DAG object based on the parents
        
        self.input_dim_a = 1
        # Construct the 'active input indices', a list of indices of the actions for each node.
        # The full input indices also includes the adversary actions. 
        active_input_indices, full_input_indices = self.adapt_active_input_indices() 
        self.active_input_indices = active_input_indices


        # Construct the input space objects
        action_input = FNActionInput(active_input_indices)
        full_input = FNActionInput(full_input_indices)
        super(Dropwave, self).__init__(dag, action_input, discrete=discrete, input_dim_a = self.input_dim_a, full_input=full_input)

        # A list of noise scales for each node, for if we want additive gaussian noise at each node. 
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )

        self.output_range_set=False
        self.output_range_min, self.output_range_max = self.get_output_range()
        self.output_range_set = True

    def adapt_active_input_indices(self):
        active_input_indices = [[0, 1], []]
        full_input_indices = [[0, 1, 2], []]
        return active_input_indices, full_input_indices

    def evaluate(self, X, X_a):
        assert self.discrete != -1
        self.check_discrete(X, self.discrete)
        X = self.discrete_map(X)

        if X_a is not None:
            self.check_discrete(X_a, self.discrete)
            self.check_input_adversary(X_a, self.input_dim_a)
            X_a = (10.24* (self.discrete_map_adversary(X_a))  - 5.12)*0.2

        X_scaled = 10.24 * X - 5.12
        if X_a is not None:
            #if adversarial, subtract the adversary input from one of the X inputs
            X_scaled[..., 0] = X_scaled[..., 0] - X_a[..., 0]
        input_shape = X_scaled.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        norm_X = torch.norm(X_scaled, dim=-1)
        noise_0 = self.additive_noise_dists[0].rsample(
            sample_shape=output[..., 0].shape
        )
        output[..., 0] = norm_X + noise_0
        noise_1 = self.additive_noise_dists[1].rsample(
            sample_shape=output[..., 1].shape
        )
        output[..., 1] = (1.0 + torch.cos(3.0 * output[..., 0])) / (
            2.0 + 0.5 * (output[..., 0]**2)
        ) + noise_1
        
        if self.output_range_set:
            output = self.standardize_output(output)
        return output

class DropwavePenny(FNEnvPenny):
    """
    Dropwave game with a matching pennies element.
    """
    def __init__(self, noise_scales=0.0, discrete = -1):
        parent_nodes = [[], [0]]
        dag = DAG(parent_nodes)
        
        self.input_dim_a = 1
        active_input_indices, full_input_indices = self.adapt_active_input_indices()
        self.active_input_indices = active_input_indices

        action_input = FNActionInput(active_input_indices)
        full_input = FNActionInput(full_input_indices)
        super(DropwavePenny, self).__init__(dag, action_input, discrete=discrete, input_dim_a = self.input_dim_a, full_input=full_input)

        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )   
        self.output_range_set=False
        self.output_range_min, self.output_range_max = self.get_output_range()
        self.output_range_set = True

    def adapt_active_input_indices(self):
        active_input_indices = [[0, 1], []]
        full_input_indices = [[0, 1], [2]]
        return active_input_indices, full_input_indices 

    def evaluate(self, X, X_a):
        assert self.discrete != -1
        self.check_discrete(X, self.discrete)
        X = self.discrete_map(X)
        
        if X_a is not None:
            self.check_discrete(X_a, self.discrete)
            self.check_input_adversary(X_a, self.input_dim_a)
            X_a = (( self.discrete_map_adversary(X_a)) - 0.5) * 10
        
        X_scaled = 2*X 

        input_shape = X_scaled.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        norm_X = torch.norm(X_scaled, dim=-1)
        noise_0 = self.additive_noise_dists[0].rsample(
            sample_shape=output[..., 0].shape
        )
        output[..., 0] = norm_X + noise_0
        noise_1 = self.additive_noise_dists[1].rsample(
            sample_shape=output[..., 1].shape
        )
        output[..., 1] =  X_a[..., 0] * (torch.cos(3.0 * output[..., 0])) / (
            2.0 + 0.5 * (output[..., 0]**2)
        ) + noise_1 
        
        if self.output_range_set:
            output = self.standardize_output(output)
        return output

class Alpine2(FNEnvPenny):
    """
    A modification of the classic Alpine test function to the Function Networks
    setting, with a matching pennies element. 
    """
    def __init__(self, noise_scales=0.0, discrete = -1):
        
        parent_nodes = [[], [0], [1], [2]]
        dag = DAG(parent_nodes)
        self.input_dim_a = 1
        active_input_indices, full_input_indices = self.adapt_active_input_indices()
        self.active_input_indices = active_input_indices

        action_input = FNActionInput(active_input_indices)
        full_input = FNActionInput(full_input_indices)
        super(Alpine2, self).__init__(dag, action_input, discrete=discrete, input_dim_a=self.input_dim_a, full_input=full_input)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        self.output_range_set=False
        self.output_range_min, self.output_range_max = self.get_output_range()
        self.output_range_set = True

    def adapt_active_input_indices(self):
        active_input_indices = [[0], [1], [], [2]]
        full_input_indices = [[0], [1], [3], [2]]
        return active_input_indices, full_input_indices
    
    def discrete_map_adversary(self, X_a):
        # For alpine we use the standard uniform map
        return self.discrete_map(X_a)

    def evaluate(self, X, X_a = None):
        
        assert self.discrete != -1
        self.check_discrete(X, self.discrete)
        
        X = self.discrete_map(X)
        if X_a is not None:
            self.check_discrete(X_a, self.discrete)
            self.check_input_adversary(X_a, self.input_dim_a)
            X_a = (10.0 * self.discrete_map_adversary(X_a)) + 1.0 # I add 1 so that we don't get 0 actions
        
        X_scaled = 10.0 * X 
        # Same with X_a
        output = torch.empty(X_scaled.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        # store which numbered action we are on when iterating through nodes
        action_index_number = 0
        adversary_index_number = 0
        adversary_targets = [2]
        for i in range(self.dag.get_n_nodes()):
            # if there is an adversary at this node, only the adversary action matters
            if i in adversary_targets:
                x_i = X_a[..., adversary_index_number]
                adversary_index_number += 1
            elif self.active_input_indices[i] != []:
                x_i = X_scaled[..., action_index_number]
                action_index_number += 1
            else:
                #if no action at this node set x_i to 10
                x_i = torch.ones(X_scaled.shape[:-1]) * 10.0
            if i == 0:
                output[..., i] = -torch.sqrt(x_i) * torch.sin(x_i) 

                noise_0 = self.additive_noise_dists[0].rsample(
                    sample_shape=output[..., 0].shape
                )
                output[..., i] = output[..., i] + noise_0 
            else:
                # if the adversary can act here, the agent can't
                
                output[..., i] = (
                    torch.sqrt(x_i) * torch.sin(x_i) * output[..., i - 1] 
                )
                noise_i = self.additive_noise_dists[i].rsample(
                    sample_shape=output[..., i].shape
                )
                output[..., i] = output[..., i] + noise_i

        if self.output_range_set:
            output = self.standardize_output(output)
        return output

class Alpine2Perturb(FNEnvPerturb):
    """
    A modification of the classic Alpine test function to the Function Networks
    setting, where the adversary can perturb the users input. 
    """
    def __init__(self, noise_scales=0.0, discrete = -1):
        #adversary targets is a list of the nodes each adversary can effect
        parent_nodes = [[], [0], [1], [2]]
        dag = DAG(parent_nodes)
        self.input_dim_a = 3
        active_input_indices, full_input_indices = self.adapt_active_input_indices()
        self.active_input_indices = active_input_indices

        action_input = FNActionInput(active_input_indices)
        full_input = FNActionInput(full_input_indices)
        super(Alpine2Perturb, self).__init__(dag, action_input, discrete=discrete, input_dim_a=self.input_dim_a, full_input=full_input)
        
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        self.output_range_set=False
        self.output_range_min, self.output_range_max = self.get_output_range()
        self.output_range_set = True

    def adapt_active_input_indices(self):
        """
        Sets the active input indices based upon the adversary_targets.
        Also creates a joint indice list for the adversary target and active indices
        """
        active_input_indices = [[0], [1], [2], [3]]
        full_input_indices = [[0, 4], [1, 5], [2, 6], [3]]
        return active_input_indices, full_input_indices

    def evaluate(self, X, X_a = None):
        # if adversary_targets is [], just ignore X_a
        assert self.discrete != -1
        self.check_discrete(X, self.discrete)
        
        X = self.discrete_map(X)
        if X_a is not None:
            self.check_discrete(X_a, self.discrete)
            self.check_input_adversary(X_a, self.input_dim_a)
            X_a = 2.0 * (self.discrete_map_adversary(X_a)) # *2.0 so the adversary doesnt have too much control
            # only positive else doesnt work with sqrt
            # current setting leads to 7 unique optimal actions

        X_scaled = 10.0 * X 
        # Same with X_a
        output = torch.empty(X_scaled.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        adversary_targets = [0, 1, 2]
        for i in range(self.dag.get_n_nodes()):
            # if there is an adversary at this node, only the adversary action matters
            if i in adversary_targets:
                x_ai = X_a[..., i]
            else:
                #x_ai is zeros of same size
                x_ai = torch.zeros(X_a.shape[:-1])

            x_i = X_scaled[..., i]
            
            if i == 0:
                x_i = x_i + x_ai
                output[..., i] = -torch.sqrt(x_i) * torch.sin(x_i) 

                noise_0 = self.additive_noise_dists[0].rsample(
                    sample_shape=output[..., 0].shape
                )
                output[..., i] = output[..., i] + noise_0 
            else:
                x_i = x_i + x_ai
                output[..., i] = (
                    torch.sqrt(x_i) * torch.sin(x_i) * output[..., i - 1] 
                )
                noise_i = self.additive_noise_dists[i].rsample(
                    sample_shape=output[..., i].shape
                )
                output[..., i] = output[..., i] + noise_i

        
        if self.output_range_set:
            output = self.standardize_output(output)

        return output

class AckleyPerturb(FNEnvPerturb):
    """
    A modification of the classic Ackley test function to the Function Networks
    setting where the adversary can perturb the users input.
    """
    def __init__(self, noise_scales=0.0, discrete = -1):
        parent_nodes = [[], [], [0, 1]]
        dag = DAG(parent_nodes)
        active_input_indices = [[0, 1, 2, 3], [0, 1, 2, 3], []]
        full_input_indices = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], []]
        self.discrete = discrete
        self.input_dim_a = 2
        action_input = FNActionInput(active_input_indices)
        full_input = FNActionInput(full_input_indices)
        super(AckleyPerturb, self).__init__(dag, action_input, discrete=discrete, input_dim_a = self.input_dim_a, full_input=full_input)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        self.output_range_set=False
        self.output_range_min, self.output_range_max = self.get_output_range()
        self.output_range_set = True

    def evaluate(self, X, X_a=None):
        self.check_input(X)
        X = self.discrete_map(X)
        X_scaled = 4.0 * (X - 0.5)
        if X_a is not None:
            self.check_discrete(X_a, self.discrete)
            self.check_input_adversary(X_a, self.input_dim_a)
            X_a = 4.0 * (self.discrete_map_adversary(X_a) - 0.5) * 0.5

        X_scaled[..., 0] = X_scaled[..., 0] + X_a[..., 0]
        X_scaled[..., 1] = X_scaled[..., 1] + X_a[..., 1]

        output = torch.empty(X_scaled.shape[:-1] + (self.dag.get_n_nodes(),))

        noise_0 = self.additive_noise_dists[0].rsample(
            sample_shape=output[..., 0].shape
        )

        output[..., 0] = (
            1 / self.input_dim * torch.sum(torch.square(X_scaled[..., :]), dim=-1)
            + noise_0
        )
        noise_1 = self.additive_noise_dists[1].rsample(
            sample_shape=output[..., 1].shape
        )
        output[..., 1] = (
            1
            / self.input_dim
            * torch.sum(torch.cos(2 * math.pi * X_scaled[..., :]), dim=-1)
            + noise_1
        )
        noise_2 = self.additive_noise_dists[2].rsample(
            sample_shape=output[..., 2].shape
        )
        output[..., 2] = (
            20 * torch.exp(-0.2 * torch.sqrt(output[..., 0]))
            + torch.exp(output[..., 1])
            - 20
            - math.e
            + noise_2
        )
        if self.output_range_set:
            output = self.standardize_output(output)
        return output

class Ackley(FNEnvPenny):
    """
    A modification of the classic Ackley test function to the Function Networks
    setting.
    """
    def __init__(self, noise_scales=0.0, discrete = -1):
        parent_nodes = [[], [], [0, 1]]
        dag = DAG(parent_nodes)
        active_input_indices = [[0, 1, 2, 3], [0, 1, 2, 3], []]
        full_input_indices = [[0, 1, 2, 3], [0, 1, 2, 3], [4]]
        self.discrete = discrete
        self.input_dim_a = 1
        action_input = FNActionInput(active_input_indices)
        full_input = FNActionInput(full_input_indices)
        super(Ackley, self).__init__(dag, action_input, discrete=discrete, input_dim_a = self.input_dim_a, full_input=full_input)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        self.output_range_set=False
        self.output_range_min, self.output_range_max = self.get_output_range()
        self.output_range_set = True

    def evaluate(self, X, X_a=None):
        self.check_input(X)
        X = self.discrete_map(X)
        X_scaled = 4.0 * (X - 0.5)
        if X_a is not None:
            self.check_discrete(X_a, self.discrete)
            self.check_input_adversary(X_a, self.input_dim_a)
            X_a = 2 * (self.discrete_map_adversary(X_a) - 0.5)

        output = torch.empty(X_scaled.shape[:-1] + (self.dag.get_n_nodes(),))

        noise_0 = self.additive_noise_dists[0].rsample(
            sample_shape=output[..., 0].shape
        )

        output[..., 0] = (
            1 / self.input_dim * torch.sum(torch.square(X_scaled[..., :]), dim=-1)
            + noise_0
        )
        
        noise_1 = self.additive_noise_dists[1].rsample(
            sample_shape=output[..., 1].shape
        )
        output[..., 1] = (
            1
            / self.input_dim
            * torch.sum(torch.cos(2 * math.pi * X_scaled[..., :]), dim=-1)
            + noise_1
        )
        
        noise_2 = self.additive_noise_dists[2].rsample(
            sample_shape=output[..., 2].shape
        )
        output[..., 2] = (
            20 * torch.exp(-0.2 * torch.sqrt(output[..., 0])) * X_a[..., 0]
            + torch.exp(output[..., 1])
            - 20
            - math.e
            + noise_2
        )
        if self.output_range_set:
            output = self.standardize_output(output)
        return output

class Rosenbrock(FNEnvPenny):
    """
    A modification of the classic Rosenbrock test function to the Function Networks
    setting, with a penny adversary.
    """
    def __init__(self, noise_scales=0.0, discrete = -1,):
        parent_nodes = [[], [0], [1]]
        dag = DAG(parent_nodes)
        self.discrete = discrete
        self.input_dim_a = 2
        self.active_input_indices = [[0, 1], [1, 2], [2, 3]]
        self.full_input_indices = [[0, 1], [1, 2, 4], [2, 3, 5]]

        action_input = FNActionInput(self.active_input_indices)
        full_input = FNActionInput(self.full_input_indices)
        
        super(Rosenbrock, self).__init__(dag, action_input, discrete=discrete, input_dim_a = self.input_dim_a, full_input=full_input)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        self.output_range_set=False
        self.output_range_min, self.output_range_max = self.get_output_range()
        self.output_range_set = True

    def evaluate(self, X, X_a = None):
        
        if self.discrete != -1:
            self.check_discrete(X, self.discrete)
            
            X = self.discrete_map(X)
            if X_a is not None:
                self.check_discrete(X_a, self.discrete)
                self.check_input_adversary(X_a, self.input_dim_a)
                X_a = self.discrete_map_adversary(X_a)
        else:
            self.check_input(X)
        # Same with X_a
        # store which numbered action we are on
        action_index_number = 0
        adversary_index_number = 0

        self.check_input(X)
        X_scaled = X 
        output = torch.empty(X_scaled.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        noise_0 = self.additive_noise_dists[0].rsample(
            sample_shape=output[..., 0].shape
        )
        output[..., 0] = (
            -100 * torch.square(X_scaled[..., 1] - torch.square(X_scaled[..., 0]))
            - torch.square(1 - X_scaled[..., 0]) + 10
            + noise_0
        )

        adversary_targets = [1,2]

        if 0 in adversary_targets:
            output[..., 0] *=  X_a[..., adversary_index_number]
            adversary_index_number += 1

        for i in range(1, self.dag.get_n_nodes()):
            x_i = X_scaled[..., i]
            j = i + 1
            x_j = X_scaled[..., j]
            noise_i = self.additive_noise_dists[i].rsample(
                sample_shape=output[..., i].shape
            )
            output[..., i] = (
                -100 * torch.square(x_j - torch.square(x_i))
                - torch.square(1 - x_i) + 10
                + output[..., i - 1]
                + noise_i
            )
            if i in adversary_targets:
                output[..., i] *=  X_a[..., adversary_index_number]
                adversary_index_number += 1
        if self.output_range_set:
            output = self.standardize_output(output)
        return output

class RosenbrockPerturb(FNEnvPerturb):
    """
    A modification of the classic Rosenbrock test function to the Function Networks
    setting, with a perturbation adversary.
    """
    def __init__(self, noise_scales=0.0, discrete = -1,):
        parent_nodes = [[], [0], [1]]
        dag = DAG(parent_nodes)
        self.discrete = discrete
        self.input_dim_a = 2
        self.active_input_indices = [[0, 1], [1, 2], [2, 3]]
        self.full_input_indices = [[0, 1, 4], [1, 2, 4, 5], [2, 3, 5]]

        action_input = FNActionInput(self.active_input_indices)
        full_input = FNActionInput(self.full_input_indices)
        
        super(RosenbrockPerturb, self).__init__(dag, action_input, discrete=discrete, input_dim_a = self.input_dim_a, full_input=full_input)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        self.output_range_set=False
        self.output_range_min, self.output_range_max = self.get_output_range()
        self.output_range_set = True

    def evaluate(self, X, X_a = None):
        
        if self.discrete != -1:
            self.check_discrete(X, self.discrete)
            
            X = self.discrete_map(X)
            if X_a is not None:
                self.check_discrete(X_a, self.discrete)
                self.check_input_adversary(X_a, self.input_dim_a)
                X_a = 4.0 * (self.discrete_map_adversary(X_a) - 0.5) * 0.5
        else:
            self.check_input(X)
        
        # store which numbered action we are on
        action_index_number = 0
        adversary_index_number = 0

        self.check_input(X)
        X_scaled = 4.0 * (X - 0.5)
        # shift the input
        X_scaled[..., 1] = X_scaled[..., 1] - X_a[..., 0]
        X_scaled[...,2] = X_scaled[..., 2] - X_a[..., 1]
        #print(X_scaled)
        output = torch.empty(X_scaled.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        noise_0 = self.additive_noise_dists[0].rsample(
            sample_shape=output[..., 0].shape
        )
        output[..., 0] = (
            -100 * torch.square(X_scaled[..., 1] - torch.square(X_scaled[..., 0]))
            - torch.square(1 - X_scaled[..., 0])
            + noise_0
        )

        for i in range(1, self.dag.get_n_nodes()):
            x_i = X_scaled[..., i]
            j = i + 1
            x_j = X_scaled[..., j]
            noise_i = self.additive_noise_dists[i].rsample(
                sample_shape=output[..., i].shape
            )
            output[..., i] = (
                -100 * torch.square(x_j - torch.square(x_i))
                - torch.square(1 - x_i)
                + output[..., i - 1]
                + noise_i
            )
        if self.output_range_set:
            output = self.standardize_output(output)
        return output
