import torch


def generate_initial_design_function_networks(
    num_samples: int, input_dim: int, seed=None, discrete = -1, input_dim_a = 0
):
    
    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        if discrete == -1:
            X = torch.rand([num_samples, input_dim])
            X_a = None
        else:
            X = torch.randint(0, discrete, [num_samples, input_dim])
            X_a = torch.randint(0, discrete, [num_samples, input_dim_a])
        torch.random.set_rng_state(old_state)
    else:
        if discrete == -1:
            X = torch.rand([num_samples, input_dim])
            X_a = None
        else:
            X = torch.randint(0, discrete, [num_samples, input_dim])
            X_a = torch.randint(0, discrete, [num_samples, input_dim_a])
    return X, X_a


def random_causal(target, env_profile, N=1):
    r"""
    Returns N random interventions with a specific target.
    """
    values = torch.rand([N, env_profile["dag"].get_n_nodes()])
    return torch.cat([target.repeat(N, 1), values], dim=-1)


def generate_initial_design_causal(
    algo_profile: dict,
    env_profile: dict,
):
    X = random_causal(
        torch.zeros(env_profile["valid_targets"][0].shape),
        env_profile,
        N=algo_profile["initial_obs_samples"],
    )
    for target in env_profile["valid_targets"]:
        if torch.all(target == 0):
            continue
        else:
            X = torch.cat(
                [
                    X,
                    random_causal(
                        target, env_profile, N=algo_profile["initial_int_samples"]
                    ),
                ],
                dim=0,
            )
    return X


def generate_initial_design(
    algo_profile: dict,
    env_profile: dict,
):
    r"""
    Outputs an initial expoloratory X.
    """
    #TODO discrete input only handled for function networks so far
    if env_profile["interventional"]:
        X = generate_initial_design_causal(algo_profile, env_profile)
        X_a = None
    else:
        X, X_a = generate_initial_design_function_networks(
            num_samples=algo_profile["n_init_evals"],
            input_dim=env_profile["input_dim"],
            seed=algo_profile["seed"],
            discrete = algo_profile["discrete"],
            input_dim_a=env_profile["input_dim_a"],
        )
    return X, X_a
