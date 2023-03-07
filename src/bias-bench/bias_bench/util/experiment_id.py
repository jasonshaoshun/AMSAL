def generate_experiment_id(
    name,
    model=None,
    model_name_or_path=None,
    bias_type=None,
    seed=None,
    pre_assignment_type=None,
):
    experiment_id = f"{name}"

    # Build the experiment ID.
    if isinstance(model, str):
        experiment_id += f"_m-{model}"
    if isinstance(model_name_or_path, str):
        experiment_id += f"_c-{model_name_or_path}"
    if isinstance(bias_type, str):
        experiment_id += f"_t-{bias_type}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"
    if isinstance(pre_assignment_type, str):
        experiment_id += f"_p-{pre_assignment_type}"

    return experiment_id
