import yaml


def update_accumulated_output(accumulated_output, step_output):
    step_output = step_output["raw"]

    for key, step_value in step_output.items():
        if key in accumulated_output:
            accumulated_output[key].extend(list(step_value))
        else:
            accumulated_output[key] = list(step_value)
    return


def read_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def dump_yaml(save_path, dictionary):
    with open(save_path, "w") as f:
        yaml.dump(
            dictionary,
            f,
            default_flow_style=False,
            sort_keys=False
        )
