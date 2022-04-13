def update_accumulated_output(accumulated_output, step_output):
    step_output = step_output["raw"]

    for key, step_value in step_output.items():
        if key in accumulated_output:
            accumulated_output[key].extend(list(step_value))
        else:
            accumulated_output[key] = list(step_value)
    return
