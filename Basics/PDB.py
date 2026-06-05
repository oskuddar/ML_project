
# PDB Debugger Example
# import pdb; pdb.set_trace() - old way to set a breakpoint
# breakpoint() - new way to set a breakpoint
# python3 -m pdb my_script.py - post-mortem debugging

# p variable_name print variable value
# c continue running
# q quit debugger

def get_sum_metrics(predictions, metrics=None):
    if not metrics:
        metrics = []
    for i in range(3):
        metrics.append(add_me(i))
    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)
    return sum_metrics

def add_me(i):
    return lambda x: x + i

predictions = 50
metrics = [lambda x: x / 2, lambda x: x+3, lambda x: x*2]
get_sum_metrics(predictions, metrics)

