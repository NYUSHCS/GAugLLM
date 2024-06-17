import random
import os
import numpy as np
import time
import datetime
import pytz

class MultipleOptimizer(object):
    def __init__(self,*op):
        self.optimizers = op
        self.param_groups=[]
        for op in self.optimizers:
            self.param_groups.extend(op.param_groups) 

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
            
def mkdir_p(path, log=True):

    import errno
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file




def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper


def get_neighbors_list(edge_index):
    # Convert edge_index tensors to CPU lists
    edge_index = [[int(e.item()) for e in sublist] for sublist in edge_index]
    edge_index = list(zip(*edge_index))

    # Step 1: Create an empty neighbor list for each node
    neighbors = {}
    for source, target in edge_index:
        if source not in neighbors:
            neighbors[source] = []
        neighbors[source].append(target)

    # Determine the number of nodes
    num_nodes = max(max(neighbors.keys()), max([max(lst) for lst in neighbors.values()])) + 1

    # Step 2: Determine first_neighbor and random_neighbor for each node
    first_neighbor = [None] * num_nodes
    random_neighbor = [None] * num_nodes

    for node, neighbor_list in neighbors.items():
        first_neighbor[node] = neighbor_list[0]
        random_neighbor[node] = random.choice(neighbor_list)

    # Step 3: Generate neighbor lists for each node
    all_neighbors = [[] for _ in range(num_nodes)]
    for node, neighbor_list in neighbors.items():
        all_neighbors[node] = neighbor_list

    return first_neighbor, random_neighbor, all_neighbors


def reorder_text_by_neighbors(text, first_neighbor):
    # Initialize an empty list for reordered text
    reordered_text = [None] * len(text)

    # Reorder the text based on the first_neighbor list
    for idx, neighbor in enumerate(first_neighbor):
        if neighbor is not None:
            reordered_text[idx] = text[neighbor]

    return reordered_text
