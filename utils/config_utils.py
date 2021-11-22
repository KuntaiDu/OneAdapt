
import random
# from collections import defaultdict
# import torch

# class Space:

#     def __init__(self):

#         self.space = defaultdict(list)

#     def __getitem__(self, key):

#         return self.space[key]

#     def set_serialize_format(self, serialize_order, serialize_format):

#         assert all(i in serialize_order for i in self.space.keys())

#         self.serialize_order = serialize_order
#         self.serialize_format = serialize_format

#     def get_value(self, key, value):

#         space = self.space
#         assert key in space
#         assert 1e-7 < value <= 1

#         num = len(space[key]) - 1
#         if num == 0:
#             return ((space[key][0], 1.), (space[key][0], 0.))
        
#         delta = 1/num

#         for i in range(num):
#             left = 1 - (delta * i)
#             right = 1 - (delta * (i+1))
#             x = value

#             if left >= x > right:
#                 # high quality up front.
#                 return ((space[key][i], (x - right)/(left-right)), (space[key][i + 1], (left-x)/(left-right)))

#     def serialize(self, state):

#         assert state.keys() == self.space.keys()

#         def transform(x):
#             if isinstance(x, torch.Tensor):
#                 return x.item()
#             else:
#                 return x

#         params = [transform(state[key]) for key in self.serialize_orders]

#         def helper(params, list):


serialize_order = []
space = {}



def state2config(state, serialize=False):

    config = {}

    for key in state:

        assert 1 >= state[key] > 0

        num = len(space[key]) - 1
        if num == 0:
            config[key] = ((space[key][0], 1.), (space[key][0], 0.))
            continue
        
        delta = 1/num

        for i in range(num):
            left = 1 - (delta * i)
            right = 1 - (delta * (i+1))
            x = state[key]

            if left >= x > right:
                # high quality up front.
                if serialize:
                    config[key] = ((space[key][i], ((x - right)/(left-right)).item()), (space[key][i + 1], ((left-x)/(left-right)).item()))
                else:
                    config[key] = ((space[key][i], (x - right)/(left-right)), (space[key][i + 1], (left-x)/(left-right)))
                break

    return config

def serialize(prefix, config, lq_key = None):

    if lq_key is not None:
        assert lq_key in serialize_order

    prefix = prefix

    for key in serialize_order:
        if key == lq_key:
            prefix = prefix + '_' + key + '_' + str(config[key][1][0]) 
        else:
            prefix = prefix + '_' + key + '_' + str(config[key][0][0])

    prefix = prefix + '.mp4'

    return prefix

def serialize_gt(prefix):

    prefix = prefix
    for key in serialize_order:
        prefix = prefix + '_' + key + '_' + str(space[key][0])
    return prefix + '.mp4'

def random_serialize(prefix, config):

    prefix = prefix

    for key in serialize_order:

        weight = random.random()
        weight = 0

        if weight > config[key][0][1]:
            prefix = prefix + '_' + key + '_' + str(config[key][1][0]) 
        else:
            prefix = prefix + '_' + key + '_' + str(config[key][0][0])

    prefix = prefix + '.mp4'

    return prefix

def serialize_all_states(prefix, config, prob, keys):

    if len(keys) == 0:
        yield (prefix + '.mp4', prob)
    else:
        key = keys[0]
        yield from serialize_all_states(prefix + '_' + key + '_' + str(config[key][0][0]), config, prob * config[key][0][1], keys[1:])
        yield from serialize_all_states(prefix + '_' + key + '_' + str(config[key][1][0]), config, prob * config[key][1][1], keys[1:])

def lookup(video_name, stats):
    for i in stats:
        if i['video_name'] == video_name:
            return i
    assert False, 'No entry for %s found.' % video_name
