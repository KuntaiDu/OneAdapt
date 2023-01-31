
import torch

import pickle

from munch import munchify

__all__ = ['serialize_db_argument']


def serialize_db_argument(args):

    for key in args:

        if isinstance(args[key], torch.Tensor):
            args[key] = args[key].tolist()

    return munchify(args)
