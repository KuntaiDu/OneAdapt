
import pymongo
from munch import *


def find_in_collection(query, collection, tag, force=False):

    if force:
        logger.info('Force %s on %s', tag, query)
        return None

    if collection.find_one(query) is not None:
        for x in collection.find(query).sort("_id", pymongo.DESCENDING):
            return munchify(x)
    else:
        return None