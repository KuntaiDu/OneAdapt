
import hashlib
import json

__all__ = ['sha256_hash']

def sha256_hash(x):

    return hashlib.sha256(str.encode(json.dumps(x))).hexdigest()