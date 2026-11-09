"""Exact once-per-epoch permutations with auditable, resumable cursors."""
import hashlib
import math


def epoch_seed(language, generation, epoch):
    return int.from_bytes(hashlib.sha256(f'{language}|{generation}|{epoch}|full_epoch_v1'.encode()).digest()[:8],'little')


def make_order(n, seed):
    import numpy as np
    if n<=0:raise ValueError('Empty training set')
    return np.random.default_rng(seed).permutation(n).astype(np.int64)


def planned_steps(n, microbatch=32, accumulation=2):
    if min(n,microbatch,accumulation)<=0:raise ValueError('Positive sizes required')
    return math.ceil(n/(microbatch*accumulation))


def batches(order, position=0, microbatch=32, accumulation=2):
    if not 0<=position<=len(order):raise ValueError('Bad resume cursor')
    width=microbatch*accumulation
    while position<len(order):
        stop=min(len(order),position+width)
        yield position,stop,[order[i:min(i+microbatch,stop)] for i in range(position,stop,microbatch)]
        position=stop


def verify_complete(order, cursor, visits):
    import numpy as np
    n=len(order)
    if cursor!=n or visits.shape!=(n,) or not np.all(visits==1):
        raise RuntimeError('Cannot certify epoch: skipped or repeated training rows')
    if not np.array_equal(np.sort(order),np.arange(n,dtype=np.int64)):
        raise RuntimeError('Invalid epoch permutation')
    return {'eligible_rows':n,'unique_rows_seen':int(np.count_nonzero(visits)),
            'total_row_visits':int(visits.sum()),'missing_rows':int(np.count_nonzero(visits==0)),
            'repeated_rows':int(np.count_nonzero(visits>1)),'coverage':1.0,
            'permutation_sha256':hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()}
