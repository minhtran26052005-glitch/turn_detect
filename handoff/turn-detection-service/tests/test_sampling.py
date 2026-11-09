import numpy as np
import pytest
from turn_detection.sampling import batches, epoch_seed, make_order, verify_complete


def test_resume_consumes_each_row_once_including_partial_batch():
    order = make_order(137, epoch_seed("vi", "fixture", 1))
    visits = np.zeros(len(order), dtype=np.uint8)
    first = next(batches(order))
    visits[order[first[0]:first[1]]] += 1
    for begin, end, _ in batches(order, first[1]):
        visits[order[begin:end]] += 1
    assert verify_complete(order, len(order), visits)["total_row_visits"] == 137
    visits[order[0]] += 1
    with pytest.raises(RuntimeError):
        verify_complete(order, len(order), visits)
