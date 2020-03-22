import pytest
from rvg import NumPyRVG
import numpy as np
import os
import string

dtypes = [
    np.int, np.int8, np.int16, np.int32, np.int64,
    np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
    np.half, np.float32, np.float64, np.double
]

letters = [c for c in string.ascii_lowercase]

def create_struct_type():
    members = np.random.randint(2, 100)
    return np.dtype(
        [(''.join(np.random.choice(letters) for _ in range(5)), np.random.choice(dtypes)) for _ in range(members)]
    )


def test_scalar_types():
    rand = NumPyRVG(1000)
    for dtype in dtypes:
        val = rand(dtype)
        assert type(val) == dtype

def test_array_types():
    rand = NumPyRVG(1000)
    for dtype in dtypes:
        length = np.random.randint(10, 100)

        arr = rand(dtype, length)

        assert type(arr) == np.ndarray
        assert len(arr) == length
        assert arr.dtype == dtype

def test_structured_types():
    rand = NumPyRVG(1000)
    for _ in range(10):
        length = np.random.randint(10, 100)
        struct_dtype = create_struct_type()

        scl = rand(struct_dtype)

        assert type(scl) == tuple

        arr = rand(struct_dtype, length)

        assert type(arr) == np.ndarray
        assert len(arr) == length
        assert arr.dtype == struct_dtype

def test_nested_structured_types():
    rand = NumPyRVG(1000)
    for _ in range(10):
        members = np.random.randint(2, 100)
        length = np.random.randint(10, 100)
        nested_struct_dtype = np.dtype(
            [(''.join(np.random.choice(letters) for _ in range(5)), create_struct_type()) for _ in range(members)]
        )

        val = rand(nested_struct_dtype)

def test_negative_limit():
    neglim = np.random.randint(-100, 0)
    with pytest.raises(ValueError) as e:
        NumPyRVG(neglim)
    assert str(e.value) == 'argument `limit` must be a number greater than 0'

def test_negative_length():
    rand = NumPyRVG(1000)
    neglen = np.random.randint(-100, 0)
    with pytest.raises(ValueError) as e:
        rand(np.int32, neglen)
    assert str(e.value) == 'argument `length` must be a number greater or equal to 0'
