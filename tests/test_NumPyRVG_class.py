import pytest
from rvg import NumPyRVG
import numpy as np
import string

dtypes = [
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float32, np.float64, np.double
]

##### helpers #####
letters = [c for c in string.ascii_lowercase]

def randtype():
    return np.random.choice(dtypes)

def randname():
    return ''.join(np.random.choice(letters) for _ in range(5))

def create_struct_dtype():
    members = np.random.randint(2, 100)
    name = randname()
    return np.dtype(
        [(name + str(i), randtype()) for i in range(members)]
    )
###################

def test_scalar_dtypes():
    rand = NumPyRVG(1000)
    for dtype in dtypes:
        for _ in range(50):
            val = rand(dtype)
            assert isinstance(val, dtype)

def test_array_dtypes():
    rand = NumPyRVG(1000)
    for dtype in dtypes:
        samples = np.random.randint(10, 100)

        arr = rand(dtype, samples=samples)

        assert isinstance(arr, np.ndarray)
        assert len(arr) == samples
        assert arr.dtype == dtype

def test_structured_dtypes():
    rand = NumPyRVG(1000)
    for _ in range(10):
        samples = np.random.randint(10, 100)
        struct_dtype = create_struct_dtype()

        scl = rand(struct_dtype)

        assert scl.dtype == struct_dtype

        arr = rand(struct_dtype, samples=samples)

        assert isinstance(arr, np.ndarray)
        assert len(arr) == samples
        assert arr.dtype == struct_dtype

def test_nested_structured_dtypes_simple():

    rand = NumPyRVG(1000)

    dtypes = [
        [randtype(), randtype()],
        [randtype(), randtype()],
        [randtype(), randtype()]
    ]

    name = randname()

    struct1 = np.dtype([(name + '1', dtypes[0][0]), (name + '2', dtypes[0][1])])
    struct2 = np.dtype([(name + '3', dtypes[1][0]), (name + '4', dtypes[1][1])])
    struct3 = np.dtype([(name + '5', dtypes[2][0]), (name + '6', dtypes[2][1])])

    nested_struct_dtype_simple = np.dtype([
        ('struct1', struct1),
        ('struct2', struct2),
        ('struct3', struct3),
    ])

    val = rand(nested_struct_dtype_simple)

    # recursive structural equality check
    assert val.dtype == nested_struct_dtype_simple and len(val) == 3

def test_nested_structured_dtypes():
    rand = NumPyRVG(1000)
    for _ in range(10):
        members = np.random.randint(2, 100)
        samples = np.random.randint(10, 100)
        nested_struct_dtype = np.dtype(
            [(''.join(np.random.choice(letters) for _ in range(5)), create_struct_dtype()) for _ in range(members)]
        )

    # TODO: a more complete test
    rand(nested_struct_dtype, samples=samples)

def test_negative_limit():
    neglim = np.random.randint(-100, 0)
    with pytest.raises(ValueError) as e:
        NumPyRVG(neglim)
    assert str(e.value) == 'argument `limit` must be a number greater than 0'

def test_negative_samples():
    rand = NumPyRVG(1000)
    neglen = np.random.randint(-100, 0)
    with pytest.raises(ValueError) as e:
        rand(np.int32, neglen)
    assert str(e.value) == 'argument `samples` must be a number greater or equal to 0'

def test_a_b_limits_errors():

    a, b = 2, 1
    with pytest.raises(ValueError) as e:
        NumPyRVG(a, b)
    assert str(e.value) == 'argument `a` must be less than `b`'

    a, b = -10, -5
    with pytest.warns(Warning, match=f'value {b} for argument `b` will cause a runtime error if generation of values of unsigned type is attempted'):
        rand = NumPyRVG(a, b)

    with pytest.raises(ValueError) as e:
        rand(np.uint32)
        assert str(e.value).startswith('unproper limits')

def test_a_b_limits_improper_usage():

    a, b = 1, 1000
    rand = NumPyRVG(a, b)

    vals = rand(np.int8, samples=100)
    assert (vals >= a).all() and (vals <= b).all()

def test_a_b_limits_proper_usage():

    a, b = -17, 42
    rand = NumPyRVG(a, b)

    vals = rand(np.int32, samples=100)
    assert (vals >= a).all() and (vals <= b).all()

    vals = rand(np.uint32, samples=100)
    assert (vals >= 0).all() and (vals <= b).all()
