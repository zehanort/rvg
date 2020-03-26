import numpy as np
import rvg

def struct(t, dtype):
    return np.array([t], dtype=dtype)[0]

def equal(a, b):
    print(a)
    print(b)
    return (a == b).all()

#  Define a struct with 3 scalar fields
simple_struct = np.dtype([
    ('f0', np.float32),
    ('f1', np.int64),
    ('f2', np.longlong)
])

#  Define distribution parameters for each field as a dictionary tree
#      that mirrors `simple_struct`'s structure
simple_struct_param = {
    'f0' : 17,
    'f1' : 128,
    'f2' : 42
}

#  Use it with parameters and `shape=(5, 2)` to generate
#      an `np.ndarray` of `simple_struct`s
def test_simple_struct_array():
    # Set seed for reproducibility
    np.random.seed(42)

    randstruct = rvg.NumPyRVG(dtype=simple_struct)
    random = randstruct(simple_struct_param, (5, 2))
    expected = np.array([
        [( -4.265636 ,   21, -22), ( 15.324286 ,  -76, -10)],
        [(  7.887794 , -127,  33), (  3.3543885,  -41,  15)],
        [(-11.695366 ,  107, -21), (-11.696186 ,   29,   6)],
        [(-15.025157 ,  -91,  16), ( 12.449989 ,    1,  -1)],
        [(  3.4379103,   63,  17), (  7.0744677,   59,  37)]
    ], dtype=np.dtype([('f0', '<f4'), ('f1', '<i8'), ('f2', '<i8')]))

    assert(equal(random, expected))

#  Or omit `shape` to get a single `simple_struct` scalar back
def test_simple_struct_single():
    # Set seed for reproducibility
    np.random.seed(42)

    randstruct = rvg.NumPyRVG(dtype=simple_struct)
    random = randstruct(simple_struct_param)
    expected = struct(
            (-4.265636, -36, -28),
            dtype=[('f0', '<f4'), ('f1', '<i8'), ('f2', '<i8')])

    assert(equal(random, expected))

#  Also works for nested structs
nested_struct = np.dtype([
    ('f0', simple_struct),
    ('f1', np.uint16),
    ('f2', simple_struct)
])

#  We can leave out the details if we want the whole subtree of a field
#      to use the same parameters (note `f0`)
nested_struct_param = {
    'f0' : (0, 42),
    'f1' : 42,
    'f2' : simple_struct_param
}

def test_nested_struct():
    # Set seed for reproducibility
    np.random.seed(42)

    randstruct = rvg.NumPyRVG(dtype=nested_struct)
    random = randstruct(nested_struct_param, (5, 2))
    expected = np.array([[((15.730685 , 21, 24),  8, (-16.457148 ,  125,   1)),
        ((39.93     ,  1, 26), 19, ( -9.14961  ,    5,  -9))],
       [((30.743746 , 23, 41), 38, ( -8.805134 ,  -75,  31)),
        ((25.143656 , 29, 27), 39, (  6.2309594,  -23,  19))],
       [(( 6.552783 , 37, 15), 17, (  3.7398863, -125, -29)),
        (( 6.5517697,  1, 14), 37, ( 11.328627 ,  -75,   5))],
       [(( 2.4395118, 20,  2),  3, (-11.105601 ,   92, -28)),
        ((36.3794   , 32, 36), 24, ( -3.7039394,   62,  29))],
       [((25.24683  , 11,  6), 13, (-10.803973 ,   17,  35)),
        ((29.739048 , 21, 20),  8, (  8.682288 ,   89,  19))]],
      dtype=[('f0', [('f0', '<f4'), ('f1', '<i8'), ('f2', '<i8')]),
             ('f1', '<u2'),
             ('f2', [('f0', '<f4'), ('f1', '<i8'), ('f2', '<i8')])])

    assert(equal(random, expected))

#  Array members are also supported!
struct_array_member = np.dtype([
    ('i', np.int8),
    ('a', (simple_struct, 3))
])

struct_array_member_param = {
    'i' : 42,
    'a' : simple_struct_param
}

def test_struct_array_member():
    # Set seed for reproducibility
    np.random.seed(42)

    randstruct = rvg.NumPyRVG(dtype=struct_array_member)
    random = randstruct(struct_array_member_param, (3, 2))
    expected = np.array([
        [(  9, [(  7.887794 ,  -70, -40), (  3.3543885,  126,   8), (-11.695366 ,   41, -36)]),
         ( 19, [(-11.696186 ,  127, -22), (-15.025157 ,   91,  30), ( 12.449989 ,   59,  -4)])],
        [( 33, [(  3.4379103,   79, -25), (  7.0744677, -114, -39), (-16.300127 ,   61,  17)]),
         (-39, [( 15.976935 ,   61, -29), ( 11.30305  ,   46, -34), ( -9.78047  ,   61,  10)])],
        [(-28, [(-10.817951 ,  -78, -41), (-10.764247 ,  -21,  41), ( -6.6557636,  -74,  17)]),
         (-21, [(  0.8417187,  115,  28), ( -2.3138695,  -65,   1), ( -7.0982094,  120, -35)])]
    ], dtype=[('i', 'i1'), ('a', [('f0', '<f4'), ('f1', '<i8'), ('f2', '<i8')], (3,))])
    assert(equal(random, expected))
