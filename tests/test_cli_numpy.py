import subprocess as sp
import numpy as np
import string

dtypes = [
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
    'float32', 'float64', 'double'
]

##### helpers #####
def randtype():
    return np.random.choice(dtypes)

def command(cmd):
    cmdout = sp.run(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
    return cmdout.stdout.decode('ascii'), cmdout.stderr.decode('ascii')
###################

def test_default_behavior():
    cout, cerr = command('rvg')
    assert not cerr
    assert 0 <= float(cout) <= 1

def test_scalar_dtypes():
    for dtype, limit in zip(dtypes[:4], [100, 10000, 1000000, 1000000000]):
        a, b = sorted(np.random.randint(-limit, limit, 2))
        b += 1
        cout, _ = command(f'rvg --numpy {dtype} --limits {a} {b}')
        assert int(cout) >= a
        assert int(cout) <= b

    for dtype, limit in zip(dtypes[4:8], [200, 20000, 2000000, 2000000000]):
        a, b = sorted(np.random.randint(0, limit, 2))
        b += 1
        cout, _ = command(f'rvg --numpy {dtype} --limits {a} {b}')
        assert int(cout) >= a
        assert int(cout) <= b

    for dtype in dtypes[8:]:
        a, b = sorted(np.random.randint(-1000, 1000, 2))
        cout, _ = command(f'rvg --numpy {dtype} --limits {a} {b}')
        assert float(cout) >= a
        assert float(cout) <= b

def test_array_dtypes():
    samples = np.random.randint(10, 100)
    cout, cerr = command(f'rvg --numpy uint8 --limits 0 10 --samples {samples}')
    assert not cerr
    assert len(cout.splitlines()) == samples

def test_negative_limit():
    neglim = np.random.randint(-100, -1)
    _, cerr = command(f'rvg --numpy int8 --limits {neglim}')
    assert 'argument `limit` must be a number greater than 0' in cerr

def test_negative_samples():
    neglen = np.random.randint(-100, -1)
    _, cerr = command(f'rvg --numpy int8 --limits 100 --samples {neglen}')
    assert 'argument `samples` must be a number greater or equal to 0' in cerr

def test_a_b_limits_errors():

    a, b = 2, 1
    _, cerr = command(f'rvg --numpy uint16 --limits {a} {b}')
    assert 'argument `a` must be less than `b`' in cerr

    a, b = -10, -5
    _, cerr = command(f'rvg --numpy int16 --limits {a} {b}')
    assert f'value {b} for argument `b` will cause a runtime error if generation of values of unsigned type is attempted' in cerr

    _, cerr = command(f'rvg --numpy uint16 --limits {a} {b}')
    assert 'unproper limits' in cerr

def test_a_b_limits_improper_usage():

    a, b = 1, 1000
    cout, _ = command(f'rvg --numpy int16 --limits {a} {b}')
    for val in map(int, cout.splitlines()):
        assert a <= val <= b

def test_a_b_limits_proper_usage():

    a, b = -17, 42

    cout, _ = command(f'rvg --numpy int32 --limits {a} {b}')
    for val in map(int, cout.splitlines()):
        assert a <= val <= b

    cout, _ = command(f'rvg --numpy uint32 --limits {a} {b}')
    for val in map(int, cout.splitlines()):
        assert 0 <= val <= b
