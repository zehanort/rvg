# &#127922; rvg &#127922; - Random Values Generator
![Testing](https://github.com/zehanort/rvg/workflows/Testing/badge.svg)
[![codecov](https://codecov.io/gh/zehanort/rvg/branch/dev/graph/badge.svg)](https://codecov.io/gh/zehanort/rvg)
[![CodeFactor](https://www.codefactor.io/repository/github/zehanort/rvg/badge)](https://www.codefactor.io/repository/github/zehanort/rvg)
![PyPI release](https://img.shields.io/pypi/v/rvg?label=PyPI%20release)

## Description

`rvg` is a Python 3 package utility to create random values of any Python 3 data type.

Its main purpose is to help in applications where reliable -in terms of type safety- random values are needed (e.g. statistics, machine learning, general testing etc), and in specific layouts (e.g. "I want a `numpy` structured array of random pairs of ints and floats and I want it now").

## Authors
[Sotiris Niarchos](https://github.com/zehanort) and [George Papadopoulos](https://github.com/gepapado)

## Installation

You can either:
- download the source code from the [releases page](https://github.com/zehanort/rvg/releases) or clone the repo (the `master` branch will always mirror the latest release)
- use `pip`:
```
pip install rvg
```

## Current Status

### Alpha release

For the time being, only `numpy` types are supported. More specifically:
- `numpy` scalar data types
- `numpy` arrays of scalar data types
- `numpy` arrays of structured data types

After the alpha release, more features will be implemented, focusing mainly on Python 3 native types.

## Usage

Right now, `rvg` provides 2 interafaces for random values generation through the `NumPyRVG` class:
1. Create a generator of a *certain data type*
2. Create a generator with *specific numerical limits*
A demonstration follows:

```Python
from rvg import NumPyRVG
import numpy as np

# Interface 1
randuint = NumPyRVG(dtype=np.uint16)

# Interface 2
randsmall = NumPyRVG(limit=10) # same as limits=(-10, 10)
randbig = NumPyRVG(limits=(1e10, 1e100))
```

The functionalities of `NumPyRVG` include the generation of:

- `numpy` scalar data types:

```Python console
>>> randsmall(np.uint8)
8
>>> randbig(np.double)
1.9296971162995923e+99
>>> res = [randsmall(t) for t in [np.int8, np.uint16, np.float32, np.double]]
>>> res
[7, 1, 3.0503626, 3.759943941132951]
>>> list(map(type, res))
[<class 'numpy.int8'>, <class 'numpy.uint16'>, <class 'numpy.float32'>, <class 'numpy.float64'>]
```

- `numpy` array data types from scalar types:

```Python console
>>> randuint((50, 100), shape=3)
array([79, 81, 85], dtype=uint16)
>>> randsmall(np.float16, shape=(4, 2))
array([[-7.23 , -9.31 ],
       [-4.97 , -6.06 ],
       [-5.19 , -3.344],
       [-6.586, -3.133]], dtype=float16)
```

- `numpy` structured array data types (structured datatypes can be nested). Limits and shapes can be given in the form of a dictionary, describing all limits and/or shapes of each field of each level of the structured data type. An example follows:

```Python
#  Consider the struct definition below, in C:
##############################################
#  typedef struct knode {
#      int location;
#      int indices [3];
#      int  keys [3];
#      bool is_leaf;
#      int num_keys;
#  } knode;
##############################################

import numpy as np
from rvg import NumPyRVG

knode = np.dtype([
    ('location', int),
    ('indices', (int, 3)),
    ('keys', (int, 3)),
    ('is_leaf', int),
    ('num_keys', int)
])

knode_params = {
    'location'  : (0, 10),
    'indices'   : 42,
    'keys'      : 117,
    'is_leaf'   : (0, 2),
    'num_keys'  : (0, 256)
}

random_knode = NumPyRVG(dtype=knode)
knodes_array = random_knode(knode_params, 5)
print(knodes_array)

```
The output of the above script is a nested structured `numpy` array consisting of 5 `knode` structs, randomly initialized!

```
[(0, [-18, -11,  34], [  89,   35,  -57], 1, 189)
 (6, [-35,   0,   4], [ -56,  -65,   26], 1, 217)
 (4, [-29,  40,  37], [  93, -116,   91], 0,  38)
 (2, [-28, -42, -36], [-101,    0,  -43], 0,  82)
 (0, [-14, -19, -13], [ -98,  -46,  -78], 1, 238)]
```
If the script was run in an interpreter, you could inspect its type as well:

```Python console
>>> knodes_array
array([(0, [-18, -11,  34], [  89,   35,  -57], 1, 189),
       (6, [-35,   0,   4], [ -56,  -65,   26], 1, 217),
       (4, [-29,  40,  37], [  93, -116,   91], 0,  38),
       (2, [-28, -42, -36], [-101,    0,  -43], 0,  82),
       (0, [-14, -19, -13], [ -98,  -46,  -78], 1, 238)],
      dtype=[('location', '<i8'), ('indices', '<i8', (3,)), ('keys', '<i8', (3,)), ('is_leaf', '<i8'), ('num_keys', '<i8')])
```
The feature of nesting is not limited in arrays; you can create data types that are as complex as you want and/or need! See the relative [test](https://github.com/zehanort/rvg/blob/master/tests/test_NumPyRVG_class_with_types.py) as an example of struct nesting.
