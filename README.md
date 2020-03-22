# &#127922; rvg &#127922; - Random Values Generator
![Testing](https://github.com/zehanort/rvg/workflows/Testing/badge.svg)
![PyPI release](https://img.shields.io/pypi/v/rvg?label=PyPI%20release)

## Description

`rvg` is a Python 3 package utility to create random values of any Python 3 data type.

Its main purpose is to help in applications where reliable -in terms of type safety- random values are needed (e.g. statistics, machine learning, general testing etc), and in specific layouts (e.g. "I want a `numpy` structured array of random pairs of ints and floats and I want it now").

## Installation

Either download the source code from the [releases page](https://github.com/zehanort/rvg/releases) or use `pip`:
```
pip install rvg
```

## Current Status

### Pre-release

For the time being, only `numpy` types are supported. More specifically:
- `numpy` scalar data types
- `numpy` arrays of scalar data types
- `numpy` arrays of structured data types

After the alpha release, more features will be implemented, focusing mainly on Python 3 native types.

## Usage

Right now, `rvg` provides the `NumPyRVG` interface to generate random `numpy` values.
```Python console
>>> from rvg import NumPyRVG
>>> import numpy as np
>>> rand = NumPyRVG(limit=1000) # `limit`: lower and upper limit of numeric values to be generated
```
The functionalities of `NumPyRVG` include the generation of:
- `numpy` scalar data types:
```Python console
>>> rand(np.int8)
78
>>> rand(np.uint16)
866
>>> i = rand(np.int8)
>>> u = rand(np.uint16)
>>> f = rand(np.float32)
>>> d = rand(np.double)
>>> i, u, f, d
(-101, 720, -234.16493, -882.7847115143803)
>>> type(i), type(u), type(f), type(d)
(<class 'numpy.int8'>, <class 'numpy.uint16'>, <class 'numpy.float32'>, <class 'numpy.float64'>)
```
- `numpy` array data types from scalar types:
```Python console
>>> rand(np.uint8, length=3) # `length`: length of the output numpy array
array([ 72, 222, 146], dtype=uint8)
>>> rand(np.float32, length=4)
array([ 939.84973, -903.3939 , -805.2647 , -676.1155 ], dtype=float32)
```
- `numpy` structured array data types (structured datatypes can be nested):
```Python console
>>> type1 = np.dtype('int,float')
>>> type2 = np.dtype('double,uint,long')
>>> rand(type1)
(849, -626.7902777962995)
>>> type(rand(type1)) # single values that correspond to structured data types
<class 'tuple'>       # are considered to be tuples
>>> rand(type1, length=3)
array([(-594,  451.64958214), ( 965,  -22.77642568),
       (-713, -100.61156315)], dtype=[('f0', '<i8'), ('f1', '<f8')])
>>> rand(type2, length=3)
array([(506.01690599,  48,  946), (643.06826309, 363, -865),
       (264.05285682, 214,  395)],
      dtype=[('f0', '<f8'), ('f1', '<u8'), ('f2', '<i8')])
>>> type3 = np.dtype([('t1', type1), ('t2', type2)]) # a nested structured data type
>>> rand(type3, length=2)
array([((716,  434.4316939), (-143.98226673, 610, -354)),
       ((307, -894.4985342), ( 234.0804783 , 678,  327))],
      dtype=[('t1', [('f0', '<i8'), ('f1', '<f8')]), ('t2', [('f0', '<f8'), ('f1', '<u8'), ('f2', '<i8')])])
```
