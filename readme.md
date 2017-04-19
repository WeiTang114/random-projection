# Random-Projection

A python-based tool to generate random matrices, and do random projection on high dimensional data for feature dimension reduction and binarizaiton.

## Usage

### Generate Random Matrix

```bash
python generate_random_projection_matrix.py <Dim in> <Dim out> <Outfile>

# 9216 -> 256, save to random_mat.npy
python generate_random_projection_matrix.py 9216 256 random_mat.npy
```

### Project data

Data should be in .hkl format, including at least a field "feature". "feature" is a numpy array of shape (#data N, #dim D).

```
>>> import hickle as hkl
>>> data = hkl.load('data.hkl')
>>> data['feature'].shape  # will be (N x D), N: #data, D: #dim
(1000, 9216)
```

Do the projection with random_projection.py:

```bash
python random_projection.py [--binary] [-M random_mat.npy] [-D dim] in_data.hkl out_data.hkl

# help
python random_proejction.py --help
```

Example 1: Random projection to 256-D:

```bash
python random_proejction.py -D 256 in.hkl out.hkl
```

Example 2: project with a pre-computed matrix and binarize to (0, 1):

```bash
python random_proejction.py --binary -M random_mat.npy in.hkl out.hkl
```

# About

Author: Lee, Tang
