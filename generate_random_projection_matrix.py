import sys
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
import numpy as np


in_dim = int(sys.argv[1])
out_dim = int(sys.argv[2])
out_file = sys.argv[3]

# dummy data
X = np.zeros((2, in_dim), dtype=float)

g = GaussianRandomProjection(out_dim)
g.fit_transform(X)

# random mat, transpose() from (out_d, int_d) to (in_d, out_d)
random_mat = g.components_.transpose()

random_mat.dump(out_file)

