import argparse
import hickle as hkl
import numpy as np
from sklearn.random_projection import GaussianRandomProjection

DEFAULT_OUT_DIM = 256

def random_project(X, out_dim=DEFAULT_OUT_DIM, binary=False):
    
    if out_dim == -1:
        transformer = GaussianRandomProjection()     
    else:
        transformer = GaussianRandomProjection(out_dim)

    print 'random project %d data from dim (%d) to dim(%d)' % \
            (X.shape[0], X.shape[1], transformer.n_components)
    X_new = transformer.fit_transform(X)

    # binary hashing
    if binary:
        X_new[X_new > 0] = 1
        X_new[X_new < 0] = 0
        X_new = X_new.astype(np.bool)

    # (outDim, D) to (D, outDim)
    random_mat = transformer.components_.transpose()

    return X_new, random_mat
    

def project(X, random_mat, binary=False):
    if len(X.shape) == 1:
        X = X.reshape((1, -1))

    assert X.shape[1] == random_mat.shape[0], 'feature dim must be equal to random_mat.shape[0]'

    Y = X.dot(random_mat)

    return Y
    


if __name__ == '__main__':
    # Set up the command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('hkl_in',
            help='a .hkl file containing original NxD feature in data["feature"], where data = hkl.load(...). \
                  Set to "test" to run with auto-generated data.')
    parser.add_argument('hkl_out', nargs='?', 
            help='a .hkl file containing original NxD feature in data["feature"], where data = hkl.load(...)')
    parser.add_argument('--binary', action='store_true', help='whether to do binary hashing')
    parser.add_argument('-D', '--dim', default=DEFAULT_OUT_DIM, help='output feature dimension (default: %(default)s)')

    args = parser.parse_args()
    

    # load data
    if args.hkl_in != 'test':
        data = hkl.load(args.hkl_in)
        X = data['feature']
    else:
        data = dict()
        X = np.random.random((100, 10000))

    Xrp, random_mat = random_project(X, out_dim=args.dim, binary=args.binary)
    print 'random_mat:', random_mat.shape
    
    # output data
    if args.hkl_out:
        print 'writing to', args.hkl_out
        data['feature'] = Xrp
        hkl.dump(data, args.hkl_out)

