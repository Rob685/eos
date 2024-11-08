'''
Implement Adam's preferred smoothing algorithm

Can benchmark against the default scipy alternatives...
'''
import time
from scipy.interpolate import RegularGridInterpolator as RGI
import numpy as np
import itertools

class AdamInterpolator(object):
    """
    Adam's algorithm for N-dimensional interpolation

    Intentionally on extrapolation
    """
    def __init__(self, *args, extrapolate=False):
        '''
        Initialize with AdamInterpolator(x1, x2... xn, z)

        x_i have shape (N,)
        z has shape (N, N, N...) up to n times

        asserts x_i are uniformly spaced on initialization
        '''
        super(AdamInterpolator, self).__init__()
        x_arrs = args[ :-1]

        self.extrapolate = False
        self.z = args[-1] # function to be interpolated
        self.N = len(x_arrs) # dimensionality of interpolant
        # store initial values along each axis + step size
        self.x0s = np.zeros(self.N)
        self.dxs = np.zeros(self.N)
        # store max extent of tables
        self.lengths = np.zeros(self.N)

        assert len(np.shape(self.z)) == self.N, (
            'Error in AdamInterpolator init: ' +
            'dimensions of function do not match number of axes' +
            ('(received a %d-D function but %d axes)'
             % (len(np.shape(self.z)), self.N))
        )

        for i, x in enumerate(x_arrs):
            # check that grid is uniformly spaced
            assert len(np.unique(np.diff(x))) == 1, (
                'Error in AdamInterpolator init: ' +
                'grid must be regularly spaced, ' +
                ('but entry %d is not' % i)
            )
            self.x0s[i] = x[0]
            self.dxs[i] = x[1] - x[0]
            self.lengths[i] = len(x)

    def __call__(self, *x):
        '''
        evaluates interpolant at (x1, x2, ..., xn)

        Description of algorithm: for a given input point x[], it takes
        contributions from all 2^N neighboring points:
            - Each point can be thought of as either left or right of x[], so
              there are 2^N such points
        Then, if a point is at a fraction $f$ in between x1_L and x1_R (the left
        and right grid boundaries), the contribution from the grid points along
        plane x1_L is a fraction $f$, and the contribution from grid points along the
        plane x1_R contributes fraction $(1 - f)$.

        In other words, mathematically, this looks like:

        f(x[]) = Sum_{x_1=x1_L}^x1_R Sum_{x_2=x2_L}^x2_R...Sum_{x_n=xn_L}^xn_R
            z[x_1, x_2,...] *
            Product_{i=1}^D
                (x[i] - x_1 if x_1 = x1_L
                 x_1 - x[i] if x_1 = x1_R)

        We avoid recursion (slow) by using itertools.product to compute all grid
        points at which this must be evaluated
        '''

        idxs = (x - self.x0s) / self.dxs
        idx_lefts = np.floor(idxs).astype(np.int32)
        frac_weights_left = 1 - (idxs - idx_lefts)
        frac_weights_right = (idxs - idx_lefts)

        if self.extrapolate:
            raise ValueError('AdamInterpolator does not support extrapolation atm')
        # guaranteed not to be extrapolating, continue

        if np.any((idx_lefts < 0) | (idx_lefts > (self.lengths - 2))):
            raise ValueError(
                'AdamInterpolator out of bounds: ' +
                'Argument was' + str(x)
            )

        offset_choices = itertools.product(range(2), repeat=self.N)
        def iterate_over_offsets(offsets):
            offsets = np.array(offsets, dtype=np.int32)
            ret = self.z[tuple(idx_lefts + offsets)] * np.product(
                offsets * frac_weights_right
                + (1 - offsets) * frac_weights_left)
            return ret
        contributions = map(
            iterate_over_offsets,
            offset_choices
        )
        return sum(contributions)

if __name__ == '__main__':
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)
    z = np.linspace(0, 10, 11)
    xgrid, ygrid, zgrid = np.meshgrid(x, y, z)
    f = xgrid**2 + ygrid**2 + zgrid**2
    interp = AdamInterpolator(x, y, z, f)
    interpRGI = RGI((x, y, z), f)

    # run perf tests
    N = 30000
    testo = np.reshape(
        np.random.uniform(0, 10, size=(3 * N)),
        (N, 3))
    sol_me = np.zeros(N)
    sol_RGI = np.zeros(N)

    print('Testing mine...')
    curr = time.time()
    for i in range(N):
        sol_me[i] = interp(*testo[i])
    print('%f seconds elapsed' % (time.time() - curr))

    print('Testing RGI...')
    curr = time.time()
    for i in range(N):
        sol_RGI[i] = interpRGI(testo[i])
    print('%f seconds elapsed' % (time.time() - curr))

    print('Max diff is', np.max(np.abs(sol_me - sol_RGI)))
