'''
Implement Adam's preferred smoothing algorithm

Can benchmark against the default scipy alternatives...
'''
import time
from scipy.interpolate import RegularGridInterpolator as RGI
import numpy as np
import itertools

# class AdamInterpolator(object):
#     """
#     Adam's algorithm for N-dimensional interpolation

#     Intentionally on extrapolation
#     """
#     def __init__(self, *args, extrapolate=False):
#         '''
#         Initialize with AdamInterpolator(x1, x2... xn, z)

#         x_i have shape (N,)
#         z has shape (N, N, N...) up to n times

#         asserts x_i are uniformly spaced on initialization
#         '''
#         super(AdamInterpolator, self).__init__()
#         x_arrs = args[ :-1]

#         self.extrapolate = False
#         self.z = args[-1] # function to be interpolated
#         self.N = len(x_arrs) # dimensionality of interpolant
#         # store initial values along each axis + step size
#         self.x0s = np.zeros(self.N)
#         self.dxs = np.zeros(self.N)
#         # store max extent of tables
#         self.lengths = np.zeros(self.N, dtype=np.int32)

#         assert len(np.shape(self.z)) == self.N, (
#             'Error in AdamInterpolator init: ' +
#             'dimensions of function do not match number of axes' +
#             ('(received a %d-D function but %d axes)'
#              % (len(np.shape(self.z)), self.N))
#         )

#         for i, x in enumerate(x_arrs):
#             # check that grid is uniformly spaced
#             assert len(np.unique(np.diff(x))) == 1, (
#                 'Error in AdamInterpolator init: ' +
#                 'grid must be regularly spaced, ' +
#                 ('but entry %d is not' % i)
#             )
#             self.x0s[i] = x[0]
#             self.dxs[i] = x[1] - x[0]
#             self.lengths[i] = len(x)

#     def __call__(self, *x):
#         '''
#         evaluates interpolant at (x1, x2, ..., xn)

#         Description of algorithm: for a given input point x[], it takes
#         contributions from all 2^N neighboring points:
#             - Each point can be thought of as either left or right of x[], so
#               there are 2^N such points
#         Then, if a point is at a fraction $f$ in between x1_L and x1_R (the left
#         and right grid boundaries), the contribution from the grid points along
#         plane x1_L is a fraction $f$, and the contribution from grid points along the
#         plane x1_R contributes fraction $(1 - f)$.

#         In other words, mathematically, this looks like:

#         f(x[]) = Sum_{x_1=x1_L}^x1_R Sum_{x_2=x2_L}^x2_R...Sum_{x_n=xn_L}^xn_R
#             z[x_1, x_2,...] *
#             Product_{i=1}^D
#                 (x[i] - x_1 if x_1 = x1_L
#                  x_1 - x[i] if x_1 = x1_R)

#         We avoid recursion (slow) by using itertools.product to compute all grid
#         points at which this must be evaluated
#         '''

#         x = np.array(x, dtype=np.float64)

#         idxs = (x - self.x0s) / self.dxs
#         idx_lefts = np.floor(idxs).astype(np.int32)
#         frac_weights_left = 1 - (idxs - idx_lefts)
#         frac_weights_right = (idxs - idx_lefts)

#         if self.extrapolate:
#             raise ValueError('AdamInterpolator does not support extrapolation atm')
#         # guaranteed not to be extrapolating, continue

#         if np.any((idx_lefts < 0) | (idx_lefts > (self.lengths - 2))):
#             raise ValueError(
#                 'AdamInterpolator out of bounds: ' +
#                 'Argument was' + str(x)
#             )

#         offset_choices = itertools.product(range(2), repeat=self.N)
#         def iterate_over_offsets(offsets):
#             offsets = np.array(offsets, dtype=np.int32)
#             ret = self.z[tuple(idx_lefts + offsets)] * np.product(
#                 offsets * frac_weights_right
#                 + (1 - offsets) * frac_weights_left)
#             return ret
#         contributions = map(
#             iterate_over_offsets,
#             offset_choices
#         )
#         return sum(contributions)

import numpy as np
import itertools

class AdamInterpolator:
    """
    Adam's algorithm for N-dimensional interpolation with linear extrapolation support.
    """
    def __init__(self, *args, extrapolate=False):
        '''
        Initialize with AdamInterpolator(x1, x2, ..., xn, z)

        Parameters:
        - x_i: arrays of shape (Ni,), representing the grid points along each axis.
        - z: N-dimensional array of shape (N1, N2, ..., Nn), representing the function values
             at the grid points.

        The grid must be uniformly spaced along each axis, but the number of points (Ni) can
        differ between axes.

        The function asserts that x_i are uniformly spaced on initialization.
        '''
        super(AdamInterpolator, self).__init__()
        x_arrs = args[:-1]
        self.z = args[-1]  # Function values to be interpolated
        self.N = len(x_arrs)  # Dimensionality of the interpolant

        self.extrapolate = extrapolate

        # Store initial values along each axis and step sizes
        self.x0s = np.zeros(self.N)
        self.dxs = np.zeros(self.N)
        # Store the lengths of each axis
        self.lengths = np.zeros(self.N, dtype=np.int32)

        assert len(self.z.shape) == self.N, (
            'Error in AdamInterpolator init: '
            'dimensions of function do not match number of axes '
            f'(received a {len(self.z.shape)}-D function but {self.N} axes)'
        )

        for i, x in enumerate(x_arrs):
            if len(x) < 2:
                raise ValueError(f"Grid array x[{i}] must contain at least two points.")
            # Check that grid is uniformly spaced within a tolerance
            dxs = np.diff(x)
            if not np.allclose(dxs, dxs[0], rtol=1e-5, atol=1e-8):
                raise ValueError(
                    f'Error in AdamInterpolator init: grid must be regularly spaced, '
                    f'but entry {i} is not.'
                )
            self.x0s[i] = x[0]
            self.dxs[i] = dxs[0]
            self.lengths[i] = len(x)

    def __call__(self, *x):
        '''
        Evaluates the interpolant at a point (x1, x2, ..., xn), supporting linear extrapolation.

        Parameters:
        - x: Coordinates at which to evaluate the interpolant.

        Returns:
        - Interpolated or extrapolated value at the given point.

        The algorithm computes contributions from all 2^N neighboring grid points surrounding
        the input point. It uses the fractional distances to the neighboring grid points
        to compute weights for linear interpolation or extrapolation along each axis.
        '''
        x = np.array(x, dtype=np.float64)
        idxs = (x - self.x0s) / self.dxs
        idx_lefts = np.floor(idxs).astype(np.int32)
        idx_rights = idx_lefts + 1

        # Fractions for left and right grid points
        frac_right = idxs - idx_lefts
        frac_left = 1 - frac_right

        # Handle extrapolation by extending indices and fractions beyond grid bounds
        idx_lefts_clipped = np.clip(idx_lefts, 0, self.lengths - 2)
        idx_rights_clipped = idx_lefts_clipped + 1

        frac_left = np.where(idx_lefts < 0, 1 + idxs, frac_left)
        frac_left = np.where(idx_rights > self.lengths - 1, -idxs + (self.lengths - 1), frac_left)
        frac_right = 1 - frac_left

        # Ensure weights are between 0 and 1
        frac_left = np.clip(frac_left, 0, 1)
        frac_right = np.clip(frac_right, 0, 1)

        offset_choices = itertools.product(range(2), repeat=self.N)

        def iterate_over_offsets(offsets):
            offsets = np.array(offsets, dtype=np.int32)
            indices = idx_lefts_clipped + offsets
            weights = np.where(
                offsets == 0,
                frac_left,
                frac_right
            )
            ret = self.z[tuple(indices)] * np.prod(weights)
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
