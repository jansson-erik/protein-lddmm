import numpy as np
import matplotlib.pyplot as plt

class Grid:
    """
    Class for representing a grid.
    """
    def __init__(self, res, span):
        """
        Parameters
        ----------
        res : int, the number of grid points in the x and y directions.
        span : float, the span of the grid
        """
        self.res = res
        self.span = span
        self.ds = 2 * span / (res - 1)

        self.xs = np.linspace(-span, span, num=res)
        self.ys = np.linspace(-span, span, num=res)

        self.x, self.y = np.meshgrid(self.xs, self.ys, indexing='xy')

    def r2_to_grid(self, p):
        """
        Converts a 2D point to a grid index.

        Parameters
        ----------
        """
        return [int(round((p[0] + self.span) / self.ds)), int(round((p[1] + self.span) / self.ds))]

    def grid_to_r2(self, i, j):
        """
        Converts a grid index to a 2D point.

        Parameters
        ----------
        i : int
        j : int
        Returns
        -------
        p : array of shape (2,)
        """
        return np.array([self.xs[i], self.ys[j]])

    def create_kernel(self, sigma):
        """
        Creates a 2D Gaussian kernel.

        Parameters
        ----------
        sigma : float
        Returns
        -------
        None 
        """
        return None