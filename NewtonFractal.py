"""
plots the fractal of a polynomial in complex plane
"""
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import newton

from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb

colors = [to_rgb(c) for c in ["#6EFF6A", "#FF4178", "#5990FF"]]


def plotFractal(p):
    poly = Polynomial(p)
    roots = poly.roots()

    x, y = np.meshgrid(np.linspace(-5, 5, 1000), np.linspace(-5, 5, 1000))
    mesh = x + 1j * y

    def color(z):
        distances = np.array([abs(r - z) for r in roots])
        return colors[min(range(3), key=distances.__getitem__)]

    plt.imshow(np.vectorize(color, signature="()->(3)")(newton(poly, mesh, fprime=poly.deriv(), maxiter=10)))
    plt.show()


plotFractal([-1, 0, 0, 1])
