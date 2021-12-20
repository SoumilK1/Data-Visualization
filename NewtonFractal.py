"""
plots the fractal of a polynomial in complex plane
"""
import numpy as np
from numpy import pi
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import newton

from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap


def fractal(f, df_dx, roots, center=0j, dims=10+10j, xres=1000, yres="auto", maxiter=10, file="Newton.png"):
    cmap = get_cmap("viridis", len(roots))
    colors = np.array([cmap(i) for i in range(cmap.N)])

    if yres == "auto":
        yres = int(xres * dims.imag / dims.real)

    x, y = np.meshgrid(np.linspace(center.real - dims.real / 2, center.real + dims.real / 2, xres),
                       np.linspace(center.imag - dims.imag / 2, center.imag + dims.imag / 2, yres))
    mesh = x + 1j * y

    def color(z):
        distances = np.array([abs(r - z) for r in roots])
        return colors[min(range(len(roots)), key=distances.__getitem__)]

    plt.imsave(file, np.vectorize(color, signature="()->(4)")(newton(f, mesh, fprime=df_dx, maxiter=maxiter)))


def polyFractal(p, *args, **kwargs):
    poly = Polynomial(p)
    fractal(poly, poly.deriv(), poly.roots(), *args, **kwargs)


fractal(lambda x: np.sin(x) / x,
        lambda x: (x * np.cos(x) - np.sin(x)) / x**2,
        [i * pi + 0j for i in range(-5, 5 + 1, 1)],
        dims=2+1j)
