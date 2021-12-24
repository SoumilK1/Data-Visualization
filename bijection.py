from numpy import exp, log
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap


def get_chunks(x):
    chunks = []
    place = 0.1
    chunk = []
    for _ in range(20):
        digit = x // place
        x -= digit * place
        chunk.append(digit)

        if digit != 0.0:
            chunks.append(chunk)
            chunk = []

        place *= 0.1

    return chunks


def interleave(chunk1, chunk2):
    chunk = []
    alter = True
    while chunk1 and chunk2:
        if alter:
            chunk.append(chunk1.pop(0))
        else:
            chunk.append(chunk2.pop(0))
        alter = not alter

    return chunk


def join(chunks):
    num = 0
    digits = []
    for chunk in chunks:
        digits.extend(chunk)

    for i in range(len(digits)):
        num += 10**(-i - 1) * digits[i]

    return num


def RtoUnit(x):
    return 1 / (1 + exp(-x))


def unitToR(x):
    return log(x / (1 - x))


@np.vectorize
def f(x, y):
    return unitToR(join(interleave(get_chunks(RtoUnit(x)), get_chunks(RtoUnit(y)))))


def main():
    n = 100
    _x = np.linspace(-10, 10, n)
    _y = np.linspace(-10, 10, n)

    xx, yy = np.meshgrid(_x, _y, sparse=True)
    z = f(xx, yy)

    # COLORMAP
    # plt.imshow(z, extent=(min(_x), max(_x), min(_y), max(_y)), origin="lower")
    #
    # plt.ylabel(r"$x$")
    # plt.xlabel(r"$y$")
    #
    # plt.xlim([_x.min(), _x.max()])
    # plt.ylim([_y.min(), _y.max()])
    #
    # plt.colorbar()
    #
    # plt.show()

    # distances = []
    # diff = []
    #
    # for i in range(n):
    #     for j in range(n):
    #         for x in range(n):
    #             for y in range(n):
    #                 distances.append(((_x[i] - _x[x])**2 + (_y[j] - _y[y])**2)**0.5)
    #                 diff.append(abs(z[y, x] - z[j, i]))
    #
    # plt.scatter(distances, diff)
    # plt.show()

    # 3D PLOT
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, yy, z)
    plt.show()


if __name__ == '__main__':
    main()
