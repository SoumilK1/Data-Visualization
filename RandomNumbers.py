from math import log2
from random import randint
from time import perf_counter

import numpy as np


def generator1(a, b):
    add = 0.25
    pos = 0.5
    n = b-a+1
    while add * n**3 > 1:
        if randint(0, 1):
            pos += add
        else:
            pos -= add
        add /= 2
        
    return int(a + pos*n)


def generator2(a, b):
    while True:
        r = randint(0, 1)
        for _ in range(int(log2(b-a))):
            r <<= 1
            r += randint(0, 1)

        if a + r <= b:
            return a + r


a, b = 0, 5
N = 1000000

for generator in (generator1, generator2, randint):
    t = perf_counter()
    rands = [generator(a, b) for _ in range(N)]
    t = perf_counter() - t

    print(f"""
{generator.__name__}:
    mean = {np.mean(rands)}
    variance = {np.var(np.bincount(rands)*(b-a+1)/N)}
    time = {t / N}
""")
