import numpy as np
import perfplot
import math


def f(x):
    # return math.sqrt(x)
    return np.sqrt(x)


vf = np.vectorize(f)


def array_for(x):
    return np.array([f(xi) for xi in x])


def array_map(x):
    return np.array(list(map(f, x)))


def fromiter(x):
    return np.fromiter((f(xi) for xi in x), x.dtype)


def vectorize(x):
    return np.vectorize(f)(x)


def vectorize_without_init(x):
    return vf(x)


b = perfplot.bench(
    setup=np.random.rand,
    n_range=[2 ** k for k in range(20)],
    kernels=[
        f,
        array_for,
        array_map,
        fromiter,
        vectorize,
        vectorize_without_init,
    ],
    xlabel="len(x)",
)
b.save("out1.svg")
b.show()