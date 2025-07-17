# This question already has answers here:
# How to apply a function on every row on a dataframe? (2 answers)
# Closed 3 years ago.
# I have to calculate the distance on a hilbert-curve from 2D-Coordinates. With the hilbertcurve-package i built my own "hilbert"-function, to do so. The coordinates are stored in a dataframe (col_1 and col_2). As you see, my function works when applied to two values (test).

# However it just does not work when applied row wise via apply-function! Why is this? what am I doing wrong here? I need an additional column "hilbert" with the hilbert-distances from the x- and y-coordinate given in columns "col_1" and "col_2".

import pandas as pd
from hilbertcurve.hilbertcurve import HilbertCurve

df = pd.DataFrame({'ID': ['1', '2', '3'],
                   'col_1': [0, 2, 3],
                   'col_2': [1, 4, 5]})


def hilbert(x, y):
    n = 2
    p = 7
    hilcur = HilbertCurve(p, n)
    dist = hilcur.distance_from_coordinates([x, y])
    return dist


test = hilbert(df.col_1[2], df.col_2[2])

df["hilbert"] = df.apply(hilbert(df.col_1, df.col_2), axis=0)


#---
df.apply(lambda x: hilbert(x['col_1'], x['col_2']), axis=1)