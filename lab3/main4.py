'''

    x* = 2.0

       i:  1      2       3       4       5
    x(i): 0.0    1.0     2.0     3.0     4.0
    y(i): 0.0    2.0     3.4142  4.7321  6.0


    y'  - ?
    y'' - ?
    
'''
from bisect import bisect
import numpy as np
import matplotlib.pyplot as plt

def find_position(x, y, X):
    li, ri = max(0, bisect(x, X)-1), min(len(x)-1, bisect(x, X))
    i = li if X - x[li] < x[ri] - X else ri
    if i == 0: i += 1
    elif i == len(x)-1: i -=1
    return i

def df(x, y, X):
    i = find_position(x, y, X)
    return (y[i]-y[i-1])/(x[i]-x[i-1])       \
        + ((y[i+1]-y[i])/(x[i+1]-x[i])       \
        - (y[i]-y[i-1])/(x[i]-x[i-1]))       \
        / (x[i+1]-x[i-1])                    \
        * (2*X-x[i-1]-x[i])

def ddf(x, y, X):
    i = find_position(x, y, X)
    return 2 * \
        ((y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1])) \
        / (x[i+1]-x[i-1])


def main():
    x = [0.0 ,   1.0  ,   2.0  ,   3.0   ,  4.0]
    y = [0.0 ,   2.0  ,   3.4142,  4.7321,  6.0]
    X = 2.0


    df_ = df(x, y, X) 
    ddf_ = ddf(x, y, X)

    print(f"f'({X}) = {df_}")
    print(f"f''({X}) = {ddf_}") 

if __name__ == "__main__":
    main()