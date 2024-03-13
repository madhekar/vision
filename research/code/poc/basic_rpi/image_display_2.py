import matplotlib.pyplot as plt
import numpy as np
import time

def demo(a):
    y = [xt*a+1 for xt in x]
    ax.plot(x,y)

if __name__ == '__main__':
    #plt.ion()
    fig, ax = plt.subplots()
    ax.set_ylim([0,15])
    x = range(0,5)
    for a in range(1,4):
        demo(a)
        plt.pause(3)
        plt.draw()