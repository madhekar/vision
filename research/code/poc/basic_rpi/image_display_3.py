import matplotlib.pyplot as plt
import numpy as np  

first = True
randArray = []

plt.show(block=False)
fig = plt.figure()
for i in range(0,5):
    #create a bunch of random numbers
    randArray = np.random.randint(0,50, size=(8,8))
    #print the array, just so I know you're not broken
    for x in randArray:
        print(x)
    #This is my attempt to clear.   
    if (first == False):
        plt.clf()
    first = False
    #basical visualization
    ax = fig.add_subplot(111)
    ax.imshow(randArray, cmap='hot', interpolation='nearest')
    fig.canvas.draw()
    fig.canvas.flush_events()

    print("Pausing...")
    plt.pause(1)



""" a = np.array([[1,2,3],[4,5,6],[7,8,9]])
plt.ion()
plt.imshow(a)
for t in range(0,4):
    for i in range(0,a.shape[0]):
        for j in range(0,a.shape[1]):
            a[i][j] *= 2.9

    plt.imshow(a)
    plt.pause(1)
    #plt.close()
plt.show() """

""" for i in range(10):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.05)

plt.show() """
