import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style 
import time  

fig = plt.figure()
axl = fig.add_subplot(1, 1, 1)

def animate(i):
    pullData = open("negative.txt" , "r").read()
    lines = pullData.split('\n')

    xar = []
    yar = []

    x = 0 
    y = 0 

    for l in lines:
        x += 1
        if "pos" in l :
            y += 1 
        elif "neg" in l :
            y -= 1
    axl.clear()
    axl.plot(xar, yar)

ani = animation.FuncAnimation(fig , animate , interval = 1000)
plt.show()


