import numpy as np
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import logging



fig,ax = plt.subplots()
identifiers = list(range(1,14))
first_plot  = True

for identifier in identifiers:
    trajectory = np.load(os.path.join( os.path.dirname(__file__),"states",f"state_history_{identifier}.npy"))   
    state      = trajectory[:,1:]
    time       = trajectory[:,0]
    
    if first_plot == 1 :
        ax.scatter(state[0,0],state[0,1],marker="o",color="green", label="Initial position")
        ax.scatter(state[-1,0],state[-1,1],marker="o",color="red", label = "Final position")
        first_plot = False
    else :
        ax.scatter(state[0,0],state[0,1],marker="o",color="green")
        ax.scatter(state[-1,0],state[-1,1],marker="o",color="red")
    
    ax.plot(state[:,0],state[:,1],label=f"Agent {identifier}")


ax = plt.gca()
# Display the background image, ensuring it covers the axis limits
background_image = mpimg.imread("/home/gregorio/Desktop/papers/journal/decentralized_STL_task_decomposition/code/assets/thira.png")
ax.imshow(background_image, extent=[-25,25, -7.5,7.5], aspect='auto')
plt.show()
