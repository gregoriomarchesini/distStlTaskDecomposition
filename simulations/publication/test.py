import numpy as np
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import logging


plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

identifiers = list(range(1,16))
first_plot  = True

split_times = [0., 7., 15., 30., 40.]

figs, axs = [],[]
for ii in range(len(split_times)-1) :
    fig,ax = plt.subplots()
    figs.append(fig)
    axs.append(ax)



for identifier in identifiers:
    trajectory = np.load(os.path.join( os.path.dirname(__file__),"states",f"state_history_{identifier}.npy"))   
    state      = trajectory[:,1:]
    time       = trajectory[:,0]
    print(time[-1])
    state_sections = []
    
    for jj in range(0,len(axs)) :
        select_times = np.bitwise_and(time >= split_times[jj], time <= split_times[jj+1])
        state_sections.append(state[select_times,:].copy())

    for ii,state_section in enumerate(state_sections) :
        
        ax = axs[ii]
        if identifier == 1:
            ax.scatter(state_section[:,0],state_section[:,1],label=fr'Agent_{identifier}', marker="o", s=5, c="red")
        else :
            ax.scatter(state_section[:,0],state_section[:,1],label=fr'Agent_{identifier}', marker="o", s=2, c="green")


# Display the background image, ensuring it covers the axis limits
background_image = mpimg.imread("/home/gregorio/Desktop/papers/journal/decentralized_STL_task_decomposition/code/assets/thira.png")
for ax in axs :
    ax.imshow(background_image, extent=[-25,25, -7.5,7.5], aspect='auto')
    ax.set_xlabel(r'Km')
    ax.set_ylabel(r'Km')
    ax.set_xlim(-11.5,24.4)
    
    
results = np.load(os.path.join( os.path.dirname(__file__),"decomposition_result","results.npy"),allow_pickle=True).item()
penalties = results["penalties"]
cost      = results["cost"]
cost      = results["cost"]

    
# plt.show()
