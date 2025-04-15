import numpy as np
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
# swt ticks size
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
# change size of label also
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14

identifiers = list(range(1,16))
first_plot  = True

split_times = [0., 20., 20.01, 39.98, 40.0]




fig = plt.figure(layout="constrained")
ax_dict = fig.subplot_mosaic(
    [
        ["0", "1"],
        ["2", "3"],
    ],
)

stepping_points = 100
for identifier in identifiers:
    trajectory = np.load(os.path.join( os.path.dirname(__file__),"states",f"state_history_{identifier}.npy"))   
    state      = trajectory[:,1:]
    time       = trajectory[:,0]
    state_sections = []
    
    for jj in range(0,len(split_times)-1) :
        select_times = np.bitwise_and(time >= split_times[jj], time <= split_times[jj+1])
        state_sections.append(state[select_times,:].copy())

    for ii,state_section in enumerate(state_sections) :
        
        ax = ax_dict[str(ii)]
        
        if ii in [1,3] :
            if identifier == 1:
                ax.scatter(state_section[::stepping_points,0],state_section[::stepping_points,1],label=fr'Agent_{identifier}', marker="o", s=20, c="red")
            else :
                ax.scatter(state_section[::stepping_points,0],state_section[::stepping_points,1],label=fr'Agent_{identifier}', marker="o", s=20, c="green")
        else : # snapshot images
            if identifier == 1:
                ax.scatter(state_section[::stepping_points,0],state_section[::stepping_points,1],label=fr'Agent_{identifier}', marker="o", s=5, c="red")
            else :
                ax.scatter(state_section[::stepping_points,0],state_section[::stepping_points,1],label=fr'Agent_{identifier}', marker="o", s=5, c="green")
            

# Display the background image, ensuring it covers the axis limits
background_image = mpimg.imread("/home/gregorio/Desktop/papers/journal/decentralized_STL_task_decomposition/code/assets/thira.png")
for jj in range(4) :
    ax_dict[str(jj)].imshow(background_image, extent=[-25,25, -7.5,7.5], aspect='auto')
    ax_dict[str(jj)].set_xlabel(r'Km')
    ax_dict[str(jj)].set_ylabel(r'Km')
    ax_dict[str(jj)].set_xlim(-11.5,24.4)
    ax_dict[str(jj)].text(-10, -6.5, str(jj), fontsize=17,color="white")


fig = plt.figure(layout="constrained")
ax_dict = fig.subplot_mosaic(
    [

        ["penalties","penalties"],
        ["accuracy" ,"accuracy"],
    ],
)


    
cost_end_penalties = np.load(os.path.join( os.path.dirname(__file__),"decomposition_result","cost_and_penalties.npy"),allow_pickle=True)[()]
decomposition_accuracy_per_task = np.load(os.path.join( os.path.dirname(__file__),"decomposition_result","decomposition_accuracy_per_task.npy"),allow_pickle=True)[()] # this indexing is because for some reason the pickling saves the dict as a zero numpy array


fig,axs = plt.subplots(2,1)
axins   = (zoomed_inset_axes(ax_dict["accuracy"], 5.00, loc=1),zoomed_inset_axes(ax_dict["penalties"], 7.50, loc='center right'))

num_iterations = len(list(cost_end_penalties.values())[0]["penalties"])

for task_id,accuracy in decomposition_accuracy_per_task.items() :
    ax_dict["accuracy"].plot(range(0,num_iterations),accuracy,label=fr'Task_{task_id}')
    axins[0].plot(range(3000,num_iterations),accuracy[3000:])

axins[0].set_ylim(0.7,1.1)
ax_dict["accuracy"].grid()
ax_dict["accuracy"].set_ylabel(r"Accuracy")
# show penalties


for edge, results in cost_end_penalties.items():
    ax_dict["penalties"].plot(results["penalties"]) 
    axins[1].plot(range(280,650),results["penalties"][280:650])

ax_dict["penalties"].grid()
ax_dict["penalties"].set_ylabel(r"Penalties $\rho_i$")
ax_dict["accuracy"].set_xlabel("Iterations")

mark_inset(ax_dict["accuracy"], axins[0], loc1=3, loc2=4, fc="none", ec="0.5")
mark_inset(ax_dict["penalties"], axins[1], loc1=3, loc2=4, fc="none", ec="0.5")
axins[0].grid()
axins[1].grid()
# axins[1].aspect= 200

    
plt.show()
