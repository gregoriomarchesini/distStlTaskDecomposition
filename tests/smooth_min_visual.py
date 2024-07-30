from stl.stl import *

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt



# Test Gamma Function
gamma = GammaFunction(gamma_0=5.,time_flattening=10.,t_0 = 0.)
gamma.plot()

# try derivative
gamma_prime = gamma.compute_gradient(2)
print(gamma_prime)

# try switch functions
switch = SwitchOffFunction(switching_time=6.)
switch.plot()



polytope    = regular_2D_polytope(5,1)
predicate   = CollaborativePredicate(polytope,source_agent_id=0,target_agent_id=1,center=np.array([0,0]))
task1        = G(5,10)  @ predicate

predicate   = CollaborativePredicate(polytope,source_agent_id=0,target_agent_id=1,center=np.array([3,3]))
task2        = G(5,10) @ predicate



agents   = {0: np.array([2,3]),1: np.array([1,3])}
edge_vec = agents[1]- agents[0]
barriers1 : list[CollaborativeLinearBarrierFunction] = create_linear_barriers_from_task(task  = task1 , 
                                                                                       initial_conditions = agents, 
                                                                                       t_init = 0,
                                                                                       maximum_control_input_norm=1.5)

barriers2 : list[CollaborativeLinearBarrierFunction] = create_linear_barriers_from_task(task  = task2 , 
                                                                                       initial_conditions = agents, 
                                                                                       t_init = 0,
                                                                                       maximum_control_input_norm=1.5)

sm1 = CollaborativeSmoothMinBarrierFunction(list_of_barrier_functions= barriers1 ,eta = 25)
sm2 = CollaborativeSmoothMinBarrierFunction(list_of_barrier_functions= barriers2 ,eta = 25)
sm3 = CollaborativeSmoothMinBarrierFunction(list_of_barrier_functions= barriers1 + barriers2 ,eta = 25)

fig,ax = plt.subplots(1,3)

def plot_contour(smooth_min,ax,t):
    x = np.linspace(-8,8,100)
    y = np.linspace(-8,8,100)
    X,Y = np.meshgrid(x,y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]) :
        for j in range(X.shape[1]) :
            try :
                Z[i,j] = smooth_min.compute(x_target=np.array([[X[i,j]],[Y[i,j]]]),
                                            x_source=np.zeros((2,1)) ,
                                            t=t)
            except :
                raise NotImplementedError("plotting only allowed for 2D states")
    return ax.contourf(X,Y,Z,cmap=plt.cm.bone)


cs1 = plot_contour(sm1,ax[0],0)
cs2 = plot_contour(sm2,ax[1],0)
cs3 = plot_contour(sm3,ax[2],0)

ax[0].scatter(edge_vec[0],edge_vec[1],c='r',s=10)
ax[1].scatter(edge_vec[0],edge_vec[1],c='r',s=10)
ax[2].scatter(edge_vec[0],edge_vec[1],c='r',s=10)

cbar1 = fig.colorbar(cs1)
cbar2 = fig.colorbar(cs2)
cbar3 = fig.colorbar(cs3)




plt.show()