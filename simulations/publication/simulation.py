import numpy as np
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import logging

from stl.stl               import CollaborativePredicate,IndependentPredicate , regular_2D_polytope, G,F
from stl.graphs        import TaskGraph, CommunicationGraph, normalize_graphs, show_graphs
from simulator.sim        import MultiAgentSystem, simulate_agents
from stl.dynamics          import SingleIntegrator2D
from stlddec.decomposition import run_task_decomposition

logging.basicConfig(level=logging.ERROR)

# set seeds for replication
np.random.seed(100)

###########################################################################################
########################      Task Graph     ##############################################
###########################################################################################


task_graph = TaskGraph()

# self task
# edge 1-1

polytope     = regular_2D_polytope(5,3)
predicate    = IndependentPredicate( polytope_0 = polytope, center = np.array([5.,0.]), agent_id =1 ) 
task         = F(10,15) @ predicate
task_graph.attach(task)

# outer ring
# edge 1-12
polytope     = regular_2D_polytope(5,3)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([8.,8.]), source_agent_id =1, target_agent_id =12 )
task         = F(10,15) @ predicate
task_graph.attach(task)

# edge 1-13
polytope     = regular_2D_polytope(5,3)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([8.,-8.]), source_agent_id =1, target_agent_id =13 )
task         = F(10,15) @ predicate
task_graph.attach(task)

# edge 1-10
polytope     = regular_2D_polytope(5,3)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([-8.,8.]), source_agent_id =1, target_agent_id =10 )
task         = F(10,15) @ predicate
task_graph.attach(task)

# edge 1-8
polytope     = regular_2D_polytope(5,3)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([-8.,0.]), source_agent_id =1, target_agent_id =8 )
task         = F(10,15) @ predicate
task_graph.attach(task)

# edge 1-6
polytope     = regular_2D_polytope(5,3)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([-8.,-8.]), source_agent_id =1, target_agent_id =7 )
task         = F(10,15) @ predicate
task_graph.attach(task)

# edge 1-11
polytope     = regular_2D_polytope(4,4)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([-18.,0.]), source_agent_id =1, target_agent_id =11 )
task         = F(10,15) @ predicate
task_graph.attach(task)


# inner ring
# edge 1-4
polytope     = regular_2D_polytope(3,2)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([3.,5.]), source_agent_id =1, target_agent_id =4 )
task         = G(10,15) @ predicate
task_graph.attach(task)

# edge 1-2
polytope     = regular_2D_polytope(3,2)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([3.,-5.]), source_agent_id =1, target_agent_id =2 )
task         = G(10,15) @ predicate
task_graph.attach(task)

# edge 1-6
polytope     = regular_2D_polytope(3,2)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([-3.,-5.]), source_agent_id =1, target_agent_id =6 )
task         = G(10,15) @ predicate
task_graph.attach(task)

# edge 1-9
polytope     = regular_2D_polytope(3,2)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([-3.,5.]), source_agent_id =1, target_agent_id =9 )
task         = G(10,15) @ predicate
task_graph.attach(task)


###########################################################################################
#####################      Communication Graph     ########################################
###########################################################################################

comm_graph = CommunicationGraph()
edges      = [(1,4),(1,2),(1,6),(1,9),(9,10),(8,9),(5,4),(2,3),(6,7),(12,5),(13,3),(11,8)]
comm_graph.add_edges_from(edges)


normalize_graphs(task_graph,comm_graph)
show_graphs(comm_graph,task_graph, titles=["Communication Graph","Task Graph",])

new_task_graph, edge_computing_graph = run_task_decomposition(communication_graph               = comm_graph, 
                                                              task_graph                        = task_graph,
                                                              number_of_optimization_iterations = 100, 
                                                              communication_radius              = 7)


centers = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),"centers.npy"))

# Create some initial conditions 
agents_state  = {jj+1: centers[jj] for jj in range(centers.shape[0])}
agents_models = {jj+1: SingleIntegrator2D(max_velocity = 2.5, unique_identifier = jj+1) for jj in range(centers.shape[0])}


show_graphs(comm_graph, new_task_graph, titles=["Communication Graph","Task Graph"])
# storage class to create a multi agent system
system = MultiAgentSystem(task_graph          = new_task_graph,
                          communication_graph = comm_graph,
                          agents_states       = agents_state,
                          agents_models       = agents_models,
                          current_time        = 0)


simulate_agents(system,final_time=15)

ax = plt.gca()
# Display the background image, ensuring it covers the axis limits
background_image = mpimg.imread("/home/gregorio/Desktop/papers/journal/decentralized_STL_task_decomposition/code/assets/thira.png")
ax.imshow(background_image, extent=[ax.get_xlim()[0],ax.get_xlim()[1], ax.get_ylim()[0],ax.get_ylim()[1]], aspect='auto')
plt.show()

