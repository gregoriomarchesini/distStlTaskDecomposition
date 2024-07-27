import numpy as np
from  matplotlib import pyplot as plt
import os, sys
import logging
from stl.stl               import CollaborativePredicate,regular_2D_polytope, G, F
from stlddec.graphs        import TaskGraph, CommunicationGraph, normalize_graphs, show_graphs
from simulator.sim        import MultiAgentSystem, simulate_agents
from stl.dynamics          import SingleIntegrator2D
from stlddec.decomposition import run_task_decomposition

# set seeds for replication
np.random.seed(100)

###########################################################################################
########################      Task Graph     ##############################################
###########################################################################################


task_graph = TaskGraph()

# outer ring
# edge 1-5
polytope     = regular_2D_polytope(5,2)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([5,5]), source_agent_id =1, target_agent_id =5 )
task         = F(10,15) @ predicate
task_graph.attach(task)

# edge 1-3
polytope     = regular_2D_polytope(5,2)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([5,-5]), source_agent_id =1, target_agent_id =3 )
task         = F(10,15) @ predicate
task_graph.attach(task)

# edge 1-10
polytope     = regular_2D_polytope(5,2)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([-5,5]), source_agent_id =1, target_agent_id =10 )
task         = F(10,15) @ predicate
task_graph.attach(task)

# edge 1-8
polytope     = regular_2D_polytope(5,2)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([-5,0]), source_agent_id =1, target_agent_id =8 )
task         = F(10,15) @ predicate
task_graph.attach(task)

# edge 1-6
polytope     = regular_2D_polytope(5,2)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([-5,-5]), source_agent_id =1, target_agent_id =7 )
task         = F(10,15) @ predicate
task_graph.attach(task)

# inner ring
# edge 1-4
polytope     = regular_2D_polytope(3,1.5)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([1,2]), source_agent_id =1, target_agent_id =4 )
task         = G(10,15) @ predicate
task_graph.attach(task)

# edge 1-2
polytope     = regular_2D_polytope(3,1.5)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([1,-2]), source_agent_id =1, target_agent_id =2 )
task         = G(10,15) @ predicate
task_graph.attach(task)

# edge 1-6
polytope     = regular_2D_polytope(3,1.5)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([-1,-2]), source_agent_id =1, target_agent_id =6 )
task         = G(10,15) @ predicate
task_graph.attach(task)

# edge 1-9
polytope     = regular_2D_polytope(3,1.5)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([-1,2]), source_agent_id =1, target_agent_id =9 )
task         = G(10,15) @ predicate
task_graph.attach(task)


###########################################################################################
#####################      Communication Graph     ########################################
###########################################################################################

comm_graph = CommunicationGraph()
edges      = [(1,4),(1,2),(1,6),(1,9),(9,10),(8,9),(5,4),(2,3),(6,7)]
comm_graph.add_edges_from(edges)


normalize_graphs(task_graph,comm_graph)
show_graphs(comm_graph,task_graph, titles=["Communication Graph","Task Graph",])

new_task_graph, edge_computing_graph = run_task_decomposition(communication_graph               = comm_graph, 
                                                              task_graph                        = task_graph,
                                                              number_of_optimization_iterations = 10, 
                                                              communication_radius              = 4)


centers = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),"centers.npy"))

# Create some initial conditions 
agents_state  = {jj+1: centers[jj] for jj in range(centers.shape[0])}
agents_models = {jj+1: SingleIntegrator2D(max_velocity = 1., unique_identifier = jj+1) for jj in range(centers.shape[0])}


show_graphs(comm_graph, new_task_graph, titles=["Communication Graph","Task Graph"])
# storage class to create a multi agent system
system = MultiAgentSystem(task_graph          = new_task_graph,
                          communication_graph = comm_graph,
                          agents_states       = agents_state,
                          agents_models       = agents_models,
                          current_time        = 0)


simulate_agents(system,final_time=30)


plt.show()
