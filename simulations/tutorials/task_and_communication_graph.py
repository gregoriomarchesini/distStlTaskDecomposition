import numpy as np

import stl.stl        as pmod
from   stlddec.graphs import TaskGraph, CommunicationGraph, normalize_graphs, show_graphs

# set seeds for replication
np.random.seed(100)


###########################################################################################
######################## Create Communication and Task Graph ##############################
###########################################################################################

# We will try to create a matching communication and task graph
task_edges = [(1,2),(2,3)]
comm_edges = [(1,2),(2,3)]     

task_graph = TaskGraph()
comm_graph = CommunicationGraph()

# We first fill the communication graph with the given edges
comm_graph.add_edges_from(comm_edges)

# To create the task graph we can first create some regular task
polytope     = pmod.regular_2D_polytope(5,2) # creates a regulat polytope cenntered at 0,0 with 5 hyperplanes (penthagon) each of which is 2 units away from the center
predicate    = pmod.CollaborativePredicate( polytope_0 = polytope, center=np.array([-5,-5]), source_agent_id=1, target_agent_id=2) # create a collaborative formation task between agents 1 and 2
task         = pmod.G(5,10) @ predicate # creates a task by assigning an always operator to the predicate
task_graph.attach(task) # finally attach the task to the task graph (add the required edge to the task graph)
 
# look at the type now!
print(type(task))

# we do the same for edge 2 3
polytope     = pmod.regular_2D_polytope(5,2)
predicate    = pmod.CollaborativePredicate( polytope_0=  polytope, center=np.array([5,-5]), source_agent_id=3, target_agent_id=2)
task         = pmod.G(5,10) @ predicate
task_graph.attach(task)

# a come back task :)
polytope     = pmod.regular_2D_polytope(3,2)
predicate    = pmod.CollaborativePredicate( polytope_0=  polytope, center=np.array([1,0]), source_agent_id=3, target_agent_id=2)
task         = pmod.F(20,30) @ predicate
task_graph.attach(task)


# Now that you have create the task and communication graph you can visualize them 
# it is always better to normalize the nodes of the graphs (make sure that both graphs have the same nodes)
normalize_graphs(task_graph,comm_graph)
show_graphs(task_graph,comm_graph, titles=["Task Graph","Communication Graph"])

# nx.draw(comm_graph,with_labels=True,ax= axs[0],pos=pos_comm)
# nx.draw(task_graph,with_labels=True,ax= axs[1],pos=pos_task)

# agents_state  = {1: np.array([-1.,0.]), 2: np.array([0.,0.]), 3: np.array([+1.,0.])}
# agents_models = {1: dyn.SingleIntegrator2D(max_velocity = 3., unique_identifier = 1) , 
#                  2: dyn.SingleIntegrator2D(max_velocity = 3., unique_identifier = 2) , 
#                  3: dyn.SingleIntegrator2D(max_velocity = 3., unique_identifier = 3) }

# system = MultiAgentSystem(task_graph          = task_graph,
#                           communication_graph = comm_graph,
#                           agents_states       = agents_state,
#                           agents_models       = agents_models,
#                           current_time        = 0)


# # simulate_agents(system, 30)
# # plt.show()