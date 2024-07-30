import numpy as np
import matplotlib.pyplot as plt

from stl.stl import CollaborativePredicate,regular_2D_polytope, G, F
from stlddec.graphs import TaskGraph, CommunicationGraph, normalize_graphs, show_graphs
from simulator.sim   import MultiAgentSystem, simulate_agents
from stl.dynamics     import SingleIntegrator2D

# set seeds for replication
np.random.seed(100)
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
polytope     = regular_2D_polytope(5,2) # creates a regulat polytope cenntered at 0,0 with 5 hyperplanes (penthagon) each of which is 2 units away from the center
predicate    = CollaborativePredicate( polytope_0 = polytope, center=np.array([-5,-5]), source_agent_id=1, target_agent_id=2) # create a collaborative formation task between agents 1 and 2
task         = G(5,10) @ predicate # creates a task by assigning an always operator to the predicate
task_graph.attach(task) # finally attach the task to the task graph (add the required edge to the task graph)
 
# look at the type now!
print(type(task))

# we do the same for edge 2 3
polytope     = regular_2D_polytope(5,2)
predicate    = CollaborativePredicate( polytope_0=  polytope, center=np.array([5,-5]), source_agent_id=3, target_agent_id=2)
task         = G(5,10) @ predicate
task_graph.attach(task)

# a come back task :)
polytope     = regular_2D_polytope(3,2)
predicate    = CollaborativePredicate( polytope_0=  polytope, center=np.array([1,0]), source_agent_id=3, target_agent_id=2)
task         = F(20,30) @ predicate
task_graph.attach(task)

###########################################################################################
######################## Simulate the system                ##############################
###########################################################################################

# Create some initial conditions 
agents_state  = {1: np.array([-1.,0.]), 
                 2: np.array([0.,0.]), 
                 3: np.array([+1.,0.])}

agents_models = {1: SingleIntegrator2D(max_velocity = 3., unique_identifier = 1) , 
                 2: SingleIntegrator2D(max_velocity = 3., unique_identifier = 2) , 
                 3: SingleIntegrator2D(max_velocity = 3., unique_identifier = 3) }

# storage class to create a multi agent system
system = MultiAgentSystem(task_graph          = task_graph,
                          communication_graph = comm_graph,
                          agents_states       = agents_state,
                          agents_models       = agents_models,
                          current_time        = 0)


simulate_agents(system,final_time=30)
plt.show()