import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import logging 

import stl.stl        as pmod
import stlddec.graphs as gmod
import stl.dynamics   as dyn

from simulator.sim import MultiAgentSystem, simulate_agents 


np.random.seed(100)
random.seed(100)

# List all the edges in the network with communication.
task_edges = [(1,2),(2,3)]
comm_edges = [(1,2),(2,3)]     

task_graph = gmod.create_task_graph_from_edges(task_edges)
comm_graph = gmod.create_communication_graph_from_edges( comm_edges)

# EDGE 12
# Create aa regular polytope.
polytope     = pmod.regular_2D_polytope(5,2)
predicate    = pmod.CollaborativePredicate( polytope_0=  polytope,
                                            center=np.array([-5,-5]),
                                            source_agent_id=1,
                                            target_agent_id=2)
task  = pmod.G(5,10) @ predicate
task_graph.attach(task)
    

# EDGE 32
# Create aa regular polytope.
polytope     = pmod.regular_2D_polytope(5,2)
predicate    = pmod.CollaborativePredicate( polytope_0=  polytope,
                                           center=np.array([5,-5]),
                                           source_agent_id=3,
                                           target_agent_id=2)
task  = pmod.G(5,10) @ predicate
task_graph.attach(task)



# EDGE 32
# Create aa regular polytope.
polytope     = pmod.regular_2D_polytope(3,2)
predicate    = pmod.CollaborativePredicate( polytope_0=  polytope,
                                           center=np.array([1,0]),
                                           source_agent_id=3,
                                           target_agent_id=2)
task  = pmod.F(20,30) @ predicate
task_graph.attach(task)


agents_state  = {1: np.array([-1.,0.]), 2: np.array([0.,0.]), 3: np.array([+1.,0.])}
agents_models = {1: dyn.SingleIntegrator2D(max_velocity = 3., unique_identifier = 1) , 
                 2: dyn.SingleIntegrator2D(max_velocity = 3., unique_identifier = 2) , 
                 3: dyn.SingleIntegrator2D(max_velocity = 3., unique_identifier = 3) }

system = MultiAgentSystem(task_graph          = task_graph,
                          communication_graph = comm_graph,
                          agents_states       = agents_state,
                          agents_models       = agents_models,
                          current_time        = 0)


simulate_agents(system, 30)
plt.show()