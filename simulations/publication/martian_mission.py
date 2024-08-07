import numpy as np
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import logging
import pickle

from stl.stl               import CollaborativePredicate,IndependentPredicate , regular_2D_polytope, G,F
from stlddec.graphs        import TaskGraph, CommunicationGraph, normalize_graphs, show_graphs, visualize_tasks_in_the_graph
from simulator.sim        import MultiAgentSystem, simulate_agents
from stl.dynamics          import SingleIntegrator2D
from stlddec.decomposition import run_task_decomposition,DecompositionParameters

JIT                  = False  # if you want to just in time compile the code.
USE_SAVED_TASK_GRAPH = False  # if you want to use the already decomposed graph from a previous simulation.
SAVE_SIMULATION      = False # if you want to save the results of the simulation.

# set seeds for replication
np.random.seed(100)
meters_to_km = 1e-3
km_to_meters = 1e3


## Communication Graph.     
comm_graph = CommunicationGraph()
edges      = [(1,2),(2,3),(3,4),(2,5),(1,6),(6,7),(7,8),(6,9),(9,10),(1,11),(11,12),(12,13),(14,10),(10,15)]
comm_graph.add_edges_from(edges)

## Task Graph .    
# units are all in Km.
task_graph = TaskGraph()

########################################################################################################
########################   Part 1 : Far from Home ######################################################
########################################################################################################

## Independent task : Agents 1 moves toward the crater.
polytope     = regular_2D_polytope(4, 400* meters_to_km    )
predicate    = IndependentPredicate( polytope_0 = polytope, center = np.array([5.,0.]), agent_id =1 )
task         = G(10,20) @ predicate
task_graph.attach(task)

## Two robots to the sides of the crater: agent 8 on the upper side and robot 4 on the lower side.
polytope     = regular_2D_polytope(6, 400* meters_to_km    )
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([10.,4.]),source_agent_id =1, target_agent_id =8 )
task         = F(10,15) @ predicate
task_graph.attach(task)

polytope     = regular_2D_polytope(6, 400* meters_to_km    )
predicate    = CollaborativePredicate( polytope_0 = polytope,  center = np.array([10.,-4.]), source_agent_id =1, target_agent_id =4 )
task         = F(10,15) @ predicate
task_graph.attach(task)


## Two robots the upper crater and they should stay together all time
polytope     = regular_2D_polytope(4, 700* meters_to_km    )
predicate    = CollaborativePredicate( polytope_0 = polytope,  center = np.array([-12.,6.0]), source_agent_id =1, target_agent_id =14 )
task         = G(13,15) @ predicate
task_graph.attach(task)



## one robots the first lower crater
polytope     = regular_2D_polytope(5, 400* meters_to_km    )
predicate    = CollaborativePredicate( polytope_0 = polytope,  center = np.array([-6.0,-6.0]), source_agent_id =1, target_agent_id =13 )
task         = F(10,15) @ predicate
task_graph.attach(task)


## one robots the second lower crater
polytope     = regular_2D_polytope(5, 400* meters_to_km    )
predicate    = CollaborativePredicate( polytope_0 = polytope,  center = np.array([-3.5,-4.]), source_agent_id =1, target_agent_id =5 )
task         = F(13,15) @ predicate
task_graph.attach(task)

## Triangualar formation in the center

polytope     = regular_2D_polytope(5, 400* meters_to_km    )
predicate    = CollaborativePredicate( polytope_0 = polytope,  center = np.array([0.,2.]), source_agent_id =1, target_agent_id =6 )
task         = G(10,15) @ predicate
task_graph.attach(task)


polytope     = regular_2D_polytope(5, 400* meters_to_km    )
predicate    = CollaborativePredicate( polytope_0 = polytope,  center = np.array([0.,-2.]), source_agent_id =1, target_agent_id =2 )
task         = G(10,15) @ predicate
task_graph.attach(task)

polytope     = regular_2D_polytope(5, 400* meters_to_km    )
predicate    = CollaborativePredicate( polytope_0 = polytope,  center = np.array([-4.,0]), source_agent_id =1, target_agent_id =11 )
task         = G(10,15) @ predicate
task_graph.attach(task)

## Triangular formation agents 14-10-15


polytope     = regular_2D_polytope(5, 300* meters_to_km    )
predicate    = CollaborativePredicate( polytope_0 = polytope,  center = np.array([0.,0.]), source_agent_id =10, target_agent_id =15 )
task         = G(18,20) @ predicate
task_graph.attach(task)


########################################################################################################
########################   Part 2 : The come back ######################################################
########################################################################################################


# regather with 1
for ii in range(2,16):
    polytope     = regular_2D_polytope(8, 2000* meters_to_km )
    predicate    = CollaborativePredicate( polytope_0 = polytope,  center = np.array([0.,0.]), source_agent_id =1, target_agent_id =ii )
    task         = G(38,40) @ predicate
    task_graph.attach(task)
    
# while 1 goes back to zero
polytope     = regular_2D_polytope(8, 300* meters_to_km    )
predicate    = IndependentPredicate( polytope_0 = polytope, center = np.array([0.,0.]), agent_id =1 )
task         = G(30,40) @ predicate
task_graph.attach(task)


########################################################################################################
########################   Task Decomposition and Simulation ###########################################
########################################################################################################

# normalize and visualize.
normalize_graphs(task_graph,comm_graph)
show_graphs(comm_graph,task_graph, titles=["Communication Graph","Task Graph",])

# Run task decomposition
if  USE_SAVED_TASK_GRAPH :
    with open(os.path.join( os.path.dirname(__file__),'new_task_graph.gpickle'), 'rb') as f:
        new_task_graph = pickle.load(f)
else :
    # DO NOT CHANGE THESE PARAMETERS FOR THIS EXAMPLE.
    parameters = DecompositionParameters(learning_rate_0     = 0.3,  # higher value : increases learning speed bu gives rise to jittering
                                         decay_rate          = 0.7,  # higher value : learning rate decays faster
                                         penalty_coefficient  = 100,  # higher value : can speed up convergence but too high values will enhance jittering
                                         communication_radius  = 8.5, # practically infinity
                                         number_of_optimization_iterations = 3500)
    
    
    
    
    
    new_task_graph, edge_computing_graph, agents_costs_and_penalties,decomposition_accuracy_per_task  = run_task_decomposition(communication_graph  = comm_graph, 
                                                                                                                               task_graph           = task_graph,
                                                                                                                               parameters           = parameters,
                                                                                                                               jit                  = JIT)

    with open(os.path.join( os.path.dirname(__file__),'new_task_graph.gpickle'), 'wb') as f:
        pickle.dump(new_task_graph, f, pickle.HIGHEST_PROTOCOL)
        
        
initial_conditions = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),"initial_states.npy"))
# Create some initial conditions 
agents_state  = {jj+1: initial_conditions[jj] for jj in range(initial_conditions.shape[0])}
agents_models = {jj+1: SingleIntegrator2D(max_velocity = 1.8, unique_identifier = jj+1) for jj in range(initial_conditions.shape[0])}


show_graphs(comm_graph, new_task_graph, titles=["Communication Graph","Task Graph"])
# storage class to create a multi agent system
system = MultiAgentSystem(task_graph          = new_task_graph,
                          communication_graph = comm_graph,
                          agents_states       = agents_state,
                          agents_models       = agents_models,
                          current_time        = 0)

visualize_tasks_in_the_graph(new_task_graph)

# save state trajectories.
state_history = simulate_agents(system,final_time=40., jit=JIT, look_ahead_time= 20.)

ax = plt.gca()
# Display the background image, ensuring it covers the axis limits
background_image = mpimg.imread("/home/gregorio/Desktop/papers/journal/decentralized_STL_task_decomposition/code/assets/thira.png")
ax.imshow(background_image, extent=[-25,25, -7.5,7.5], aspect='auto')
ax.set_xlabel("Km")
ax.set_ylabel("Km")
ax.set_xlim(-11.5,24.4)

if SAVE_SIMULATION :
    for unique_identifier in state_history.keys():
        time  = np.array(list(state_history[unique_identifier].keys()))[:,np.newaxis]
        state = np.vstack(list(state_history[unique_identifier].values()))
        
        save_array = np.hstack((time,state))
        os.makedirs(os.path.join( os.path.dirname(__file__),"states"),exist_ok=True)
        np.save(os.path.join( os.path.dirname(__file__),"states",f"state_history_{unique_identifier}.npy"),save_array)

    # save decomposition results.
    os.makedirs(os.path.join( os.path.dirname(__file__),"decomposition_result"),exist_ok=True)
    
    if not USE_SAVED_TASK_GRAPH :
        np.save(os.path.join( os.path.dirname(__file__),"decomposition_result","cost_and_penalties.npy"),agents_costs_and_penalties,allow_pickle=True)
        np.save(os.path.join( os.path.dirname(__file__),"decomposition_result","decomposition_accuracy_per_task.npy"),decomposition_accuracy_per_task,allow_pickle=True)
        

plt.show()