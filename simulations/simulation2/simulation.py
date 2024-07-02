
import numpy as np
import matplotlib.pyplot as plt
import stlddec.stl_task as pmod
import stlddec.graphs as gmod
import stlddec.decomposition as dmod
import polytope as pc



# List all the edges in the network with communication
edges_in_the_network = [(1,2),(2,3),(3,1)]      
G = gmod.create_graph_from_edges(edges_in_the_network)
print(G.edges.values())
for edge in G.edges :
    
    # Create aa regular polytope.
    polytope     = pmod.regular_2D_polytope(5,4)
    
    # Create a predicate!
    predicate   = pmod.CollaborativePredicate( polytope_0=  polytope,
                                              center=np.array([0,0]),
                                              source_agent_id=edge[0],
                                              target_agent_id=edge[1])
    # Set a temporal operator.
    t_operator  = pmod.AlwaysOperator(pmod.TimeInterval(0,10))
    
    # Create a task.
    task        = pmod.StlTask(temporal_operator=t_operator,predicate=predicate)
    
    # Add the task to the edge.
    G[edge[0]][edge[1]][gmod.MANAGER].add_tasks(task)
    

# Now break some of the edges.
G  = gmod.break_communication_edge(G,[(1,3)])

dmod.run_task_decomposition(complete_graph=G)



# viz.simulateAgents(finalTaskGraph,endTime=20,startTime = 0,initialAgentsState=initialAgentsState)



# fig,ax = plt.subplots(3)
# nx.draw_networkx(commGraph,ax=ax[0])
# ax[0].set_title("Communication Graph")
# nx.draw_networkx(finalTaskGraph,ax=ax[1])
# ax[1].set_title("Final Task Graph")
# nx.draw_networkx(originalTaskGraph,ax=ax[2])
# ax[2].set_title("Original Task Graph")


# plt.show()

