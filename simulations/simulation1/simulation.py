import numpy as np
import matplotlib.pyplot as plt
import stlddec.stl_task as pmod
import stlddec.decomposition as dmod
import networkx as nx
import stlddec.simulation as viz


# create some edges
communicating_edges = [(1,2),(2,4),(1,3),(3,10),(10,5),(1,8),(8,9),(1,6),(6,7)]
broken_edges        = [(1,9),(1,7),(1,4),(1,5)]


# create sinmple path graph
edgeList = []
for jj in range(4) :
    edgeList += [dmod.GraphEdge(source=jj,target=jj+1,isCommunicating=1)]

initialAgentsState ={ 0:np.array([0,0]),
                      1:np.array([-10,11]),
                      2:np.array([-25,-30]),
                      3:np.array([0,-20]),
                      4:np.array([10,-20])}




predicate = pmod.regular2DPolygone(numberHyperplanes=4,distanceFromCenter=4,center=np.array([10,10])) 
task04i    = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a= 10,b = 20),predicate=predicate,source=0,target=4)

predicate = pmod.regular2DPolygone(numberHyperplanes=4,distanceFromCenter=5,center=np.array([10,8]))
task04j    = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a= 10,b = 20),predicate=predicate,source=0,target=4)

predicate = pmod.regular2DPolygone(numberHyperplanes=4,distanceFromCenter=2,center=np.array([10,4]))
task01 = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a= 10,b = 20),predicate=predicate,source=0,target=1)



edge01 :dmod.GraphEdge = edgeList[0]
edge01.addTasks(tasks=task01)


edge14    = dmod.GraphEdge(source=0,target=4,isCommunicating=0)
edge14.addTasks(tasks=[task04i,task04j])
edgeList.append(edge14)


commGraph,finalTaskGraph,originalTaskGraph = dmod.runTaskDecomposition(edgeList=edgeList)
viz.simulateAgents(finalTaskGraph,endTime=20,startTime = 0,initialAgentsState=initialAgentsState)



fig,ax = plt.subplots(3)
nx.draw_networkx(commGraph,ax=ax[0])
ax[0].set_title("Communication Graph")
nx.draw_networkx(finalTaskGraph,ax=ax[1])
ax[1].set_title("Final Task Graph")
nx.draw_networkx(originalTaskGraph,ax=ax[2])
ax[2].set_title("Original Task Graph")
plt.show()

