
import numpy as np
import matplotlib.pyplot as plt
import stlddec.stl_task as pmod
import stlddec.decomposition as dmod
import networkx as nx
import stlddec.simulation as viz


# create edges for decomposition. All edges are undirected
communicatingEdges = [(1,2),(2,4),(1,3),(3,10),(10,5),(1,8),(8,9),(1,6),(6,7)]
brokenEdges        = [(1,9),(1,7),(1,4),(1,5)]

edgeList = []
for (i,j)  in communicatingEdges :
    edgeList += [dmod.GraphEdge(source=i,target=j,isCommunicating=1)]

for (i,j)  in brokenEdges :
    edgeList += [dmod.GraphEdge(source=i,target=j,isCommunicating=0)]


# create an initial state for each node 
initialAgentsState ={ 9:np.array([5,20]),
                      6:np.array([-2,10]),
                      5:np.array([8,-25]),
                      4:np.array([-10,-20]),
                      3:np.array([15,-8]),
                      2:np.array([-3,-3]),
                      8:np.array([14,5]),
                      7:np.array([-8,13]),
                      1:np.array([0,0]),
                      10:np.array([5,-13]),
                      }
# inner ring
# edge 16
predicate  = pmod.regular2DPolygone(numberHyperplanes=5,distanceFromCenter=5,center=np.array([10,10])) 
task       = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a = 10,b = 20),predicate=predicate,source=1,target=6)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,6))
edge.addTasks(task)

# edge 18
predicate  = pmod.regular2DPolygone(numberHyperplanes=5,distanceFromCenter=5,center=np.array([-10,10])) 
task       = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a = 10,b = 20),predicate=predicate,source=1,target=8)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,8))
edge.addTasks(task)

# edge 12
predicate  = pmod.regular2DPolygone(numberHyperplanes=5,distanceFromCenter=5,center=np.array([-10,-10])) 
task       = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a = 10,b = 20),predicate=predicate,source=1,target=2)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,2))
edge.addTasks(task)

# edge 13
predicate  = pmod.regular2DPolygone(numberHyperplanes=5,distanceFromCenter=5,center=np.array([10,-10])) 
task       = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a = 10,b = 20),predicate=predicate,source=1,target=3)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,3))
edge.addTasks(task)


# outer ring
# edge 17
predicate  = pmod.regular2DPolygone(numberHyperplanes=5,distanceFromCenter=5,center=np.array([3,20])) 
task       = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a = 10,b = 20),predicate=predicate,source=1,target=7)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,7))
edge.addTasks(task)

# edge 19
predicate  = pmod.regular2DPolygone(numberHyperplanes=5,distanceFromCenter=5,center=np.array([-3,20])) 
task       = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a = 10,b = 20),predicate=predicate,source=1,target=9)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,9))
edge.addTasks(task)

# edge 14
predicate  = pmod.regular2DPolygone(numberHyperplanes=5,distanceFromCenter=5,center=np.array([-3,-20])) 
task       = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a = 10,b = 20),predicate=predicate,source=1,target=4)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,4))
edge.addTasks(task)

# edge 15
predicate  = pmod.regular2DPolygone(numberHyperplanes=5,distanceFromCenter=5,center=np.array([3,-20])) 
task       = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a = 10,b = 20),predicate=predicate,source=1,target=5)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,5))
edge.addTasks(task)

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

