
import numpy as np
import matplotlib.pyplot as plt
import stlddec.predicate_builder_module as pmod
import stlddec.decomposition_module as dmod
import networkx as nx
import stlddec.visualization_module as viz


# create edges for decomposition. All edges are undirected
communicatingEdges = [(1,2),(2,3),(3,5),(3,4),(1,6),(6,7),(7,8),(9,7)]
brokenEdges        = [(1,9),(1,8),(1,4),(1,5),(1,7),(1,3),(1,1)]

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
                      3:np.array([15,8]),
                      2:np.array([5,5]),
                      8:np.array([14,5]),
                      7:np.array([-8,13]),
                      1:np.array([0,0]),
                      }

predicate  = pmod.regular2DPolygone(numberHyperplanes = 5,distanceFromCenter = 5,center=np.array([-20,5])) 
task       = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a = 10,b = 20),predicate=predicate,source=1,target=9)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,9))
edge.addTasks(task)

predicate  = pmod.regular2DPolygone(numberHyperplanes = 5,distanceFromCenter = 5,center=np.array([20,5])) 
task       = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a = 10,b = 20),predicate=predicate,source=1,target=5)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,5))
edge.addTasks(task)

predicate  = pmod.regular2DPolygone(numberHyperplanes=5,distanceFromCenter=2,center=np.array([-7,12])) 
task       = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a = 10,b = 20),predicate=predicate,source=1,target=6)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,6))
edge.addTasks(task)

predicate  = pmod.regular2DPolygone(numberHyperplanes=5,distanceFromCenter=2,center=np.array([7,12])) 
task       = pmod.StlTask(temporalOperator="always",timeinterval=pmod.TimeInterval( a = 10,b = 20),predicate=predicate,source=1,target=2)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,2))
edge.addTasks(task)

predicate  = pmod.regular2DPolygone(numberHyperplanes=5,distanceFromCenter=2,center=np.array([-7,20])) 
task       = pmod.StlTask(temporalOperator="eventually",timeinterval=pmod.TimeInterval( a = 19,b = 23),predicate=predicate,source=1,target=8,timeOfSatisfaction=22)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,8))
edge.addTasks(task)

predicate  = pmod.regular2DPolygone(numberHyperplanes=5,distanceFromCenter=2,center=np.array([7,28])) 
task       = pmod.StlTask(temporalOperator="eventually",timeinterval=pmod.TimeInterval( a = 19,b = 23),predicate=predicate,source=1,target=4,timeOfSatisfaction=22)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,4))
edge.addTasks(task)


predicate  = pmod.regular2DPolygone(numberHyperplanes=5,distanceFromCenter=2,center=np.array([-7,20])) 
task       = pmod.StlTask(temporalOperator="eventually",timeinterval=pmod.TimeInterval( a = 30,b = 35),predicate=predicate,source=1,target=8,timeOfSatisfaction=33)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,8))
edge.addTasks(task)

predicate  = pmod.regular2DPolygone(numberHyperplanes=5,distanceFromCenter=2,center=np.array([7,20])) 
task       = pmod.StlTask(temporalOperator="eventually",timeinterval=pmod.TimeInterval( a = 30,b = 35),predicate=predicate,source=1,target=4,timeOfSatisfaction=33)
edge       = dmod.findEdge(edgeList=edgeList,edge= (1,4))
edge.addTasks(task)



commGraph,finalTaskGraph,originalTaskGraph = dmod.runTaskDecomposition(edgeList=edgeList)



# viz.simulateAgents(finalTaskGraph,endTime=40,startTime = 0,initialAgentsState=initialAgentsState,cleaningTimes=[24])



fig,ax = plt.subplots(3)
nx.draw_networkx(commGraph,ax=ax[0])
ax[0].set_title("Communication Graph")
nx.draw_networkx(finalTaskGraph,ax=ax[1])
ax[1].set_title("Final Task Graph")
nx.draw_networkx(originalTaskGraph,ax=ax[2])
ax[2].set_title("Original Task Graph")


plt.show()

