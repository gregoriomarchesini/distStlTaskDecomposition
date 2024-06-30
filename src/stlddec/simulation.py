import matplotlib.pyplot as plt
from   stlddec.control import Agent
from   stlddec.decomposition import AgentTaskDecomposition,GraphEdge
from   stlddec.stl_task import StlTask
import numpy as np
from   tqdm import tqdm
from   networkx import Graph
plt.rcParams["figure.figsize"] = (3.425, 2.325)


def simulateAgents(taskGraph :Graph,startTime:float,endTime:float,initialAgentsState:dict[int,np.ndarray],plotInstants:list =[], cleaningTimes:list = []):
    """simulation for the agents trying to satisfy their tasks

    Args:
        taskGraph          (Graph)               : sub-graph of the communication graph. It containes the taks as an attribute "tasks"
        startTime          (float)               : start time of the simulaton (sec)
        endTime            (float)               : end time of the simulation (sec)
        initialAgentsCurrentState (dict[int,np.ndarray]): initial state of the agents
        plotInstants       (list, optional)      : time instants for which a new plot of the agents is given. This is useful to cut the trajectories Defaults to [].
        cleaningTimes      (list, optional)      : times at which the barriers are cleaned by the agents . Defaults to [].

    Returns:
        agentsTrajectory (dict[int,list[np.ndarray]]): trajectories fo the agents during the simulation
    """ 
    
    
    # initialise agents states and trajectories
    agents            :dict[int,Agent]            = {} # Agents obj
    agentsCurrentState       :dict[int,np.ndarray]       = {} # save the current agents states
    agentsTrajectory  :dict[int,list[np.ndarray]] = {}
    
    # check that the agents states are matching the nodes of the task graph
    for id in taskGraph.nodes  :
        if not id in initialAgentsState :
            raise RuntimeError(f"Seems that agent {id} did not have an iniital state assigned. Please provide one")
    
    
    if len(cleaningTimes)==0 :
        cleaningTimes = [1E20]
    
    # initialise intial states and computing agents
    for agentID,initialState in initialAgentsState.items() :
        
        neighbours                  = list(taskGraph.neighbors(agentID)) # after the decposition the communication graph matches the edges of the task graph
        agentsCurrentState[agentID] = initialState
        agentsTrajectory[agentID]   = [initialState]
        agents[agentID]             = Agent(initialState= initialState  ,agentID=agentID,neighbours=neighbours) # computational agent for control
        
    
    # set up the agents for simulation
    for agentId,agent in agents.items() :
        neighboursState = {}
        # initialise the neigbouring agents states
        for index in agents[agentId].neighbours :
            neighboursState[index] = agentsCurrentState[index]
        
        tasks : list[StlTask]= taskGraph.nodes[agentId]["tasks"] # all the tasks to this agent
        agent.initializeController(tasks                   = tasks,
                                   initialNeighboursState  = neighboursState,
                                   initializationTime      = startTime,
                                   allowSlackSatisfaction  = True)
    
    
    # recapping all the task for each agent for logging info
    print("--------------------------------")
    print("Constraint report")
    print("--------------------------------")
    for agentId,agent in agents.items() :
        tasks : list[StlTask]= taskGraph.nodes[agentId]["tasks"] # all the tasks to this agent
        print(f"agentID : {agentId}")
        print(f"number of tasks : {len(tasks)}")

    timeRange               = np.arange(startTime,endTime,agents[1].timeStep) 
    barrierValues           = {key:[] for key in agents.keys()    }
    barrierConstraintValues = {key:[] for key in agents.keys()    }

    maxIt   = len(timeRange)
    plotInstantsIndex = []
    for instant in plotInstants :
        plotInstantsIndex.append(int((instant-startTime)/agents[1].timeStep)) # change the to an index
        if instant <= 0 or instant>=maxIt :
            raise ValueError(f"One or more of the time instansts required for poltting are outside the bounds of the simulation time. Given plotting instants {plotInstants}. Simulation time [{startTime},{endTime}]")
        
    
    closestCleaningTime = cleaningTimes.pop(0) # get the closest cleaning time (time at which tasks are cleaned from the controllers)
    for t in tqdm(timeRange[:-1]) :

        if t >= closestCleaningTime : # reinitialize controller
            print(f"reinitializing controller at time : {t}")
            for agent in agents.values() :
                # here target source is just a name. Doesn't mean that the edge has this direction. The direction of the edge is found later on inside the code
                neighboursState = {}
                for neighbourID in agent.neighbours :
                    neighboursState[neighbourID] = agentsCurrentState[neighbourID]
                
                agent.cleanController(initialNeighboursState = neighboursState,
                                      initializationTime = t,
                                      allowSlackSatisfaction = True) 
                if len(cleaningTimes) :
                    closestCleaningTime = cleaningTimes.pop(0)
                else :
                    closestCleaningTime= 10E20 # just a big number
        
        # sense the agents with possible collsions 
        
        for agentID,agent in agents.items() :
            otherAgentsState = []
            for id,state in agentsCurrentState.items() :
                if id !=agentID :
                    otherAgentsState += [state]
                
            agent.senseCollision(otherAgentsState = otherAgentsState)
       
        for agentID,agent in agents.items() :
 
            agentNextState   = np.squeeze(agent.step(time=t)) # to remove extradimension
            agentsCurrentState[agentID] = agentNextState
            agentsTrajectory[agentID].append(agentNextState)
            barrierValues[agentID]           += [agent.getListOfBarrierValuesAtCurrentTime()]
            barrierConstraintValues[agentID] += [agent.getListOfBarrierConstraintValuesAtCurrentTime()]
        
        if t==timeRange[0] :
            intialNumberOfBarriers = len(barrierValues[agentID])
            
        # now that all the states are updated we can the compute the next control input
        for agent in agents.values() :
            commNeighboursState = {}
            for neigbourID in agent.neighbours :
                commNeighboursState[neigbourID] = agentsCurrentState[neigbourID]
            agent.receiveNeighbourState(commNeighboursState = commNeighboursState)
        

    fig,ax   = plt.subplots()
    fig3,ax3 = plt.subplots()
    
    decimation = 1/100 # between 0 and 1 (percentage of total number of points to plot)
    
    ax.grid(visible=True)
    
    for key,trajectory in agentsTrajectory.items() :
        agentsTrajectory[key] = np.hstack((np.stack(trajectory,axis=0),timeRange[:,np.newaxis]))
        
    
    print("Initial Agents State")
    for agentId,trajectory in agentsTrajectory.items() :
        
        x = trajectory[:,0]
        y = trajectory[:,1]

        step  = int(len(x)*decimation)
        color = np.ones((1,4))
        color[0,:3] = np.random.random(3)
        C = np.repeat(color, len(x[::step ]), axis=0)
        C[:,3] =C[:,3]*np.linspace(0.3,1.,len(x[::step ]))
        
        ax.scatter(x[::step],y[::step],c = C)
        ax.scatter(x[0],y[0],c="red", linewidths=3)
        ax.scatter(x[-1],y[-1],c="green",marker="x", linewidths=5)
        
        for timeInstance,indexTimeInstance in zip(plotInstants,plotInstantsIndex) :
            ax.scatter(x[indexTimeInstance],y[indexTimeInstance],c="blue",marker="x", linewidths=5)
            ax.annotate(xy=(x[indexTimeInstance]+0.2,y[indexTimeInstance]+0.2),text=f"t = {timeInstance}")
            
        
        ax3.plot(x,y)
        ax3.scatter(x[0],y[0],c="red", linewidths=3)
        ax3.scatter(x[-1],y[-1],c="green",marker="x", linewidths=5)
        
        
        
        ax3.annotate(xy=(x[0]+0.6,y[0]+0.6),text=f"agent {agentId}")
        ax.annotate(xy=(x[0]+0.6,y[0]+0.6),text=f"agent {agentId}")
        print(f"Agent ID : {agentId}: {x[0]},{y[0]}")

    
    ax.set_xlabel("x-axis [m]")
    ax.set_ylabel("x-axis [m]")
    ax.set_title("simulated agents trajectories")
    
        
    for agentID in agents.keys() :
        maxRows     = len(barrierValues[agentID][0])
        barriers    = barrierValues[agentID]
        constraints = barrierConstraintValues[agentID]
        for barrier,constraint in zip(barriers,constraints) :
            if len(barrier) < maxRows : # ZERO APDDING
                barrier    += [float(0.),]*(maxRows-len(barrier))
                constraint += [float(0.),]*(maxRows-len(constraint))

    
    fig,ax = plt.subplots(2,len(agents))
    counter     = 0
    for agentID in agents.keys() :     
        barrier     = np.array(barrierValues[agentID])
        constraint  =  np.array(barrierConstraintValues[agentID])
        
        for kk in range(len(barrier[0,:])) :
            ax[0][counter].plot(timeRange[:-1],barrier[:,kk])
            ax[1][counter].plot(timeRange[:-1],constraint[:,kk])
            ax[1][counter].set_xlabel("time [s]")
            ax[0][counter].set_title(f"Agent {agentID}")
            
        counter +=1
    
      
    ax[1][0].set_ylabel("db(x,t)/dt + alpha(b(x,t))")
    ax[0][0].set_ylabel("b(x,t)")
    plt.tight_layout()
    
    
    return agentsTrajectory




########################################################################################################################### 
# Visualization
###########################################################################################################################



def visualizeGraphs(communicationGraph:nx.Graph, initialTaskGraph:nx.Graph, finalTaskGraph:nx.Graph) :
    
    nodes = communicationGraph.nodes(data=True)
    xx = [node[1]["pos"][0] for node in nodes]
    yy = [node[1]["pos"][1] for node in nodes]
    xxmin,xxmax= min(xx)*1.6,max(xx)*1.6
    yymin,yymax = min(yy)*1.6,max(yy)*1.6

    nodes = communicationGraph.nodes(data=True)
    
    # define figure object
    fig, ax = plt.subplots() 
    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])


    # Define graphs
    fig, ax = plt.subplots(1,3) 

    # Communicatino Graph
    ax[0].set_xlim([xxmin,xxmax])
    ax[0].set_ylim([yymin,yymax])
    # # Drawing of the network
    edgeLabels = { (i,j):"link" for  i,j in communicationGraph.edges}    
    nx.draw_networkx(communicationGraph,{node:nodeDict["pos"] for node,nodeDict in nodes},ax=ax[0])

    nx.draw_networkx_edge_labels(
        communicationGraph,
        {node:nodeDict["pos"] for node,nodeDict in nodes},
        edge_labels = edgeLabels,
        font_color='black',
        ax=ax[0]
    )

    ax[0].set_title("communication graph")


    # final Task Graph
    ax[1].set_xlim([-20,20])
    ax[1].set_ylim([-20,20])
    # # Drawing of the network
    edgeLabels = { (i,j):"Task" for  i,j,attr in finalTaskGraph.edges(data=True) if attr["edgeObj"].hasSpecifications}    
    taskPlot = nx.draw_networkx(finalTaskGraph,{node:nodeDict["pos"] for node,nodeDict in nodes},ax=ax[1])

    nx.draw_networkx_edge_labels(
        finalTaskGraph,
        {node:nodeDict["pos"] for node,nodeDict in nodes},
        edge_labels = edgeLabels,
        font_color='black',
        ax=ax[1]
    )

    ax[1].set_title("final Task Graph")
    
    ax[1].set_xlim([xxmin,xxmax])
    ax[1].set_ylim([yymin,yymax])
    
    
    # Initial Task Graph
    ax[2].set_xlim([-20,20])
    ax[2].set_ylim([-20,20])
    # # Drawing of the network
    edgeLabels = { (i,j):"Task" for  i,j,attr in initialTaskGraph.edges(data=True) if attr["edgeObj"].hasSpecifications}    
    taskPlot = nx.draw_networkx(initialTaskGraph,{node:nodeDict["pos"] for node,nodeDict in nodes},ax=ax[2])

    nx.draw_networkx_edge_labels(
        initialTaskGraph,
        {node:nodeDict["pos"] for node,nodeDict in nodes},
        edge_labels = edgeLabels,
        font_color='black',
        ax=ax[2]
    )

    ax[2].set_title("Initial Task Graph")
    ax[2].set_xlim([xxmin,xxmax])
    ax[2].set_ylim([yymin,yymax])
