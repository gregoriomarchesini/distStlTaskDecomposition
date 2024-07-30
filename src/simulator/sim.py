"""
MIT License

Copyright (c) [2024] [Gregorio Marchesini]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import matplotlib.pyplot as plt
import numpy as np
from   time import perf_counter
from   tqdm import tqdm
from   traceback import format_exc
from   dataclasses import dataclass
import networkx as nx

from stlddec.graphs     import CommunicationGraph,TaskGraph
from stlcont.controller import STLController, get_logger
from stlcont.utils      import token_passing_algorithm, LeadershipToken
from stl.dynamics       import DynamicalModel,SingleIntegrator2D


plt.rcParams["figure.figsize"] = (3.425, 2.325)

logger = get_logger(__name__)

@dataclass
class MultiAgentSystem:
    task_graph          : TaskGraph
    communication_graph : CommunicationGraph
    agents_states       : dict[int,np.ndarray]
    agents_models       : dict[int,DynamicalModel]
    current_time        : float = 0.
    
    def __post_init__(self):
        
        if  set(self.agents_models.keys()) in set(self.agents_states.keys()) :
            raise ValueError(f"The keys of the agents_states the agents_states must contain the keys of agents_models dictionaries must be the same. Agents in agents states are {list(self.agents_states.keys())} and agents in agents models are {list(self.agents_models.keys())}")
        if set(self.communication_graph.nodes) in set(self.agents_states.keys()) :
            raise ValueError(f"The keys of the agents_states the agents_states must contain the node of communication_graph dictionaries must be the same. Nodes in the commmunicaton graph are {list(self.communication_graph.nodes)} and keys in the agents_states are {list(self.agents_states.keys())}")
        if set(self.task_graph.nodes) in  set(self.agents_states.keys()):
            raise ValueError(f"The keys of the agents_states must contain the node of task_graph. Nodes in the task graph are {list(self.task_graph.nodes)} and keys in the agents_states are {list(self.agents_states.keys())}")
        if set(self.task_graph.nodes) != set(self.communication_graph.nodes):
            raise ValueError(f"The keys of the task_graph and communication_graph dictionaries must be the same. Nodes in the task graph are {list(self.task_graph.nodes)} and nodes in the communication graph are {list(self.communication_graph.nodes)}")
        
        # remove extra keys
        agents_state_copy = self.agents_states.copy()
        models_copy = self.agents_models.copy()
        for key in self.agents_states.keys():
            if key not in self.communication_graph.nodes:
                agents_state_copy.pop(key)
            
        for key in self.agents_models.keys():
            if key not in self.communication_graph.nodes:
                models_copy.pop(key)
        
        self.agents_models = models_copy    
        self.agents_states = agents_state_copy

def simulate_agents(multi_agent_system : MultiAgentSystem, final_time:float, log_level = "INFO", jit : bool = False, look_ahead_time: float = float("inf")) -> None:
    
    print("============================ Starting Simulation =================================")
    state_history   = {unique_identifier : {} for unique_identifier in multi_agent_system.agents_states.keys()}
    time_step       = DynamicalModel._time_step
    number_of_steps = int(final_time/time_step)
    if look_ahead_time == float("inf"):
        look_ahead_time = final_time
    else :
        look_ahead_time = time_step*round(look_ahead_time/time_step)
    switch_times    = np.arange(0.,final_time,look_ahead_time)
    
    
    # Run token passing algorithm.
    tokens_sets = token_passing_algorithm(multi_agent_system.task_graph)
    
    
    # Collect all the tasks for each agent.
    tasks_per_agent = {i:[] for i in multi_agent_system.agents_states.keys()}
    print("Recorganised edges in the task graph")
    for i,j in multi_agent_system.task_graph.edges:
        stl_tasks = multi_agent_system.task_graph.task_list_for_edge(i,j)
        print(f"Edge {(i,j)}")
        
        if i!=j :
            tasks_per_agent[i] += stl_tasks
            tasks_per_agent[j] += stl_tasks
        else :
            tasks_per_agent[i] += stl_tasks
    
    
    number_rounds = round(nx.diameter(multi_agent_system.task_graph)/2)+1

    # Start simulation (sequential).
    with tqdm(total=number_of_steps) as pbar:
        
        for iteration in range(number_of_steps):    
            time = iteration*time_step
            #########################################################
            # Create new controllers at every switch time.
            #########################################################
            if time in switch_times:  
                print("Switching to new time window.")
                controllers : dict[int,STLController]= dict()
                # Set up the controller for each agent.
                for agent_id,model in multi_agent_system.agents_models.items():
                    
                    controllers[agent_id] = STLController(unique_identifier = agent_id, 
                                                        dynamical_model   = model,
                                                        log_level         = log_level,
                                                        look_ahead_time_window=look_ahead_time) 
                    
                
                # exchange the callbacks among agents
                for agent_id, Ti in tokens_sets.items():
                    for agent_j,token in Ti.items() :
                        if token == LeadershipToken.LEADER :
                            controllers[agent_j].register("worse_impact",controllers[agent_id].on_worse_impact)
                        elif token == LeadershipToken.FOLLOWER :
                            controllers[agent_j].register("best_impact",controllers[agent_id].on_best_impact)
                        else :
                            raise ValueError("The token must be either LEADER or FOLLOWER. The presence of a token of type UNDEFINED is a bug. Contact the developers.")
                        
                # Initialize the controllers.
                for agent_id,controller in controllers.items():
                    controller.setup_optimizer(initial_conditions = multi_agent_system.agents_states,
                                               leadership_tokens = tokens_sets[agent_id],
                                               stl_tasks         = tasks_per_agent[agent_id],
                                               initial_time      = multi_agent_system.current_time,
                                               jit = jit)
            
            #########################################################
            # Main control loop.
            #########################################################
            for agent_id in multi_agent_system.agents_models.keys():
                state_history[agent_id][time] = multi_agent_system.agents_states[agent_id].flatten()
            
            try :
                logger.debug(f"New round, Time : {multi_agent_system.current_time}")
                undone_controllers = controllers.copy()
                
                for rounds in range(number_rounds):
                    
                    done_controllers = []
                    # Compute gamma nd notify if you can.
                    for agent_id,controller in undone_controllers.items():
                        if controller.is_ready_to_compute_gamma:
                            controller.compute_gamma()
                            controller.compute_best_impact_for_follower(agents_states=multi_agent_system.agents_states,
                                                                        current_time = multi_agent_system.current_time)
                            controller.compute_worse_impact_for_leaders()
                            done_controllers += [agent_id]
                    
                    # remove the agents which already have computed the value if gamma.
                    for agent_id in done_controllers:
                        undone_controllers.pop(agent_id)
                    
                    for agent_id,controller in undone_controllers.items():
                        controller.compute_gamma_tilde_values(current_time = multi_agent_system.current_time, 
                                                              agents_state = multi_agent_system.agents_states)
                    
            except Exception as e:
                print(format_exc())
                break   
            
            if len(undone_controllers) > 0:
                raise ValueError(f"The controllers did not finish the computation of the control inputs. This is a bug. Contact the developers. Undone agents are {undone_controllers.keys()}")
            
            inputs_per_agent = dict()
            # after this deterministic number of rounds everyone has all the information required to compute its control input
            for agent_id,controller in controllers.items():
                inputs_per_agent[agent_id] = controller.compute_control_input(current_states = multi_agent_system.agents_states, 
                                                                            current_time   = multi_agent_system.current_time)
                
            for agent_id,model in multi_agent_system.agents_models.items():
                multi_agent_system.agents_states[agent_id] = model.step_fun(multi_agent_system.agents_states[agent_id],inputs_per_agent[agent_id]).full()
            multi_agent_system.current_time += time_step
            
            
            pbar.update(1)
            pbar.set_description(f"Time elapsed in simulation: {multi_agent_system.current_time:.2f}/{final_time:.2f}s")
    
    fig, ax = plt.subplots()
    counter = 1
    for agent_id in multi_agent_system.agents_states.keys():
        state = np.vstack(tuple(state_history[agent_id].values()))
        
        if counter == 1 :
            ax.scatter(state[0,0],state[0,1],marker="o",color="green", label="Initial position")
            ax.scatter(state[-1,0],state[-1,1],marker="o",color="red", label = "Final position")
        else :
            ax.scatter(state[0,0],state[0,1],marker="o",color="green")
            ax.scatter(state[-1,0],state[-1,1],marker="o",color="red")
        ax.plot(state[:,0],state[:,1],label=f"Agent {agent_id}")
        
        counter = 0
        
    ax.legend()
    
    return state_history