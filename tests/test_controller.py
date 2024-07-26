import numpy as np
import casadi as ca
from   stl.stl import *
from   stl.dynamics import *
from   stlcont.controller import *
import polytope as pc



model      = SingleIntegrator2D(max_velocity=1.0, unique_identifier=0)
controller = STLController(unique_identifier=0, dynamical_model = model)

a = ca.DM([1])
a = ca.MX.sym("a",3)


agents_state     = {0: np.array([2,3]),1: np.array([1,0]), 2: np.array([0,0])}
leadership_token = {1:LeadershipToken.FOLLOWER, 2:LeadershipToken.FOLLOWER}

# create some random tasks for the agents
# task 01

polytope  = regular_2D_polytope(5,1)

predicate   = CollaborativePredicate(polytope_0=polytope, source_agent_id=0, target_agent_id=1, center = np.array([0,0]))
task01      =  G(10,20) >> predicate

# task 02
predicate  = CollaborativePredicate(polytope_0=polytope, source_agent_id=0, target_agent_id=2, center = np.array([0,0]))
task02     =  G(10,20) >> predicate

stl_tasks  = [task01, task02]

controller.setup_optimizer( initial_conditions= agents_state, 
                            leadership_tokens = leadership_token, 
                            stl_tasks         = stl_tasks,
                            initial_time=0)

print(controller._gamma_tilde)
on_computed    = controller.on_best_impact
on_worse_input = controller.on_worse_impact
on_computed(1,1.3)
on_computed(2,1.4)


controller.compute_gamma_tilde_values(agents_state, 0)
print(controller._gamma_tilde)

# controller.compute_control_input(agents_state, 0)
print(controller._worse_impact_from_follower)
print(controller._best_impact_from_leaders)

print(controller.compute_control_input(agents_state, 0))
pc.Polytope(controller._dynamical_model.input_constraints_A, controller._dynamical_model.input_constraints_b).plot()
ax = plt.gca()
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)

plt.show()