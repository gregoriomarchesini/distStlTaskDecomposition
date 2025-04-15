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

# Dynamical models for multi-agent simulation

import casadi as ca
import numpy as np
from abc import ABC, abstractmethod
from itertools import product
from typing import TypeAlias

UniqueIdentifier: TypeAlias = int

# The enumerators are used to define some input and output names that can be useful 
# for describing a dynamical system. Feel free to add more names if you need them


FREQUENCY = 100 # hz

def is_casadiMX(x):
    return isinstance(x, ca.MX)

def name_without_id(string):
    splitted_list = string.split('_')
    name = '_'.join(splitted_list[:-1])
    return name

def wrap_name(name:str,unique_identifier:UniqueIdentifier):
    return name + "_" + str(unique_identifier)

def get_id_from_name(name:str)-> UniqueIdentifier|None:
    splitted_name = name.split('_')
    if not ("state" in splitted_name):
        return None
    else :
        return int(splitted_name[1]) # index of the 



    
class DynamicalModel(ABC):
    """
                                                                                  .
    DynamicalModel - Abstract class to define a control affine dynamical system in the form   x = f(x) + g(x)u
    
    """
    
    _time_step = 1/FREQUENCY # time step common to all the agents,
    
   
    def __init__(self,unique_identifier) -> None:
        self._unique_identifier = unique_identifier # unique_identifier of the agent that the dynamical system refers to
        
    
    @property
    def unique_identifier(self)->UniqueIdentifier:
        return self._unique_identifier

    @property
    @abstractmethod
    def g(self) -> ca.MX:
        """ 
        Casadi expression that represents the g(x) matrix in the dynamical system x = f(x) + g(x)u
        
        Follow this example for a simple differential drive robot (to be define in the init method of the dynamical model
        Visit the subsequent template classes for more examples
        
        Example :
        >>> self._heading  = ca.MX.sym(wrap_name(StateName.HEADING,self._unique_identifier))
        >>> self._velocity = ca.MX.sym(wrap_name(InputName.VELOCITY1D,self._unique_identifier))

        >>> # create dynamics model
        >>> row1 = ca.horzcat(ca.cos(self._heading), -self._look_ahead_distance*ca.sin(self._heading))
        >>> row2 = ca.horzcat(ca.sin(self._heading), self._look_ahead_distance*ca.cos(self._heading))
        >>> row3 = ca.horzcat(0,1)
        >>> self._g = ca.vertcat(row1,row2,row3) # g(x) matrix
        >>> self._f = ca.vertcat(0,0,0)  # f(x) vector
        
        
        """
        pass
    

    @property
    @abstractmethod
    def f(self) -> ca.MX:
        """
        Casaadi expression that represents the f(x) vector in the dynamical system x = f(x) + g(x)u
        Follow this example for a simple differential drive robot (to be define in the init method of the dynamical model
        Visit the subsequent template classes for more examples
        
        Example :
        >>> self._heading  = ca.MX.sym(wrap_name(StateName.HEADING,self._unique_identifier))
        >>> self._velocity = ca.MX.sym(wrap_name(InputName.VELOCITY1D,self._unique_identifier))

        >>> # create dynamics model
        >>> row1 = ca.horzcat(ca.cos(self._heading), -self._look_ahead_distance*ca.sin(self._heading))
        >>> row2 = ca.horzcat(ca.sin(self._heading), self._look_ahead_distance*ca.cos(self._heading))
        >>> row3 = ca.horzcat(0,1)
        >>> self._g = ca.vertcat(row1,row2,row3) # g(x) matrix
        >>> self._f = ca.vertcat(0,0,0)  # f(x) vector
        
        """
        pass
    
    # define here the input vector and the state vector
    @property
    @abstractmethod
    def input_vector(self) -> ca.MX:
        """
        Define an input vector for your system. For example you could have the following input vector for a car
        >>> self._velocity = ca.MX.sym(wrap_name(InputName.VELOCITY1D,self._unique_identifier)) # longitudinal velocity
        >>> self._turning_rate = ca.MX.sym(wrap_name(InputName.TURNING_RATE,self._unique_identifier)) # tunring rate
        >>> self._input_vector = ca.vertcat(self._velocity,self._turning_rate)
        >>> self._state_vector = ca.MX.sym("state_vector",0,0)
        """
        
        return  ca.MX.sym("input_vector",0,0)

    @property
    @abstractmethod
    def state_vector(self) -> ca.MX:
        """
        Define a state vector for your system. For example you could have the following state vector for a car
        >>> self._position = ca.MX.sym(wrap_name(StateName.POSITION2D,self._unique_identifier), 2) # create a casadi variable with the name "position_2d_id" where unique_identifier of the agent (this is useful for debuggin so that states from different dynamical systems have different names)
        >>> self._heading  = ca.MX.sym(wrap_name(StateName.HEADING,self._unique_identifier))       # heading state
        >>> self._state_vector = ca.vertcat(self._position,self._heading)
        """
        return ca.MX.sym("state_vector",0,0)
    
    
    @property
    @abstractmethod
    def input_constraints_A(self)->ca.MX:
        """
        Matrix of the input constraints A@input_vector<=b
        
        Note : Nonlinear constraints should be approximated to be linear. A Polygon with sufficiently many faces is sufficient to represent closed convex sets.
        Support function to construct such polygons are provided
        """
        pass
    
    
    @property
    @abstractmethod
    def input_constraints_b(self)->ca.MX:
        """
        Vector of the input constraints A@input_vector<=b
        
        Note : Nonlinear constraints should be approximated to be linear. A Polygon with sufficiently many faces is sufficient to represent closed convex sets.
        Support function to construct such polygons are provided
        """
        
        pass
    
    @property
    @abstractmethod
    def maximum_expressible_speed(self)->float:
        """
        Heuristic of the maximum value of Lf + Lg@u
        
        This is applied to make some heuristic on the construction of LinearBarrier functions for example. It is important to consider this value carefully.
        For a single integrator, thenorm of the maximum allowed speed is the straight answer, but for more complex systems this is not the case.
        It is better to give a bit more conservative value rather than a less conservative one.
        """
        pass
    
    @property
    def step_fun(self)->ca.Function:
        return self._get_step_function(rk4Steps=1)
    
    
    @property
    def dynamics_exp(self) -> ca.MX:
        """casadi expression for the dynamical model of the system (input affine)"""
        return self.f + self.g @ self.input_vector


    @property
    def dynamics_fun(self)->ca.Function:
        """ 
        Return a casadi function that can be used to evaluate the dynamics for a given state and input. Note that this is different from the step funcition that is the discrete version of the continuous dynamics.
        
        Returns:
            The dynamics function expressed as a casadi.Function
        
        Example :
        >>> dynamics_fun = self.dynamics_fun
        >>> dynamics = dynamics_fun(current_state,current_input)
        >>> ## or
        >>> input_dict = {"state":current_state,"input":current_input}
        >>> dynamics = dynamics_fun.call(input_dict)["value"]
        """
        
        
        
        return ca.Function('continuous_dynamics',[self.state_vector,self.input_vector],[self.dynamics_exp],["state","input"],["value"])
   
    @property
    def g_fun(self)->ca.Function:
        """
        Casadi function that can be used to evaluate the g matrix for a given state. This is useful for example to compute the lie derivative of a barrier function along the system
        dynamics -> x_dot = f(x) + g(x)u
        Returns:
            ca.Function: function that can be used to evaluate the g matrix for a given state
        
        Example :
        >>> g_fun = self.g_fun
        >>> g = g_fun(current_state)
        >>> ## or 
        >>> input_dict = {"state":current_state}
        >>> g = g_fun.call(input_dict)["value"]
        """
        return ca.Function('g',[self.state_vector],[self.g],["state"],["value"])
    
    @property
    def f_fun(self)->ca.Function:
        """
        Casadi function that can be used to evaluate the f vector for a given state. This is useful for example to compute the lie derivative of a barrier function along the system
        dynamics -> x_dot = f(x) + g(x)u
        
        Returns:
            ca.Function: function that can be used to evaluate the f vector for a given state
            
        Example :
        >>> f_fun = self.f_fun
        >>> f = f_fun(current_state)
        >>> ## or
        >>> input_dict = {"state":current_state}
        >>> f = f_fun.call(input_dict)["value"]
        """
        
        
        return ca.Function('f',[self.state_vector],[self.f],["state"],["value"])
    

    def _get_step_function(self, rk4Steps: int) -> ca.Function:
        """
        Return a casadi function that can be used to evaluate the next state of the system.
        The function will always have the input names ['state','input'] and the output name ['value'].
        The input is then the state vector for a given tme and the input vector, while the output (the value) is the next state.
        
        Args :
            rk4Steps :  number of Runge-Kutta 4 steps to perform within the time interval
        
        Returns:
            The next step function expressed as a casadi.Function
            
        
        Example 1 :
        >>> step_fun = self._get_step_function(rk4Steps=3)
        >>> next_state = step_fun(current_state,current_input)
        
        Example 1 :
        >>> step_fun = self._get_step_function(rk4Steps=3)
        >>> input_dict = {"state":current_state,"input":current_input}
        >>> next_state = step_fun.call(input_dict)["value"]
        

        """
        return ca.Function('step',[self.state_vector,self.input_vector],[self._symbolic_step(rk4Steps)],["state","input"],["value"])

    
    def _symbolic_step(self, rk4Steps: int) -> ca.MX:
        """
        Return the next state of the system for a given initial state, input vector, time interval, and number of RK4 steps.

        Args:
            rk4Steps: The number of RK4 steps to perform within the time interval

        Returns:
            The next state of the system as a symbolic expression 

        """

        dt = self._time_step/rk4Steps; # length of a control interval
        state = self.state_vector
        for k in range(rk4Steps) : # loop over control interval
            # Runge-Kutta 4 integration
            k1 = self.dynamics_fun(state,         self.input_vector)
            k2 = self.dynamics_fun(state+dt/2*k1, self.input_vector)
            k3 = self.dynamics_fun(state+dt/2*k2, self.input_vector)
            k4 = self.dynamics_fun(state+dt*k3,   self.input_vector)
            state = state + dt/6*(k1+2*k2+2*k3+k4)
        return state
    
    

    def full_lie_derivative(self,b:ca.MX) -> ca.MX:
        """
        Returns the full lie derivative of b along the system (expression in casadi MX)
        Note that the expression should be a function of the state_vector that is an attribute of the dynamical model
        
        Args :
            b : The barrier function expressed as a casadi MX (needs to be a function of the state_vector of the dynamical model)
        Returns:
                The full lie derivative of b along the system (expression in casadi MX)
        """

        return ca.jacobian(b,self.state_vector) @ self.dynamics_exp 
    
    def lie_derivative_g(self,b:ca.MX) -> ca.MX: 
        """
        return the lie derivative of b along the system(expression in casadi MX)
        Note that the expression should be a function of the state_vector that is an attribute of the dynamical model 
        """
        return ca.jacobian(b,self.state_vector) @ self.g 
    
    def lie_derivative_f(self,b:ca.MX) -> ca.MX:
        """
        return the lie derivative of b along the system (expression in casadi MX)
        Note that the expression is a function should be a function the state_vector that is an attribute of the dynamical model 
        """
        return ca.jacobian(b,self.state_vector) @ self.f
    
    def lie_derivative_f_function(self, b:ca.Function) -> ca.Function:
        """
        Given a barrier function in the form b("state_1","state_2",...."time") is returns the Lie derivative along f as of the dynamical system
        as a function of the states and time. The state of the current dynamical system should also be among the states or an error will be thrown
        
        Args :
            b : The barrier function expressed as a casadi function
        
        Returns :
            A casadi function that takes the states and time as input and returns the Lie derivative of b along f
            
        """
        
        input_names = b.name_in()
        ids  = [get_id_from_name(name) for name in input_names if "state" in name.split('_')]
        if not self._unique_identifier in ids:
            raise ValueError(f"The barrier function {b} does not contain the state of the current dynamical system. states found are {ids} but agent has state {self._unique_identifier}")
        
        
        named_inputs = {} # symbolic inputs to be given to the function
        for name in input_names:
            if name == "time":
                named_inputs[name] = ca.MX.sym(name)
            else :
                unique_identifier = get_id_from_name(name)
                
                if unique_identifier  == None:
                    raise ValueError(f"The barrier function {b} has an input called {name} that is not among the states of the dynamical system. This is not allowed. Only names like 'state_id' and 'time' are allowed")
                
                if unique_identifier == self.unique_identifier:
                    named_inputs[name] = self.state_vector
                else: 
                    named_inputs[name] = ca.MX.sym(name,b.size1_in(name))
            
        b_exp = b.call(named_inputs)["value"]
        lie_f_b = ca.jacobian(b_exp,self.state_vector) @ self.f
        
        
        return ca.Function("lie_f_"+b.name(),list(named_inputs.values()),[lie_f_b],list(named_inputs.keys()),["value"]) 
        
    def lie_derivative_g_function(self, b:ca.Function) -> ca.Function:
        """
        Given a barrier function in the form b("state_1","state_2",...."time") is returns the Lie derivative along g as of the dynamical system
        as a function of the states and time. The state of the current dynamical system should also be among the states or an error will be thrown
        
        Args :
            b : The barrier function expressed as a casadi function
        Returns :
            A casadi function that takes the states and time as input and returns the Lie derivative of b along g
        
        Example :
        >>> state = DynamicalModel.state_vector
        >>> b_exp = state.T@state 
        >>> b = ca.Function("b",[state],[b_exp],["state_1","time"],["value"])
        >>> b = ca.Function("b",["state_1","state_2","time"],["state_1**2+state_2**2"])
        >>> lie_g_b = self.lie_derivative_g_function(b)
            
        """
        
        input_names = b.name_in()
        ids  = [get_id_from_name(name) for name in input_names if "state" in name.split('_')]
        if not self._unique_identifier in ids:
            raise ValueError(f"The barrier function {b} does not contain the state of the current dynamical system. states found are {ids} but agent has state {self._unique_identifier}")
        
        
        named_inputs = {} # symbolic inputs to be given to the function
        for name in input_names:
            if name == "time":
                named_inputs[name] = ca.MX.sym(name)
            else :
                unique_identifier = get_id_from_name(name)
                if unique_identifier == self.unique_identifier:
                    named_inputs[name] = self.state_vector
                else: 
                    named_inputs[name] = ca.MX.sym(name,b.size1_in(name))
            
        b_exp = b.call(named_inputs)["value"]
        lie_g_b = ca.jacobian(b_exp,self.state_vector) @ self.g
        
        
        return ca.Function("lie_f_"+b.name(),list(named_inputs.values()),[lie_g_b],list(named_inputs.keys()),["value"])    
    
    
    def is_relative_degree_1(self, b:ca.MX) -> int:    
        directional_gradient = self.lie_derivative_g(b)
        return directional_gradient.is_zero()


    def specs(self) -> str:
        string = ""
        string+= "Dynamical Model\n"
        string+= "input vector size: " + str(self.input_vector.size()) + "\n"
        string+= "state vector size: " + str(self.state_vector.size()) + "\n"

        return string
    
    
    def __str__(self) :
        return self.__class__.__name__ + "\n" + self.specs()


    
class DifferentialDrive(DynamicalModel):
    """
    Class Differential Drive - A simple model of a differential drive robot.
    
    Args:
        wheel_base (float): The distance between the front and rear axles of the car.
        max_velocity (float): The maximum velocity capability of the car.
        angular_velocity (float): The maximum steering angle of the car.
    """
    count = 0
     
    def __init__(self, max_speed: float, max_angular_velocity: float, look_ahead_distance: float,unique_identifier : UniqueIdentifier) -> None:
        super().__init__(unique_identifier=unique_identifier)
    
        self._max_speed             = max_speed            # max longitudinal speed
        self._max_angular_velocity  = max_angular_velocity # max angular velocity
        self._look_ahead_distance   = look_ahead_distance  # diffeormorphism to have a first order model of speed which respect to the angular vecolocity

        if self._max_speed <= 0:
            raise ValueError("Max velocity must be positive.")
        if  self._max_angular_velocity  <= 0:
            raise ValueError("Max angular velocity must be positive.")
        
        
        #input vector
        self._input_vector = ca.MX.sym(wrap_name("input",self._unique_identifier),2)
        #state vector
        self._state_vector = ca.MX.sym(wrap_name("state",self._unique_identifier),3)
        
        # separate inputs
        self._velocity     = self._input_vector[0]
        self._turning_rate = self._input_vector[1]
        
        # create the state variables. Node that a unique unique_identifier is given for debugging purposes (KEEP THIS TEMPLATE)
        self._position = self._state_vector[:2]
        self._heading  = self._state_vector[2]

        # create dynamics model
        row1 = ca.horzcat(ca.cos(self._heading), -self._look_ahead_distance*ca.sin(self._heading))
        row2 = ca.horzcat(ca.sin(self._heading), self._look_ahead_distance*ca.cos(self._heading))
        row3 = ca.horzcat(0,1)
        self._g = ca.vertcat(row1,row2,row3)
        self._f = ca.vertcat(0,0,0)

        #the order must respect the one given in the state and input vector!
        self._input_constraints_A, self._input_constraints_b,self._input_constraints_vertices = create_box_constraint_function([[-self._max_speed, self._max_speed], [-self._max_angular_velocity, self._max_angular_velocity]])    
        self._step_fun = self._get_step_function( rk4Steps=3) # created at initalization
    
    @property
    def max_speed(self) -> float:
        return self._max_speed

    @property
    def max_angular_velocity(self) -> float:
        return self._max_angular_velocity

    @property
    def look_ahead_distance(self) -> float:
        return self._look_ahead_distance

    @property
    def g(self)->ca.MX:
        return self._g
    
    @property
    def f(self)->ca.MX:
        return self._f
    
    # define here the input vector and the state vector
    @property
    def input_vector(self) -> ca.MX:
        return self._input_vector

    @property
    def state_vector(self) -> ca.MX:
        return self._state_vector

    @property
    def dynamics_exp(self) -> ca.MX:
        return self._f + self._g @ self._input_vector
    
    @property
    def step_fun(self)->ca.Function:
        return self._step_fun

    @property
    def maximum_expressible_speed(self)->float:
        return np.sqrt(self._max_speed**2 + (self._max_angular_velocity*self._look_ahead_distance)**2)
    
    @property
    def input_constraints_A(self)->ca.MX:
        """Matrix of the input constraints Au<=b"""
        return self._input_constraints_A
    
    @property
    def input_constraints_b(self)->ca.MX:
        """Vector of the input constraints Au<=b"""
        return self._input_constraints_b
    
    

        
class SingleIntegrator2D(DynamicalModel):
    """
    Class Single Integrator
    
    Args:
        max_velocity (float): The maximum velocity 
    """

    def __init__(self, max_velocity: float,unique_identifier:UniqueIdentifier) -> None:
        super().__init__(unique_identifier=unique_identifier)
        
        self._max_velocity = max_velocity
        if self._max_velocity <= 0:
            raise ValueError("Max velocity must be positive.")
        
        
        #input vector
        self._input_vector = ca.MX.sym(wrap_name("input",self._unique_identifier), 2)
        #state vector
        self._state_vector = ca.MX.sym(wrap_name("state",self._unique_identifier), 2)
        
        # create dynamics model
        self._g = np.eye(2)
        self._f = ca.vertcat(0,0)
        
        # control input
        self._position = self._state_vector

        self._input_constraints_A, self._input_constraints_b,self._input_constraints_vertices = create_approximate_ball_constraints2d(radius=self._max_velocity,points_number=20)
        self._step_fun = self._get_step_function(rk4Steps=3) # created at initalization

   
    @property
    def maximum_expressible_speed(self)->float:
        return self._max_velocity

    @property
    def g(self)->ca.MX:
        return self._g
    @property
    def f(self)->ca.MX:
        return self._f
   

    # define here the input vector and the state vector
    @property
    def input_vector(self) -> ca.MX:
        return self._input_vector

    @property
    def state_vector(self) -> ca.MX:
        return self._state_vector

    @property
    def dynamics_exp(self) -> ca.MX:
        return self._f + self._g @ self._input_vector

    
    @property
    def input_constraints_A(self)->ca.MX:
        """Matrix of the input constraints Au<=b"""
        return self._input_constraints_A
    
    @property
    def input_constraints_b(self)->ca.MX:
        """Vector of the input constraints Au<=b"""
        return self._input_constraints_b
    
    @property
    def step_fun(self)->ca.Function:
        return self._step_fun



class DoubleIntegrator2D(DynamicalModel):
    """
    Class Single Integrator
    
    Args:
        max_velocity (float): The maximum velocity 
    """

    def __init__(self, max_acceleration: float,unique_identifier:UniqueIdentifier) -> None:
        super().__init__(unique_identifier=unique_identifier) # instatiating the super class
        
        self._max_acceleration = max_acceleration
        
        if self._max_acceleration <= 0:
            raise ValueError("Max velocity must be positive.")
        
        
        self._state_vector = ca.MX.sym(wrap_name("state",self._unique_identifier), 4)
        self._input_vector = ca.MX.sym(wrap_name("input",self._unique_identifier),2)
        
        # create the state variables
        self._position = self._state_vector[:2]
        self._velocity = self._state_vector[2:]

        # create dynamics model
        self._g = ca.vertcat(np.zeros((2,2)),np.eye(2))
        self._f = ca.vertcat(self._velocity,0,0) # no state drift
        
        A,B,vertices = create_approximate_ball_constraints2d(radius = self._max_acceleration,points_number=20)
        
        self._input_constraints_A = A
        self._input_constraints_b = B
        self._input_constraints_vertices = vertices
        
        self._step_fun = self._get_step_function(rk4Steps=3) # created at initalization


    @property
    def g(self)->ca.MX:
        return self._g
    @property
    def f(self)->ca.MX:
        return self._f
    
    # define here the input vector and the state vector
    @property
    def input_vector(self) -> ca.MX:
        return self._input_vector

    @property
    def state_vector(self) -> ca.MX:
        return self._state_vector

    @property
    def dynamics_exp(self) -> ca.MX:
        return self._f + self._g @ self._input_vector

    
    @property
    def input_constraints_A(self)->ca.MX:
        """Matrix of the input constraints Au<=b"""
        return self._input_constraints_A
    
    @property
    def input_constraints_b(self)->ca.MX:
        """Vector of the input constraints Au<=b"""
        return self._input_constraints_b    

    @property
    def step_fun(self)->ca.Function:
        return self._step_fun
    



# here we provide some support classes to create circular and box type constraints matrices Ax-b <=0 

def create_approximate_ball_constraints2d(radius:float,points_number:int)-> tuple[np.ndarray,np.ndarray,np.ndarray] :
    """
    Computes constraints matrix and vector A,b for an approximation of a ball constraint with a given radius.
    The how many planes will be used to approximate the ball. We cap the value of the planes to 80 to avoid numerical errors in the computation of the normals
    
    Args :
        radius : The radius of the ball
        points_number : The number of planes used to approximate the ball
        
    Returns:
        A : The constraints matrix and vector for the ball constraint
        b : The constraints matrix and vector for the ball constraint
        vertices : The vertices of the polygon that approximates the ball
        
    
    Notes :
    To compute a center shift in the constraints it is sufficient to recompute b as b = b -+ A@center_shift. Indeeed A(x-c)-b = Ax - Ac - b = Ax - (b + Ac) 
    To compute a scaled version of the obtained polygone it is sufficient to scale b  by a scalar gamma> 0. Ax-gamma*b<=0 is the polygon scaled by gamma
    
    """

    # create constraints
    Arows    = []
    brows    = []
    vertices = []
    
    points_number = min(points_number,80) # to avoid numerical error on the computation of the normals
    
    for i in range(0,points_number):
        angle = 2 * np.pi * i / (points_number-1)
        vertices += [np.array([radius*np.cos(angle), radius*np.sin(angle)])]
        
    for j in range(0,len(vertices)) :
        
        tangent = vertices[j] - vertices[j-1]
        norm_tangent = np.linalg.norm(tangent)
        outer_normal = np.array([tangent[1],-tangent[0]])/norm_tangent
        
        b = np.sqrt(radius**2 - (norm_tangent/2)**2)
        Arows += [outer_normal]
        brows += [np.array([[b]])]
    
    A = np.row_stack(Arows)
    b = np.row_stack(brows)
    vertices = np.row_stack(vertices)

    return A,b,vertices


def create_box_constraint_function(bounds: list[list[float,float]])-> tuple[np.ndarray,np.ndarray,np.ndarray] :
    """
    Create a casadi function that checks if a given vector is inside the box bounds.
    
    Args :
        bounds : The box bounds in the form np.array([[min1, max1], [min2, max2], ..., [minN, maxN]])
    
    Returns:
        A : The constraints matrix and vector for the ball constraint
        b : The constraints matrix and vector for the ball constraint
        vertices : The vertices of the polygon that approximates the ball
        
    Notes :
    To compute a center shift in the constraints it is sufficient to recompute b as b = b -+ A@center_shift. Indeeed A(x-c)-b = Ax - Ac - b = Ax - (b + Ac) 
    To compute a scaled version of the obtained polygone it is sufficient to scale b  by a scalar gamma> 0. Ax-gamma*b<=0 is the polygon scaled by gamma
    """

    base  = np.eye(len(bounds))
    Arows = []
    brows = []
    vertices = np.array(list(product(*bounds)))

    for i in range(len(bounds)):
        Arows.append(base[i, :])
        Arows.append(-base[i, :])
        try :
            brows.append(bounds[i][1])
            brows.append(-bounds[i][0])
        except :
            raise ValueError(f"Bounds should be a list of lists with two elements. Got {bounds} instead")

    A = np.row_stack(Arows)
    b = np.row_stack(brows)
    vertices = np.row_stack(vertices)

    return A,b,vertices

