import numpy as np
import casadi as ca
from abc import ABC, abstractmethod

from ..stl.dynamics  import DynamicalModel
from .utils import NoStdStreams


class ImpactSolver(ABC):
    
    def __init__(self, model : DynamicalModel) -> None :
            
        self._model = model
        self.solver = None
    
    def compute(self, Lg:np.ndarray) -> ca.DM:
        with NoStdStreams():
            solution = self.solver(p = Lg, ubg=0)["x"]
        return solution
    

class BestImpactSolver(ImpactSolver):
    def __init__(self, model : DynamicalModel) -> None:
        super().__init__(model)

        A_u = model.input_constraints_A
        b_u = model.input_constraints_b
        
        Lg       = ca.MX.sym("Lg",model.input_vector.size1())
        cost     = -Lg.T @ model.input_vector
        
        nlp = {'x': model.input_vector, 
               'f': cost,
               'g': A_u@model.input_vector - b_u,
               'p': Lg}
        
        self.solver = ca.qpsol('solver', 'qpoases', nlp)
        

class WorseImpactSolver(ImpactSolver):
    def __init__(self, model : DynamicalModel) -> None :
        
        super().__init__(model)
        
        A_u      = model.input_constraints_A
        b_u      = model.input_constraints_b
        Lg       = ca.MX.sym("Lg",model.input_vector.size1())
        cost     = Lg.T @ model.input_vector
        
        nlp = {'x': model.input_vector, 
               'f': cost,
               'g': A_u@model.input_vector - b_u,
               'p': Lg}
        
        self.solver = ca.qpsol('solver', 'qpoases', nlp)
