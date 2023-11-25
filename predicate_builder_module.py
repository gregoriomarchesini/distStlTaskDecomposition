import numpy as np
from   matplotlib.patches import Ellipse
import casadi as ca
from typing import Self
import polytope as pc
import uuid 


class TimeInterval() :
    """time interval class"""
    # empty set is represented by a double a=None b = None
    def __init__(self,a:float|int|None = None,b:float|int|None =None) -> None:
        
        
        
        if any([a==None,b==None]) and (not all(([a==None,b==None]))) :
            raise ValueError("only empty set is allowed to have None Values for both the extreems a and b of the interval. Please revise your input")
        elif  any([a==None,b==None]) and (all(([a==None,b==None]))) : # empty set
            self._a = a
            self._b = b
        else :    
            # all the checks 
            if (not isinstance(a,float)) and  (not isinstance(a,int)) :
                raise ValueError("the input a must be a float or int")
            elif a<0 :
                raise ValueError("extremes of time interval must be positive")
            
            # all the checks 
            if (not isinstance(b,float)) and  (not isinstance(b,int)) :
                raise ValueError("the input b must be a float or int")
            elif b<0 :
                raise ValueError("extremes of time interval must be non negative")
            
            if a>b :
                raise ValueError("Time interval must be a couple of non decreasing time instants")
         
        self._a = a
        self._b = b
        
    @property
    def a(self):
        return self._a
    @property
    def b(self):
        return self._b
    
    @property
    def measure(self) :
        if self.isEmpty() :
            return None # empty set has measure None
        return self._b - self._a
    
    @property
    def aslist(self) :
        return [self._a,self._b]
    
    def isEmpty(self)-> bool :
        if (self._a == None) and (self._b == None) :
            return True
        else :
            return False
        
    def isSingular(self)->bool:
        a,b = self._a,self._b
        if a==b :
            return True
        else :
            return False
    
    def __lt__(self,timeInt:Self) -> Self:
        """strict subset relations self included in timeInt ::: Self < timeInt """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if self.isEmpty() and (not timeInt.isEmpty()) :
            return True
        elif (not self.isEmpty()) and timeInt.isEmpty() :
            return False
        elif  self.isEmpty() and timeInt.isEmpty() : # empty set included itself
            return True
        else :
            if (a1<a2) and (b2<b1): # condition for intersectin without inclusion of two intervals
                return True
            else :
                return False
    
    def __eq__(self,timeInt:Self) -> bool:
        """ equality check """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if a1 == a2 and b1 == b2 :
            return True
        else :
            return False
        
    def __ne__(self,timeInt:Self) -> bool :
        """ inequality check """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if a1 == a2 and b1 == b2 :
            return False
        else :
            return True
        
    def __le__(self,timeInt:Self) -> Self :
        """subset relations self included in timeInt  ::: Self < timeInt """
        
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if self.isEmpty() and (not timeInt.isEmpty()) :
            return True
        elif (not self.isEmpty()) and timeInt.isEmpty() :
            return False
        elif  self.isEmpty() and timeInt.isEmpty() : # empty set included itself
            return True
        else :
            if (a1<=a2) and (b2<=b1): # condition for intersectin without inclusion of two intervals
                return True
            else :
                return False
        
    def __truediv__(self,timeInt:Self) -> Self :
        """interval Intersection"""
        
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        
        # the empty set is already in this cases since the empty set is included in any other set
        if timeInt<= self :
            return TimeInterval(a =timeInt.a, b = timeInt.b)
        elif self<= timeInt :
            return TimeInterval(a =self._a, b = self._b)
        else : # intersection case
            if (b1<a2) or (b2<a1) : # no intersection case
                return TimeInterval(a = None, b = None)
            elif (a1<=a2) and (b1<=b2) :
                return TimeInterval(a = a2, b = b1)
            else :
                return TimeInterval(a = a1, b = b2)
    
    
    
    def __str__(self):
        return f"[{self.a},{self.b}]"
    
    
    
    def epsilonRightShrink(self)-> None :
        """ machine precision shrinkig of the time interval on the right hand side"""
        
        if self.isEmpty() :
            return self
        else :
            dt = 2**(-52 + np.floor(np.log2(self._b))) # machine precision difference
            if self._b<= dt :
                raise Exception("Not possible to shrink. The right extreme would be negative after shrinking")
            else :
                self._b = self._b - dt
                
    
    def epsilonRightShrink(self)-> None :
        """ machine precision shrinkig of the time interval on the left hand side"""
        if self.isEmpty() :
            return self
        else :
            dt = 2**(-52 + np.floor(np.log2(self._b))) # machine precision difference
            self._b = self._b + dt

    def getCopy(self) :
        return TimeInterval(a = self.a, b=self.b)
    
    
    
class PolytopicPredicate() :
    """class to define polytopic predicate containing the origin"""
    def __init__(self,A: np.ndarray,b : np.ndarray, center : np.ndarray,isParametric:bool=False) -> None:
        # when the predicate is parameteric, then the center is assumed to be the one assigned to the orginal predicate from which the predicate is derived for the decomspotion

        A,b = normalForm(A,b) # translate in nromal form
        self._polytope   = pc.Polytope(A,b) # zero centered polygone
        self._center     = center # center of the polygone
        self._numHyperplanes , self._stateSpaceDim = np.shape(A)
        
        
        # turn the center as a column
        if np.ndim(self._center) == 0:
            self._center = self._center[:,np.newaxis]
            
        try :
            self._vertices    = pc.extreme(self._polytope).T
            self._numVertices = len(self._vertices[0,:]) 
        except:
            raise RuntimeError("There was an error in the computation of the vertives for the polytope. Make sure that your polytope is closed since this is the main source of failure for the algorithm")

        self._isParametric     = isParametric
    
    @property
    def stateSpaceDim(self)-> int:
        return self._stateSpaceDim
    @property
    def vertices(self) -> np.ndarray:
        return self._vertices
    @property
    def verticesStaked(self) :
        return self.verticesStaked
    @property
    def numVerices(self):
        return self._numVertices
    @property
    def polytope(self):
        return self._polytope
    @property
    def center(self):
        return self._center
    @property
    def A(self):
        return self._polytope.A
    @property
    def b(self):
        return self._polytope.b
    @property
    def numHyperplanes(self):
        return self._numHyperplanes
    @property
    def isParametric(self) :
        return self._isParametric
    

class StlTask( ) :
    
    def __init__(self,temporalOperator:str,
                 timeinterval:TimeInterval,
                 predicate:PolytopicPredicate,
                 source:int = None,
                 target:int = None,
                 timeOfSatisfaction:int=None):
        
        """ 
        Input
        ------------------------------------------------------------------------------------------
        temporalOperator  (str)                    :
        timeinterval      (timeInterval)           :
        predicateFunction (convexPredicateFunction :
        
        Note : When a predicate function is not assigned, the attribute "isParametric" will be set to 1. This 
               entails that the predicate will be defined through optimization.
        """
        
        # if a predicate function is not assigned, it is considered that the predicate is parametric
        
        
        self._predicate              :PolytopicPredicate  = predicate
        self._approximationAvailable :bool                = False
        
        # when the task is parametric, these variables will be defined
        self._centerVar     :ca.MX = None # center the parameteric formula
        self._scaleVar      :ca.MX = None # scale for the parametric formula
        self._etaVar        :ca.MX = None # just the stacked version of the center and scale
        self._concensusVar  :ca.MX = None # concensus variable for the given task
        self._sourceNode    :int   = source
        self._targetNode    :int   = target
        
        
        # set temporal prefix
        if temporalOperator!= "always" and temporalOperator!= "eventually" :
            raise Exception("Only 'eventually' and 'always' temporalPrefixs are accepted")
        else :
            self._temporalOperator   = temporalOperator # always or eventually
        
        if timeinterval.isEmpty() :
            raise NotImplementedError("Sorry, empty time intervals are not currently supported by this class")
        self._timeInterval  = timeinterval
        
        if temporalOperator =="always":
            
            self._timeOfSatisfaction = timeinterval.a 
            self._timeOfRemotion = timeinterval.b
        else :
            if timeOfSatisfaction == None :
                raise ValueError(f"For eventually formulas you need to specify a time a satisfaction for the formula in the range of your time interval [{timeinterval.a},{timeinterval.b}]")
            elif timeOfSatisfaction<timeinterval.a or timeOfSatisfaction>timeinterval.b :
                raise ValueError(f"For eventually formulas you need to specify a time a satisfaction for the formula in the range of your time interval [{timeinterval.a},{timeinterval.b}]")
            else :
                self._timeOfSatisfaction = timeOfSatisfaction
                self._timeOfRemotion     = timeOfSatisfaction
        
        self._isInitialised = False                                   # check if the variables for the optimization where intialised
        
        
    """STL formula class"""
    
    @property
    def predicate(self):
        return self._predicate
    @property
    def temporalOperator(self):
        return self._temporalOperator
    @property
    def timeInterval(self):
        return self._timeInterval      
    @property
    def stateSpaceDimension(self):
        return self._predicate.stateSpaceDim     
    @property
    def isParametric(self):
        return self._predicate.isParametric  
    @property
    def sourceNode(self) :
        return self._sourceNode
    @property
    def targetNode(self):
        return self._targetNode
    @property
    def timeOfSatisfaction(self):
        return self._timeOfSatisfaction
    @property
    def timeOfRemotion(self):
        return self._timeOfRemotion
    @property
    def sourceTarget(self):
        return (self._sourceNode,self._targetNode)
    @sourceTarget.setter
    def sourceTarget(self,source:int,target:int) :
        self._sourceNode = source
        self._targetNode = target
    
    @property
    def hasUndefinedDirection(self):
        return ((self.sourceNode==None) or (self.targetNode == None))
    
    
    @property
    def center(self) : # if the task is parameteric then you get a variable, otherwise you get the true center
        if self.isParametric :
            if self._isInitialised :
                return self._centerVar #ca.MX
            else :
                raise RuntimeError("The predicate seems to be parametric but the variables where not initialised. plase make sure yoiu called the method setOptimizationVariables before calling the center")
        else:
            return self._predicate.center # np.array()
    @property 
    def scale(self) :
        if self.isParametric :
            if self._isInitialised :
                return self._scaleVar #ca.MX
            else :
                raise RuntimeError("The predicate seems to be parametric but the variables where not initialised. plase make sure yoiu called the method setOptimizationVariables before calling the center")
        else:
            return 1 
        
    @property
    def originalCenter(self):
        if self.isParametric :
            return self._predicate.center # when the predicate is parameteric, then the center assigned to the predicate is assumed the one of the orginal predicate from which the orginal derives from
        else :
            raise RuntimeError("The task is not parametric and thus there is no orginal center. You can ask for the property ""center"" to get the center of the task predicate directly. This property is only from parameteric predicates")
      
      
      
    def flip(self) :
        """Flips the direction of the predicate"""
        # swap the source and target
        dummy = self._targetNode
        self._targetNode = self._sourceNode
        self._sourceNode = dummy
        
        # change center direction of the predicate
        self._predicate._center = -self._predicate._center # change the direction of the predicate
        # self._centerVar = - self._centerVar
        
    def setAsParametric(self) :
        self._isParametric = True
        
    def setOptimizationVariables(self,optimizer: ca.Opti) :
        """Sets the optimization variables center and scale"""
        
        if self.isParametric :
            self._centerVar     = optimizer.variable(self.predicate.stateSpaceDim,1)                            # center the parameteric formula
            self._scaleVar      = optimizer.variable(1)                                                         # scale for the parametric formula
            self._etaVar        = ca.vertcat(self._centerVar,self._scaleVar)                                    # optimization variable
            self._isInitialised = True                                                                          # flag for initialization
            
        else :
            raise NotImplementedError("The formula seems to not be parameteric. If you want to set parameters for this formula make sure to set isParameteric to True")

def createParametericTaskFrom(task : StlTask, source:int, target : int) -> StlTask :
    
    newParametericTask = StlTask(temporalOperator= task.temporalOperator,
                                 timeinterval    = task.timeInterval.getCopy(),
                                 predicate       = PolytopicPredicate(A=task.predicate.A,b = task.predicate.b,center=task.predicate.center,isParametric=True),
                                 source          = source,
                                 target          = target,
                                 timeOfSatisfaction = task.timeOfSatisfaction)
    
    return newParametericTask 



def random2DPolygone(numberHyperplanes : int, distanceFromCenter: float,center:np.ndarray) -> PolytopicPredicate:
    
    numberHyperplanes = int(numberHyperplanes) # convert to int
    
    if distanceFromCenter<=0 :
        raise ValueError("distanceFromCenter must be a positive number")
    if numberHyperplanes<=2 :
        raise ValueError("number of hiperplames need to be higher than 2 in two dimensions")
    
    step = 360/numberHyperplanes
    A = np.zeros((numberHyperplanes,2))
    z = np.ones((numberHyperplanes,1))*distanceFromCenter
    
    # S =  -1+ 0.5*np.diag(np.random.random(2))
    S = np.eye(2)
    theta = 2*np.pi*np.random.rand()
    R = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    
    for jj,angle in enumerate(np.deg2rad(np.arange(0,360,step))) :
        A[jj,:] = np.squeeze(R@S@np.array([[np.sin(angle)],[np.cos(angle)]]))
    
    
    return PolytopicPredicate(A = A, b = z,center=center)
    




def regular2DPolygone(numberHyperplanes : int, distanceFromCenter: float,center:np.ndarray) -> PolytopicPredicate :
    
    numberHyperplanes = int(numberHyperplanes) # convert to int
    
    if distanceFromCenter<=0 :
        raise ValueError("distanceFromCenter must be a positive number")
    if numberHyperplanes<=2 :
        raise ValueError("number of hiperplames need to be higher than 2 in two dimensions")
    
    step = 360/numberHyperplanes
    A = np.zeros((numberHyperplanes,2))
    z = np.ones((numberHyperplanes,1))*distanceFromCenter
    theta = 2*np.pi*np.random.rand()
    R = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    for jj,angle in enumerate(np.deg2rad(np.arange(0,360,step))) :
       A[jj,:] = np.squeeze(R@np.array([[np.sin(angle)],[np.cos(angle)]]))
    
    return PolytopicPredicate(A = A, b = z,center=center)



def normalForm(A : np.ndarray,b : np.ndarray) :
    
    
    for jj in range(len(b)) :
        normAi = np.sqrt(np.sum(A[jj,:]**2))
        b[jj]   /= normAi
        A[jj,:] /= normAi

        
        if b[jj] <0 :
            b[jj]   = - b[jj]
            A[jj,:] = - A[jj,:]
    
    return A,b


def rotMatrix2D(angle:float):
    """rotation angle in radians"""
    R = np.array([ np.cos(angle),np.sin(angle),
                  -np.sin(angle),np.cos(angle)])
    
             
        
        
    


















# def getConstraintFromInclusionOf(self,formula : Self) -> list[ca.MX]:
#     """ returns inclusion constraints for "formula" inside the of the self formula instance """
#     if not isinstance(formula,STLformula) :
#         raise ValueError("input formula must be an instance of STL formula")
#     # two cases :
#     # 1) parameteric vs parametric
#     # 2) parameteric vs non-parameteric
#     numVertices = 2**self._stateSpaceDimension
#     constraints = []
    
#     if formula.isParametric and self.isParametric :
        
#         source,target = self.edgeTuple
#         # get the represenations of both formulas with the right verse of the edge
#         vertices      = formula.getHypercubeVertices(sourceNode=source,targetNode=target)
#         A,b           = self.computeLinearHypercubeRepresentation(sourceNode=source,targetNode=target)
#         constraints = []
#         for jj in range(numVertices) : # number of vertices of hypercube is computable given the stateSpaceDimension of the problem
#             constraints +=[ A@vertices[:,jj]-b<=np.zeros((self._stateSpaceDimension*2,1))]
            
#     elif  formula.isParametric and (not self.isParametric) :
#         source,target = self.edgeTuple # check the direction of definition
        
#         if (formula.edgeTuple != self.edgeTuple) and (formula.edgeTuple != (target,source)) : # this happens if the edge is not the same at all
#             raise NotImplementedError("It seems that you are trying to make an inclusion between two formulas that are not part of the same edge. This is not support for now")
        
#         vertices    = formula.getHypercubeVertices(sourceNode=source,targetNode=target)
#         constraints = []
        
#         if self._predicate._isApproximated : # in this case the predicate was approximated so you can do all of this with linear hyperplanes 
#             A,b = self.computeLinearHypercubeRepresentation(sourceNode=source,targetNode=target)
#             for jj in range(numVertices) : # number of vertices of hypercube is computable given the stateSpaceDimension of the problem
#                 constraints +=[ A@vertices[:,jj] - b<=0 ] 
#         else :
#             for jj in range(numVertices) : # number of vertices of hypercube is computable given the stateSpaceDimension of the problem
#                 constraints +=[ self._predicate.function(vertices[:,jj])<=0 ] 
            
#     elif (not formula.isParametric) and self.isParametric :  
#         raise NotImplementedError("Trying to include a non parameteric formula inside a parameteroc one. Not supported")
    
#     return constraints





if __name__ =="__main__" :
    pass
    # # simple one directional plane predicate function
    # h = polytopicSetPredicate(center = np.array([0,0]),planeDirections=[np.array([1.,0])],distances= [-2.0])   
    # print(h(np.array([-2,0])))
    
    
   
    # h = cuboidPredicate(center = np.array([0,0]),dimensions=[3,4])
    
    # print(h(np.array([0,-1.4])))
    # predicateObj= convexPredicateFunction(function=h,centerGuess=np.array([0,0]))
    # print(predicateObj._vertices)
    
    # a = ca.DM([1,2])
    # print(ca.norm_2(a))
    
