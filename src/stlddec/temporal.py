from abc import ABC, abstractmethod
import numpy as np

class TimeInterval :
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
    
    def __lt__(self,timeInt:"TimeInterval") -> "TimeInterval":
        """strict subset relations self included in timeInt ::: "TimeInterval" < timeInt """
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
    
    def __eq__(self,timeInt:"TimeInterval") -> bool:
        """ equality check """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if a1 == a2 and b1 == b2 :
            return True
        else :
            return False
        
    def __ne__(self,timeInt:"TimeInterval") -> bool :
        """ inequality check """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if a1 == a2 and b1 == b2 :
            return False
        else :
            return True
        
    def __le__(self,timeInt:"TimeInterval") -> "TimeInterval" :
        """subset relations self included in timeInt  ::: "TimeInterval" < timeInt """
        
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
        
    def __truediv__(self,timeInt:"TimeInterval") -> "TimeInterval" :
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
    

    def getCopy(self) :
        return TimeInterval(a = self.a, b=self.b)
    

class TemporalOperator(ABC):
    def __init__(self,time_interval:TimeInterval) -> None:
        self._time_interval         : TimeInterval = time_interval
    
    @property
    @abstractmethod
    def time_of_satisfaction(self) -> float:
        pass
    
    @property
    @abstractmethod
    def time_of_remotion(self) -> float:
        pass
    @property
    def time_interval(self) -> TimeInterval:
        return self._time_interval
    
    
    

class AlwaysOperator(TemporalOperator):
    def __init__(self,time_interval:TimeInterval) -> None:
        
        super().__init__(time_interval)
        self._time_of_satisfaction   : float        = self._time_interval.a
        self._time_of_remotion      : float        = self._time_interval.b
    
    @property
    def time_of_satisfaction(self) -> float:
        return self._time_of_satisfaction
    @property
    def time_of_remotion(self) -> float:
        return self._time_of_remotion

    
class EventuallyOperator(TemporalOperator):
    def __init__(self,time_interval:TimeInterval,time_of_satisfaction:float=None) -> None:
        """ Eventually operator
        Args:
            time_interval (TimeInterval): time interval that referes to the eventually operator
            time_of_satisfaction (float): time at which the eventually operator is satisfied (assigned randomly if not specified)

        Raises:
            ValueError: if time of satisfaction is outside the time interval range
        """
        super().__init__(time_interval)
        self._time_of_satisfaction : float       = time_of_satisfaction
        self._time_of_remotion    : float        = self._time_interval.b
        
        if time_of_satisfaction == None :
            self._time_of_satisfaction = time_interval.a + np.random.rand()*(time_interval.b- time_interval.a)
            
        elif time_of_satisfaction<time_interval.a or time_of_satisfaction>time_interval.b :
            raise ValueError(f"For eventually formulas you need to specify a time a satisfaction for the formula in the range of your time interval [{time_interval.a},{time_interval.b}]")
        
    @property
    def time_of_satisfaction(self) -> float:
        return self._time_of_satisfaction
    
    @property
    def time_of_remotion(self) -> float:
        return self._time_of_remotion



#!TODO: Still under development
class AlwaysEventuallyOperator(TemporalOperator):
    def __init__(self,always_time_interval:TimeInterval,eventually_time_interval:TimeInterval,eventually_time_of_satisfaction:float=None) -> None:
        
        #pick it random if not given
        if eventually_time_of_satisfaction == None :
            eventually_time_of_satisfaction = eventually_time_interval.a + np.random.rand()*(eventually_time_interval.b- eventually_time_interval.a) # random time of satisfaction
        else :
            if eventually_time_of_satisfaction<eventually_time_interval.a or eventually_time_of_satisfaction>eventually_time_interval.b :
                raise ValueError(f"For eventually formulas you need to specify a time a satisfaction for the formula in the range of your time interval [{eventually_time_interval.a},{eventually_time_interval.b}]")
        
        self._period               : TimeInterval = eventually_time_of_satisfaction # from the point "a" of the evetually, we have that the task is satisfied everty evetually_time_of_satsifaction
        
        self._time_of_satisfaction : float       = always_time_interval.a + eventually_time_interval.a # we satisfy at the initial time of the always first
        
        self._time_of_remotion     : float       = self._time_of_satisfaction +  np.ceil((always_time_interval.b - self._time_of_satisfaction)/ self._period) * self._period

        
    @property
    def time_of_satisfaction(self) -> float:
        return self._time_of_satisfaction
    
    @property
    def time_of_remotion(self) -> float:
        return self._time_of_remotion
   
    @property
    def period(self) -> TimeInterval:
        return self._period

class EventuallyAlwaysOperator(TemporalOperator):
    def __init__(self,always_time_interval:TimeInterval,eventually_time_interval:TimeInterval,eventually_time_of_satisfaction:float=None) -> None:
        
        
        if eventually_time_of_satisfaction == None :
             self._time_of_satisfaction = eventually_time_interval.a + np.random.rand()*(eventually_time_interval.b- eventually_time_interval.a) # random time of satisfaction
        else :
            if eventually_time_of_satisfaction<eventually_time_interval.a or eventually_time_of_satisfaction>eventually_time_interval.b :
                raise ValueError(f"For eventually formulas you need to specify a time a satisfaction for the formula in the range of your time interval [{eventually_time_interval.a},{eventually_time_interval.b}]")
            
        self._time_of_satisfaction : float    = eventually_time_of_satisfaction
        self._time_of_remotion     : float    = self._time_of_satisfaction  + always_time_interval.period

    @property
    def time_of_satisfaction(self) -> float:
        return self._time_of_satisfaction
    
    @property
    def time_of_remotion(self) -> float:
        return self._time_of_remotion
   