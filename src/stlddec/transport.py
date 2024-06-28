from typing import Callable
from abc import ABC, abstractmethod 




class Publisher(ABC):
    """
    Simple implementation of a subscriber publisher pattern.
    
    A publisher is simply a class that stores events_type with an assigned set of callbacks that it receives from subscribers.
    When an event occurs, the publisher will call the callbacks with a set of inputs. The subscriber needs to know which input the 
    publisher will assign to its callback for a specific event. Or on the contrary the publisher might need to know which input the subscriber
    requires based on some other information. The publisher-subscriber patter is particulalry useful to change properties of one class from another class 
    for example even if very careful access to private attributed should be considered.
    
    
    """
    def __init__(self) -> None:
        self._subscribers = {}
    
    
    def add_topic(self, event_type:str):
        """A
        dd list of topics
        
        Args:
            event_type (str): [description]
        
        Example :
        >>> A = Publisher()
        >>> a.add_topic("position_event")
        """
        if event_type not in self._subscribers:
            self._subscribers.setdefault(event_type,set())
            
    def register(self, event_type:str, callback:Callable): # register topics the publisher will publish
        """Register a subscriber to an event published by the publisher

        Args:
            event_type : event
            callback   : Collable to be applied for the given event

        Raises:
            ValueError: If the event is not among the topics of the publisher
            
            
        Example :
        >>> A = Publisher()
        >>> a.add_topic("position_event")
        >>> def callback(*args,**kwargs):
        >>>     print("I am a callback")
        >>> a.register("position_event",callback)    
        
        """
        if event_type not in self._subscribers:
            raise ValueError(f"string '{event_type}' is not a valid str for this subscriber. Available events are {list(self._subscribers.keys())}")
        self._subscribers[event_type].add(callback)


    def unregister(self, event_type:str, callback: Callable): # unregister topics the publisher will publish
        """
        Simply unregister an event
        
        
        
        """
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(callback)
        
    @abstractmethod 
    def notify(self, event_type:str, *args, **kwargs):
        """
        A notification is sent to every subscriber when a notification to the given event is sent.
        Here you can define how the call back are called based on the vent for example. 
        
        Args:
            event_type (str): the event that triggers the notification (the topic)
            *args: some argumetns for the call back
            **kwargs: Some Keywards to pass to the callback
        
        Example :
        >>>
        >>> def callback(name = "ciao"):
        >>>     print("I am a callback")
        >>>
        >>> # inside publisher
        >>> def notify(name="Greg"):
        >>>     for callback in self._subscribers[event_type]:
        >>>         callback(name=name)
        """
        
        pass
   
    @property
    def topics(self):
        return list(self._subscribers.keys())
    
    
class Message:
    def __init__(self, sender_id:str):
         self.sender_id = sender_id