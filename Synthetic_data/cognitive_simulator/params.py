from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Param(ABC):
    ability_index: int
    max_change: float
    min: float = 0.0
    max: float = 10.0

    def normalize(self, param):
        return (param - self.min) / (self.max - self.min)
    
    @abstractmethod
    def get_multiplier(self, param):
        pass
    
@dataclass(kw_only=True)
class LinParam(Param):
    def get_multiplier(self, param):
        return 1 + (self.max_change - 1) * self.normalize(param)
    
@dataclass(kw_only=True)
class PowerParam(Param):
    exponent: float = 2.0
    def get_multiplier(self, param):
        return 1 + (self.max_change - 1) * (self.normalize(param) ** self.exponent)

@dataclass(kw_only=True)
class ExpParam(Param):
    def get_multiplier(self, param):
        return (self.max_change) ** self.normalize(param)
