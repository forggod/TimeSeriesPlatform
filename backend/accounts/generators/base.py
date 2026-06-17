from abc import ABC, abstractmethod

class BaseAttractor(ABC):

    name = "base"

    @abstractmethod
    def generate(self, **params):
        pass

    @abstractmethod
    def params_schema(self):
        """описание параметров для UI"""
        pass