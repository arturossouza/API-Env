from abc import ABC, abstractmethod


class Transitions(ABC):
    @abstractmethod
    def get_next_most_likely_state(self, action):
        pass

    @abstractmethod
    def get_next_second_likely_state(self, action):
        pass
