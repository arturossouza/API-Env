import itertools
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.state_transitions.state_action import (
    Availability,
    Capacity,
    Health,
    Maintenance,
    Speed,
)


class APIEnv(gym.Env):
    """Custom environment to model the API with all possible transitions and gradual evolution."""

    def __init__(
        self,
        state_rewards: dict = {
            "availability": {"Available": 5, "Offline": -40},
            "response_speed": {"Fast": 7, "Medium": -2, "Slow": -5},
            "health": {"Healthy": 5, "Error": -10, "Overloaded": -8},
            "request_capacity": {"Low": -5, "Medium": -1, "High": 2},
        },
        actions_penalties: dict = {
            "Increase_CPU": -120,
            "Increase_CPU_Slightly": -20,
            "Decrease_CPU": 3,
            "Decrease_CPU_Slightly": 2,
            "Corrective_Maintenance": -7,
            "Preventive_Maintenance": -3,
            "Restart_Components": -4,
            "Update_Version": -6,
            "Rollback_Version": -16,
            "Add_Memory": -95,
            "Remove_Memory": -2,
        },
    ):
        super(APIEnv, self).__init__()

        # Definindo os estados (S)
        self.states = [
            f"{avail}_{speed}_{health}_{capacity}"
            for avail in ["Offline", "Available"]
            for speed in ["Slow", "Medium", "Fast"]
            for health in ["Healthy", "Overloaded", "Error"]
            for capacity in ["Low", "Medium", "High"]
        ]

        self.state_space = len(self.states)

        # Definindo as ações (A)
        self.actions = list(actions_penalties.keys())
        self.action_space = spaces.Discrete(len(self.actions))

        self.__availability_rewards = state_rewards["availability"]

        self.__response_speed_rewards = state_rewards["response_speed"]

        self.__health_rewards = state_rewards["health"]

        self.__request_capacity_rewards = state_rewards["request_capacity"]

        # Definindo a função de recompensas (R) para todos os estados
        self.states_rewards = self.generate_rewards()

        # Penalidades para ações que consomem muitos recursos
        self.action_rewards = actions_penalties

        # Definindo a função de recompensas (R) para todos os estados
        self.transition_probabilities = self.generate_transitions()

        self.state = None

    def reset(self):
        self.state = "Offline_Slow_Error_Medium"
        return self.states.index(self.state), {}

    def step(self, action):
        action_str = self.actions[action]
        state_str = self.state

        # Verifica se há uma transição válida e calcula a probabilidade
        possible_transitions = self.transition_probabilities.get(
            (state_str, action_str), [(state_str, 1.0)]
        )

        # Escolhe o próximo estado com base nas probabilidades
        next_states, probs = zip(*possible_transitions)
        new_state = np.random.choice(next_states, p=probs)

        # Definindo a recompensa com base no novo estado
        reward_state = self.states_rewards.get(
            new_state, 0
        )  # Recompensa baseada no estado
        reward_action = self.action_rewards.get(
            action_str, 0
        )  # Penalidade baseada na ação

        # Aplicando penalidade da ação no total da recompensa
        total_reward = reward_state + reward_action

        self.state = new_state

        done = new_state in ["Available_Fast_Healthy_High"]

        return self.states.index(new_state), total_reward, done, False, {}

    def render(self, mode="human"):
        if mode == "human":
            print(f"Current API state: {self.state}")
        else:
            raise NotImplementedError(f"Mode {mode} is not supported.")

    def generate_rewards(self):
        availabilities = ["Available", "Offline"]
        speeds = ["Fast", "Medium", "Slow"]
        healths = ["Healthy", "Error", "Overloaded"]
        capacities = ["Low", "Medium", "High"]

        return {
            f"{availability}_{speed}_{health}_{capacity}": self.__calculate_state_reward(
                availability, speed, health, capacity
            )
            for availability, speed, health, capacity in itertools.product(
                availabilities, speeds, healths, capacities
            )
        }

    def __calculate_state_reward(self, availability, speed, health, capacity):
        return (
            self.__availability_rewards[availability]
            + self.__response_speed_rewards[speed]
            + self.__health_rewards[health]
            + self.__request_capacity_rewards[capacity]
        )

    def generate_transitions(self):
        transitions = {}

        for state in self.states:
            for action in self.actions:
                next_states = []

                next_state_main = self.__adjust_state_component(state, action)

                # Definir uma probabilidade para a transição principal
                if action in ["Increase_CPU", "Decrease_CPU"]:
                    main_prob = random.uniform(0.8, 0.9)
                elif action in ["Corrective_Maintenance", "Preventive_Maintenance"]:
                    main_prob = random.uniform(0.7, 0.8)
                elif action in ["Restart_Components"]:
                    main_prob = random.uniform(0.9, 0.95)
                elif action in ["Add_Memory", "Remove_Memory"]:
                    main_prob = random.uniform(0.8, 0.85)
                else:
                    main_prob = random.uniform(0.7, 0.85)

                next_states.append((next_state_main, main_prob))

                next_state_secondary = self.__adjust_secondary_state(state, action)
                secondary_prob = random.uniform(0.05, 0.2)

                if main_prob + secondary_prob > 1:
                    secondary_prob = 1 - main_prob

                next_states.append((next_state_secondary, secondary_prob))

                remain_prob = 1 - (main_prob + secondary_prob)
                if remain_prob < 0:
                    remain_prob = 0

                next_states.append((state, remain_prob))

                transitions[(state, action)] = next_states

        return transitions

    def __adjust_state_component(self, state, action):
        avail, speed, health, capacity = state.split("_")

        avail = Availability(avail, health).get_next_most_likely_state(action)
        speed = Speed(speed).get_next_most_likely_state(action)
        health = Health(health).get_next_most_likely_state(action)
        capacity = Capacity(capacity).get_next_most_likely_state(action)

        if action in ["Corrective_Maintenance", "Preventive_Maintenance", "Restart_Components"]:
            speed, capacity, health = Maintenance(speed, capacity, health).get_next_most_likely_state(action)

        return f"{avail}_{speed}_{health}_{capacity}"

    def __adjust_secondary_state(self, state, action):
        avail, speed, health, capacity = state.split("_")

        avail = Availability(avail, health).get_next_second_likely_state(action)
        speed = Speed(speed).get_next_second_likely_state(action)
        health = Health(health).get_next_second_likely_state(action)
        capacity = Capacity(capacity).get_next_second_likely_state(action)

        if action in ["Corrective_Maintenance", "Preventive_Maintenance", "Restart_Components"]:
            speed, capacity, health = Maintenance(speed, capacity, health).get_next_second_likely_state(action)

        return f"{avail}_{speed}_{health}_{capacity}"


env = APIEnv()
