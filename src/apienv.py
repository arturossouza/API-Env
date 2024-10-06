import itertools
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class APIEnv(gym.Env):
    """Custom environment to model the API with all possible transitions and gradual evolution."""

    def __init__(self):
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
        self.actions = [
            "Increase_CPU",
            "Increase_CPU_Slightly",
            "Decrease_CPU",
            "Decrease_CPU_Slightly",
            "Corrective_Maintenance",
            "Preventive_Maintenance",
            "Restart_Components",
            "Update_Version",
            "Rollback_Version",
            "Add_Memory",
            "Remove_Memory",
        ]
        self.action_space = spaces.Discrete(len(self.actions))

        self.__availability_rewards = {"Available": 5, "Offline": -40}

        self.__response_speed_rewards = {"Fast": 7, "Medium": -2, "Slow": -5}

        self.__health_rewards = {"Healthy": 5, "Error": -10, "Overloaded": -8}

        self.__request_capacity_rewards = {"Low": -5, "Medium": -1, "High": 2}

        # Definindo a função de recompensas (R) para todos os estados
        self.states_rewards = self.generate_rewards()

        # Penalidades para ações que consomem muitos recursos
        self.action_rewards = {
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
        }

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
                    main_prob = random.uniform(0.7, 0.9)
                elif action in ["Corrective_Maintenance", "Preventive_Maintenance"]:
                    main_prob = random.uniform(0.6, 0.7)
                elif action in ["Restart_Components"]:
                    main_prob = random.uniform(0.8, 0.95)
                elif action in ["Add_Memory", "Remove_Memory"]:
                    main_prob = random.uniform(0.65, 0.85)
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

        # Atualização da Disponibilidade (avail)
        if action in ["Increase_CPU", "Increase_CPU_Slightly"]:
            avail = "Available"
        elif action in ["Decrease_CPU", "Decrease_CPU_Slightly"]:
            if health in ["Error", "Overloaded"]:
                avail = "Offline"
            else:
                avail = "Available"
        elif action in [
            "Corrective_Maintenance",
            "Preventive_Maintenance",
            "Restart_Components",
        ]:
            avail = "Available"
        elif action == "Update_Version":
            if random.random() > 0.5:
                avail = "Offline"
                health = "Error"
            else:
                avail = "Available"
                health = "Healthy"
        elif action == "Rollback_Version":
            avail = "Available"
            health = "Healthy"
        elif action in ["Add_Memory", "Remove_Memory"]:
            avail = "Available" if health != "Error" else "Offline"

        # Atualização da Velocidade (speed)
        if action == "Increase_CPU":
            if speed in ["Slow", "Medium"]:
                speed = "Fast"
        elif action == "Increase_CPU_Slightly":
            if speed == "Slow":
                speed = "Medium"
        elif action == "Decrease_CPU":
            if speed == "Fast":
                speed = "Medium"
            elif speed == "Medium":
                speed = "Slow"
        elif action == "Decrease_CPU_Slightly":
            if speed == "Fast":
                speed = "Medium"

        # Atualização do Estado de Saúde (health)
        if action == "Corrective_Maintenance":
            health = "Healthy"
        elif action == "Preventive_Maintenance":
            if health == "Overloaded":
                health = "Healthy"
        elif action == "Restart_Components":
            health = "Healthy"
        elif action == "Update_Version":
            health = "Error" if random.random() > 0.5 else "Healthy"
        elif action == "Rollback_Version":
            health = "Healthy"

        # Atualização da Capacidade (capacity)
        if action in ["Increase_CPU", "Increase_CPU_Slightly"]:
            if capacity == "Low":
                capacity = "Medium"
            elif capacity == "Medium":
                capacity = "High"
        elif action == "Decrease_CPU":
            if capacity == "High":
                capacity = "Medium"
            elif capacity == "Medium":
                capacity = "Low"
        elif action == "Decrease_CPU_Slightly":
            if capacity == "High":
                capacity = "Medium"
            elif capacity == "Medium":
                capacity = "Low"

        # **Intervenções em outras features por ações de manutenção**
        # Para 'Corrective_Maintenance'
        if action == "Corrective_Maintenance":
            if speed == "Slow":
                speed = "Medium"  # Correção pode melhorar a velocidade
            if capacity == "Low":
                capacity = "Medium"  # Melhora de capacidade com manutenção corretiva
            avail = "Available"  # Disponibilidade garantida após manutenção

        # Para 'Preventive_Maintenance'
        elif action == "Preventive_Maintenance":
            if speed == "Fast":
                speed = "Medium"  # Manutenção preventiva pode desacelerar um pouco o sistema
            if capacity == "High":
                capacity = (
                    "Medium"  # Prevenção pode reduzir a capacidade momentaneamente
                )
            avail = "Available"  # Sistema mantido disponível

        # Para 'Restart_Components'
        elif action == "Restart_Components":
            speed = "Medium"  # O reinício tende a estabilizar o sistema em 'Medium'
            capacity = "Medium"  # Capacidade estabilizada após reiniciar
            health = "Healthy"  # Reinício corrige problemas
            avail = "Available"  # Disponibilidade garantida após reinício

        return f"{avail}_{speed}_{health}_{capacity}"

    def __adjust_secondary_state(self, state, action):
        avail, speed, health, capacity = state.split("_")

        # Lógica de transição secundária (efeito contrário)

        # Atualização da Disponibilidade (avail)
        if action in ["Increase_CPU", "Increase_CPU_Slightly"]:
            avail = "Available" if health != "Error" else "Offline"
        elif action in ["Decrease_CPU", "Decrease_CPU_Slightly"]:
            if health in ["Error", "Overloaded"]:
                avail = (
                    "Available"  # Efeito contrário: manter disponível mesmo com erro
                )
            else:
                avail = "Offline"
        elif action in [
            "Corrective_Maintenance",
            "Preventive_Maintenance",
            "Restart_Components",
        ]:
            avail = "Offline"  # Efeito contrário: sistema cai após manutenção
        elif action == "Update_Version":
            if random.random() > 0.5:
                avail = "Available"
            else:
                avail = "Offline"
        elif action == "Rollback_Version":
            avail = "Offline"  # Efeito contrário: rollback deixa o sistema offline
        elif action in ["Add_Memory", "Remove_Memory"]:
            avail = "Offline" if health == "Error" else "Available"

        # Atualização da Velocidade (speed)
        if action == "Increase_CPU":
            if speed == "Fast":
                speed = "Medium"  # Efeito contrário: velocidade diminui
            elif speed == "Medium":
                speed = "Slow"
        elif action == "Increase_CPU_Slightly":
            if speed == "Medium":
                speed = "Slow"
        elif action == "Decrease_CPU":
            if speed == "Slow":
                speed = (
                    "Fast"  # Efeito contrário: aumento de velocidade após diminuir CPU
                )
            elif speed == "Medium":
                speed = "Fast"
        elif action == "Decrease_CPU_Slightly":
            if speed == "Slow":
                speed = "Fast"

        # Atualização do Estado de Saúde (health)
        if action == "Corrective_Maintenance":
            health = (
                "Overloaded" if health == "Healthy" else "Error"
            )  # Correção faz o sistema falhar
        elif action == "Preventive_Maintenance":
            if health == "Healthy":
                health = "Overloaded"  # Prevenção sobrecarrega o sistema
        elif action == "Restart_Components":
            health = "Error"  # Efeito contrário: reiniciar piora o estado de saúde
        elif action == "Update_Version":
            health = (
                "Error" if random.random() > 0.5 else "Overloaded"
            )  # Versão causa sobrecarga ou erro
        elif action == "Rollback_Version":
            health = "Error"

        # Atualização da Capacidade (capacity)
        if action in ["Increase_CPU", "Increase_CPU_Slightly"]:
            if capacity == "High":
                capacity = (
                    "Low"  # Efeito contrário: redução da capacidade ao aumentar CPU
                )
            elif capacity == "Medium":
                capacity = "Low"
        elif action == "Decrease_CPU":
            if capacity == "Low":
                capacity = (
                    "High"  # Efeito contrário: aumento da capacidade ao diminuir CPU
                )
            elif capacity == "Medium":
                capacity = "High"
        elif action == "Decrease_CPU_Slightly":
            if capacity == "Low":
                capacity = "Medium"

        # **Intervenções contrárias nas outras features por ações de manutenção**
        # Para 'Corrective_Maintenance'
        if action == "Corrective_Maintenance":
            if speed == "Medium":
                speed = "Slow"  # Efeito contrário: manutenção reduz a velocidade
            if capacity == "Medium":
                capacity = "Low"  # Capacidade diminui com a manutenção corretiva
            avail = "Offline"  # Disponibilidade cai após a manutenção

        # Para 'Preventive_Maintenance'
        elif action == "Preventive_Maintenance":
            if speed == "Fast":
                speed = "Slow"  # Manutenção preventiva reduz muito a velocidade
            if capacity == "High":
                capacity = "Low"  # Capacidade cai drasticamente
            avail = "Offline"  # Sistema cai preventivamente

        # Para 'Restart_Components'
        elif action == "Restart_Components":
            speed = "Slow"  # Reinício causa lentidão
            capacity = "Low"  # Capacidade reduzida após reinício
            health = "Error"  # Reinício piora o estado de saúde
            avail = "Offline"  # Sistema cai após reinício

        return f"{avail}_{speed}_{health}_{capacity}"


env = APIEnv()
