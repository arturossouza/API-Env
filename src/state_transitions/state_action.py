import random

from src.state_transitions.transitions import Transitions


class Availability(Transitions):
    def __init__(self, avail, health):
        self.avail = avail
        self.health = health

    def get_next_most_likely_state(self, action):
        if action in ["Decrease_CPU", "Decrease_CPU_Slightly"]:
            return "Offline" if self.health in ["Error", "Overloaded"] else "Available"
        elif action in ["Corrective_Maintenance", "Preventive_Maintenance", "Restart_Components"]:
            return "Available"
        elif action == "Update_Version":
            return "Offline" if self.health == "Error" else "Available"
        elif action == "Rollback_Version":
            return "Available"
        elif action == "Add_Memory":
            return "Available"
        elif action == "Remove_Memory":
            return "Available" if self.health != "Error" else "Offline"
        return self.avail

    def get_next_second_likely_state(self, action):
        if action in ["Decrease_CPU", "Decrease_CPU_Slightly"]:
            return "Offline" if self.health in ["Error", "Overloaded"] else "Available"
        elif action in ["Corrective_Maintenance", "Preventive_Maintenance", "Restart_Components"]:
            return "Offline"
        elif action == "Update_Version":
            return "Offline" if self.health == "Error" else "Available"
        elif action == "Rollback_Version":
            return "Offline"
        elif action == "Add_Memory":
            return "Available"
        elif action == "Remove_Memory":
            return "Offline" if self.health == "Error" else "Available"
        return self.avail


class Speed(Transitions):
    def __init__(self, speed):
        self.speed = speed

    def get_next_most_likely_state(self, action):
        if action == "Increase_CPU":
            return "Fast" if self.speed in ["Slow", "Medium"] else self.speed
        elif action == "Increase_CPU_Slightly":
            return "Medium" if self.speed == "Slow" else self.speed
        elif action == "Decrease_CPU":
            return "Medium" if self.speed == "Fast" else "Slow"
        elif action == "Decrease_CPU_Slightly":
            return "Medium" if self.speed == "Fast" else self.speed
        elif action == "Update_Version":
            return "Medium"  # Versão atualiza a velocidade para média
        elif action == "Rollback_Version":
            return "Slow"  # Rollback reverte para velocidade lenta
        elif action == "Remove_Memory":
            return "Slow"  # Remover memória diminui a velocidade
        return self.speed

    def get_next_second_likely_state(self, action):
        if action == "Increase_CPU":
            return "Medium" if self.speed == "Fast" else "Slow"
        elif action == "Increase_CPU_Slightly":
            return "Slow" if self.speed == "Medium" else self.speed
        elif action == "Decrease_CPU":
            return "Fast" if self.speed == "Slow" else "Medium"
        elif action == "Decrease_CPU_Slightly":
            return "Fast" if self.speed == "Slow" else self.speed
        elif action == "Update_Version":
            return "Fast"
        elif action == "Rollback_Version":
            return "Medium"
        elif action == "Add_Memory":
            return "Slow"
        elif action == "Remove_Memory":
            return "Fast"
        return self.speed

class Health(Transitions):
    def __init__(self, health):
        self.health = health

    def get_next_most_likely_state(self, action):
        if action == "Corrective_Maintenance":
            return "Healthy"
        elif action == "Preventive_Maintenance":
            return "Healthy" if self.health == "Overloaded" else self.health
        elif action == "Restart_Components":
            return "Healthy"
        elif action == "Update_Version":
            return "Error" if random.random() > 0.5 else "Healthy"
        elif action == "Rollback_Version":
            return "Healthy"
        elif action == "Add_Memory":
            return "Healthy" if self.health != "Error" else "Overloaded"
        elif action == "Remove_Memory":
            return "Overloaded"  # Remover memória pode causar sobrecarga
        return self.health

    def get_next_second_likely_state(self, action):
        if action == "Corrective_Maintenance":
            return "Error" if self.health == "Healthy" else "Overloaded"
        elif action == "Preventive_Maintenance":
            return "Overloaded" if self.health == "Healthy" else self.health
        elif action == "Restart_Components":
            return "Error"
        elif action == "Update_Version":
            return "Error" if random.random() > 0.5 else "Overloaded"
        elif action == "Rollback_Version":
            return "Error"
        elif action == "Add_Memory":
            return "Healthy"
        elif action == "Remove_Memory":
            return "Error"
        return self.health


class Capacity(Transitions):
    def __init__(self, capacity):
        self.capacity = capacity

    def get_next_most_likely_state(self, action):
        if action in ["Decrease_CPU", "Decrease_CPU_Slightly"]:
            return "Medium" if self.capacity == "High" else "Low"
        elif action == "Add_Memory":
            return "High" if self.capacity == "Medium" else "Medium"
        elif action == "Remove_Memory":
            return "Low" if self.capacity == "High" else "Medium"
        elif action in ["Corrective_Maintenance", "Preventive_Maintenance", "Restart_Components"]:
            return "Medium"
        elif action == "Update_Version":
            return "High"
        elif action == "Rollback_Version":
            return "Low"
        return self.capacity

    def get_next_second_likely_state(self, action):
        if action in ["Decrease_CPU", "Decrease_CPU_Slightly"]:
            return "High" if self.capacity == "Low" else "Medium"
        elif action in ["Add_Memory"]:
            return "Low" if self.capacity == "High" else "Medium"
        elif action in ["Remove_Memory"]:
            return "High" if self.capacity == "Low" else "Medium"
        elif action in ["Corrective_Maintenance", "Preventive_Maintenance", "Restart_Components"]:
            return "Low"
        elif action == "Update_Version":
            return "Low"
        elif action == "Rollback_Version":
            return "Medium"
        return self.capacity


class Maintenance(Transitions):
    def __init__(self, speed, capacity, health):
        self.speed = speed
        self.capacity = capacity
        self.health = health

    def get_next_most_likely_state(self, action):
        if action == "Corrective_Maintenance":
            return "Medium" if self.speed == "Slow" else self.speed, "Medium" if self.capacity == "Low" else self.capacity, "Healthy"
        elif action == "Preventive_Maintenance":
            return "Medium" if self.speed == "Fast" else self.speed, "Medium" if self.capacity == "High" else self.capacity, self.health
        elif action == "Restart_Components":
            return "Slow", "Low", "Healthy"
        return self.speed, self.capacity, self.health

    def get_next_second_likely_state(self, action):
        if action == "Corrective_Maintenance":
            return "Slow" if self.speed == "Medium" else self.speed, "Low" if self.capacity == "Medium" else self.capacity, "Error"
        elif action == "Preventive_Maintenance":
            return "Slow" if self.speed == "Fast" else self.speed, "Low" if self.capacity == "High" else self.capacity, "Error"
        elif action == "Restart_Components":
            return "Slow", "Low", "Error"
        return self.speed, self.capacity, self.health