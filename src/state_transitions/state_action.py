import random

from src.state_transitions.transitions import Transitions


class Availability(Transitions):
    def __init__(self, avail, speed, health, capacity):
        self.avail = avail
        self.speed = speed
        self.health = health
        self.capacity = capacity

    def get_next_most_likely_state(self, action):
        if action in ["Increase_CPU", "Increase_CPU_Slightly"]:
            self.avail = "Available"
        elif action in ["Decrease_CPU", "Decrease_CPU_Slightly"]:
            self.avail = (
                "Offline" if self.health in ["Error", "Overloaded"] else "Available"
            )
        elif action in [
            "Corrective_Maintenance",
            "Preventive_Maintenance",
            "Restart_Components",
        ]:
            self.avail = "Available"
        elif action == "Update_Version":
            if random.random() > 0.5:
                self.avail = "Offline"
                self.health = "Error"
            else:
                self.avail = "Available"
                self.health = "Healthy"
        elif action == "Rollback_Version":
            self.avail = "Available"
            self.health = "Healthy"
        elif action in ["Add_Memory", "Remove_Memory"]:
            self.avail = "Available" if self.health != "Error" else "Offline"

        return self.avail, self.speed, self.health, self.capacity

    def get_next_second_likely_state(self, action):
        if action in ["Increase_CPU", "Increase_CPU_Slightly"]:
            self.avail = "Available" if self.health != "Error" else "Offline"
        elif action in ["Decrease_CPU", "Decrease_CPU_Slightly"]:
            if self.health in ["Error", "Overloaded"]:
                self.avail = (
                    "Available"  # Efeito contrário: manter disponível mesmo com erro
                )
            else:
                self.avail = "Offline"
        elif action in [
            "Corrective_Maintenance",
            "Preventive_Maintenance",
            "Restart_Components",
        ]:
            self.avail = "Offline"  # Efeito contrário: sistema cai após manutenção
        elif action == "Update_Version":
            if random.random() > 0.5:
                self.avail = "Available"
            else:
                self.avail = "Offline"
        elif action == "Rollback_Version":
            self.avail = "Offline"  # Efeito contrário: rollback deixa o sistema offline
        elif action in ["Add_Memory", "Remove_Memory"]:
            self.avail = "Offline" if self.health == "Error" else "Available"

        return self.avail, self.speed, self.health, self.capacity


class Speed(Transitions):
    def __init__(self, avail, speed, health, capacity):
        self.avail = avail
        self.speed = speed
        self.health = health
        self.capacity = capacity

    def get_next_most_likely_state(self, action):
        if action == "Increase_CPU":
            if self.speed in ["Slow", "Medium"]:
                self.speed = "Fast"
        elif action == "Increase_CPU_Slightly":
            if self.speed == "Slow":
                self.speed = "Medium"
        elif action == "Decrease_CPU":
            if self.speed == "Fast":
                self.speed = "Medium"
            elif self.speed == "Medium":
                self.speed = "Slow"
        elif action == "Decrease_CPU_Slightly":
            if self.speed == "Fast":
                self.speed = "Medium"

        return self.avail, self.speed, self.health, self.capacity

    def get_next_second_likely_state(self, action):
        if action == "Increase_CPU":
            if self.speed == "Fast":
                self.speed = "Medium"  # Efeito contrário: velocidade diminui
            elif self.speed == "Medium":
                self.speed = "Slow"
        elif action == "Increase_CPU_Slightly":
            if self.speed == "Medium":
                self.speed = "Slow"
        elif action == "Decrease_CPU":
            if self.speed == "Slow":
                self.speed = (
                    "Fast"  # Efeito contrário: aumento de velocidade após diminuir CPU
                )
            elif self.speed == "Medium":
                self.speed = "Fast"
        elif action == "Decrease_CPU_Slightly":
            if self.speed == "Slow":
                self.speed = "Fast"

        return self.avail, self.speed, self.health, self.capacity


class Health(Transitions):
    def __init__(self, avail, speed, health, capacity):
        self.avail = avail
        self.speed = speed
        self.health = health
        self.capacity = capacity

    def get_next_most_likely_state(self, action):
        if action == "Corrective_Maintenance":
            self.health = "Healthy"
        elif action == "Preventive_Maintenance":
            if self.health == "Overloaded":
                self.health = "Healthy"
        elif action == "Restart_Components":
            self.health = "Healthy"
        elif action == "Update_Version":
            self.health = "Error" if random.random() > 0.5 else "Healthy"
        elif action == "Rollback_Version":
            self.health = "Healthy"

        return self.avail, self.speed, self.health, self.capacity

    def get_next_second_likely_state(self, action):
        if action == "Corrective_Maintenance":
            self.health = (
                "Overloaded" if self.health == "Healthy" else "Error"
            )  # Correção faz o sistema falhar
        elif action == "Preventive_Maintenance":
            if self.health == "Healthy":
                self.health = "Overloaded"  # Prevenção sobrecarrega o sistema
        elif action == "Restart_Components":
            self.health = "Error"  # Efeito contrário: reiniciar piora o estado de saúde
        elif action == "Update_Version":
            self.health = (
                "Error" if random.random() > 0.5 else "Overloaded"
            )  # Versão causa sobrecarga ou erro
        elif action == "Rollback_Version":
            self.health = "Error"

        return self.avail, self.speed, self.health, self.capacity


class Capacity(Transitions):
    def __init__(self, avail, speed, health, capacity):
        self.avail = avail
        self.speed = speed
        self.health = health
        self.capacity = capacity

    def get_next_most_likely_state(self, action):
        if action in ["Increase_CPU", "Increase_CPU_Slightly"]:
            if self.capacity == "Low":
                self.capacity = "Medium"
            elif self.capacity == "Medium":
                self.capacity = "High"
        elif action == "Decrease_CPU":
            if self.capacity == "High":
                self.capacity = "Medium"
            elif self.capacity == "Medium":
                self.capacity = "Low"
        elif action == "Decrease_CPU_Slightly":
            if self.capacity == "High":
                self.capacity = "Medium"
            elif self.capacity == "Medium":
                self.capacity = "Low"
        return self.avail, self.speed, self.health, self.capacity

    def get_next_second_likely_state(self, action):
        if action in ["Increase_CPU", "Increase_CPU_Slightly"]:
            if self.capacity == "High":
                self.capacity = (
                    "Low"  # Efeito contrário: redução da capacidade ao aumentar CPU
                )
            elif self.capacity == "Medium":
                self.capacity = "Low"
        elif action == "Decrease_CPU":
            if self.capacity == "Low":
                self.capacity = (
                    "High"  # Efeito contrário: aumento da capacidade ao diminuir CPU
                )
            elif self.capacity == "Medium":
                self.capacity = "High"
        elif action == "Decrease_CPU_Slightly":
            if self.capacity == "Low":
                self.capacity = "Medium"
        return self.avail, self.speed, self.health, self.capacity


class Maintenance(Transitions):
    def __init__(self, avail, speed, health, capacity):
        self.avail = avail
        self.speed = speed
        self.health = health
        self.capacity = capacity

    def get_next_most_likely_state(self, action):
        # Para 'Corrective_Maintenance'
        if action == "Corrective_Maintenance":
            if self.speed == "Slow":
                self.speed = "Medium"  # Correção pode melhorar a velocidade
            if self.capacity == "Low":
                self.capacity = (
                    "Medium"  # Melhora de capacidade com manutenção corretiva
                )
            self.avail = "Available"  # Disponibilidade garantida após manutenção

        # Para 'Preventive_Maintenance'
        elif action == "Preventive_Maintenance":
            if self.speed == "Fast":
                self.speed = "Medium"  # Manutenção preventiva pode desacelerar um pouco o sistema
            if self.capacity == "High":
                self.capacity = (
                    "Medium"  # Prevenção pode reduzir a capacidade momentaneamente
                )
            self.avail = "Available"  # Sistema mantido disponível

        # Para 'Restart_Components'
        elif action == "Restart_Components":
            self.speed = (
                "Medium"  # O reinício tende a estabilizar o sistema em 'Medium'
            )
            self.capacity = "Medium"  # Capacidade estabilizada após reiniciar
            self.health = "Healthy"  # Reinício corrige problemas
            self.avail = "Available"  # Disponibilidade garantida após reinício

        return self.avail, self.speed, self.health, self.capacity

    def get_next_second_likely_state(self, action):
        # Para 'Corrective_Maintenance'
        if action == "Corrective_Maintenance":
            if self.speed == "Medium":
                self.speed = "Slow"  # Efeito contrário: manutenção reduz a velocidade
            if self.capacity == "Medium":
                self.capacity = "Low"  # Capacidade diminui com a manutenção corretiva
            self.avail = "Offline"  # Disponibilidade cai após a manutenção

        # Para 'Preventive_Maintenance'
        elif action == "Preventive_Maintenance":
            if self.speed == "Fast":
                self.speed = "Slow"  # Manutenção preventiva reduz muito a velocidade
            if self.capacity == "High":
                self.capacity = "Low"  # Capacidade cai drasticamente
            self.avail = "Offline"  # Sistema cai preventivamente

        # Para 'Restart_Components'
        elif action == "Restart_Components":
            self.speed = "Slow"  # Reinício causa lentidão
            self.capacity = "Low"  # Capacidade reduzida após reinício
            self.health = "Error"  # Reinício piora o estado de saúde
            self.avail = "Offline"  # Sistema cai após reinício

        return self.avail, self.speed, self.health, self.capacity
