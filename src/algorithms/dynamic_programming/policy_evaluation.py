import numpy as np

def policy_evaluation(policy, env, discount_factor=0.9, theta=0.000001):
    """
    Avalia uma política, calculando a função de valor V(s) para cada estado e as recompensas totais por episódio.

    Args:
        policy: Matriz de políticas (s, a) onde s são os estados e a são as ações.
        env: Ambiente customizado com estados e ações.
        discount_factor: Fator de desconto para a função de valor.
        theta: Critério de parada baseado na convergência de V.

    Returns:
        V: Vetor contendo a função de valor para cada estado.
        total_rewards: Lista contendo a recompensa total acumulada para cada episódio.
    """
    V = np.zeros(env.state_space)
    total_rewards = []  # Lista para armazenar as recompensas acumuladas em cada episódio

    while True:
        delta = 0
        episode_reward = 0  # Recompensa acumulada para cada episódio

        for s in range(env.state_space):
            v = 0
            state_str = env.states[s]

            # Calcula o valor esperado de cada ação para o estado s
            for a, action_prob in enumerate(policy[s]):
                action_str = env.actions[a]
                transitions = env.transition_probabilities.get(
                    (state_str, action_str), [(state_str, 1.0)]
                )

                # Calcula a soma ponderada para todos os estados de transição possíveis
                for next_state_str, prob in transitions:
                    next_state = env.states.index(next_state_str)
                    reward = env.states_rewards.get(next_state_str, 0)
                    penalty = env.action_rewards.get(action_str, 0)
                    v += (
                        action_prob
                        * prob
                        * (reward + penalty + discount_factor * V[next_state])
                    )

                    # Acumula as recompensas **com base nas probabilidades**
                    episode_reward += prob * (reward + penalty)

            delta = max(delta, np.abs(v - V[s]))
            V[s] = v

        # Armazena a recompensa total acumulada no episódio
        total_rewards.append(episode_reward)

        if delta < theta:
            break

    return V, total_rewards


def policy_improvement(env, discount_factor=0.9, theta=0.000001):
    """
    Algoritmo de Policy Improvement sem limite de iterações, baseado no critério de estabilidade da política.
    """

    # Inicializa a política como uniforme
    policy = np.ones([env.state_space, env.action_space.n]) / env.action_space.n

    iteration = 0
    total_rewards = []  # Para armazenar as recompensas acumuladas por episódio

    while True:
        V, rewards = policy_evaluation(policy, env, discount_factor, theta)
        total_rewards.append(rewards)  # Armazena todas as recompensas acumuladas de cada episódio

        policy_stable = True
        for s in range(env.state_space):
            chosen_action = np.argmax(policy[s])

            # Calcula o valor de todas as ações possíveis para o estado s
            action_values = np.zeros(env.action_space.n)
            state_str = env.states[s]
            for a in range(env.action_space.n):
                action_str = env.actions[a]
                transitions = env.transition_probabilities.get(
                    (state_str, action_str), [(state_str, 1.0)]
                )

                for next_state_str, prob in transitions:
                    next_state = env.states.index(next_state_str)
                    reward = env.states_rewards.get(next_state_str, 0)
                    penalty = env.action_rewards.get(action_str, 0)
                    action_values[a] += prob * (
                        reward + penalty + discount_factor * V[next_state]
                    )

            best_action = np.argmax(action_values)

            if chosen_action != best_action:
                policy_stable = False

            policy[s] = np.eye(env.action_space.n)[best_action]

        iteration += 1
        if policy_stable:
            print(f"Política estável após {iteration} iterações")
            break

    return policy, V, total_rewards