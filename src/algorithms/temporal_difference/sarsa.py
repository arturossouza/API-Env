import itertools
from collections import defaultdict

import numpy as np


def epsilon_greedy(Q, epsilon: float, num_actions: int):
    """
    Cria uma política epsilon-greedy com base nos valores Q e epsilon fornecidos.

    Args:
        Q: Um dicionário que mapeia estados para valores de ação.
           Cada valor é um array numpy de comprimento num_actions.
        epsilon: Probabilidade de selecionar uma ação aleatória, float entre 0 e 1.
        num_actions: Número de ações no ambiente.

    Returns:
        Uma função que recebe o estado como entrada e retorna as probabilidades para cada ação
        como um array numpy de comprimento num_actions.
    """

    def policy_fn(observation):
        A = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[observation])
        A[best_action] += 1.0 - epsilon
        return A

    return policy_fn


def sarsa_learning(
    env,
    num_episodes: int,
    gamma: float = 0.00,
    alpha: float = 0.5,
    epsilon: float = 0.1,
):
    """
    Algoritmo SARSA: Aprendizado de Diferença Temporal On-policy. Encontra a política epsilon-greedy ótima.

    Args:
        env: Ambiente OpenAI Gym.
        num_episodes: Número de episódios a serem executados.
        gamma: Fator de desconto para recompensas futuras (padrão: 1.0).
        alpha: Taxa de aprendizado para a atualização TD (padrão: 0.5).
        epsilon: Probabilidade de escolher uma ação aleatória. Float entre 0 e 1 (padrão: 0.1).

    Returns:
        Q: A função de valor de ação ótima, um dicionário que mapeia estado -> valores de ação.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    policy = epsilon_greedy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state, _ = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # One step in the environment
        for t in itertools.count():
            # Take a step
            next_state, reward, done, truncated, _ = env.step(action)

            # Pick the next action
            next_action_probs = policy(next_state)
            next_action = np.random.choice(
                np.arange(len(next_action_probs)), p=next_action_probs
            )

            # TD Update
            td_target = reward + gamma * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            action = next_action
            state = next_state
        if i_episode % 100 == 0:
            print(f"Episode {i_episode}/{num_episodes} completed.")

    policy = {}
    for state in Q:
        best_action = np.argmax(Q[state])
        policy[state] = np.eye(env.action_space.n)[best_action]

    return Q, policy
