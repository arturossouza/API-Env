import itertools
from collections import defaultdict
import numpy as np

def epsilon_greedy(q_values, epsilon: float, num_actions: int):
    """
    Cria uma política epsilon-greedy com base nos valores Q e epsilon fornecidos.

    Argumentos:
        q_values: Um dicionário que mapeia estados para valores de ação.
           Cada valor é um array numpy de comprimento num_actions.
        epsilon: Probabilidade de selecionar uma ação aleatória, float entre 0 e 1.
        num_actions: Número de ações no ambiente.

    Retorno:
        Uma função que recebe o estado como entrada e retorna as probabilidades para cada ação
        como um array numpy de comprimento num_actions.
    """

    def epsilon_greedy_policy(observation):
        # Inicializa todas as probabilidades de ação com epsilon / num_actions
        action_probabilities = np.ones(num_actions, dtype=float) * epsilon / num_actions

        # Seleciona a melhor ação com base em q_values[estado]
        best_action = np.argmax(q_values[observation])
        action_probabilities[best_action] += 1.0 - epsilon
        return action_probabilities

    return epsilon_greedy_policy

def sarsa_learning(
    env,
    num_episodes: int,
    gamma: float = 1.0,  # Melhor usar 1.0 como valor padrão
    alpha: float = 0.5,
    epsilon: float = 0.1,
):
    """
    Algoritmo SARSA: Aprendizado de Diferença Temporal On-policy. Encontra a política epsilon-greedy ótima.

    Argumentos:
        env: Ambiente OpenAI Gym.
        num_episodes: Número de episódios a serem executados.
        gamma: Fator de desconto para recompensas futuras (padrão: 1.0).
        alpha: Taxa de aprendizado para a atualização TD (padrão: 0.5).
        epsilon: Probabilidade de escolher uma ação aleatória. Float entre 0 e 1 (padrão: 0.1).

    Retorno:
        q_values: A função de valor de ação ótima, um dicionário que mapeia estado -> valores de ação.
        policy: A política determinística final, derivada da função Q.
        total_rewards: Lista contendo a recompensa total acumulada por episódio.
    """

    # Função de valor-ação final
    q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    # A política epsilon-greedy a ser seguida
    policy = epsilon_greedy(q_values, epsilon, env.action_space.n)

    total_rewards = []  # Lista para armazenar a recompensa total por episódio

    for episode in range(num_episodes):
        # Reinicia o ambiente e escolhe a primeira ação
        state, _ = env.reset()
        action_probabilities = policy(state)
        action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)

        episode_reward = 0  # Acumula a recompensa do episódio

        # Executa um passo no ambiente
        for step in itertools.count():
            # Realiza a ação e obtém o próximo estado e recompensa
            next_state, reward, done, truncated, _ = env.step(action)

            # Acumula a recompensa recebida
            episode_reward += reward

            # Escolhe a próxima ação com base na política epsilon-greedy
            next_action_probabilities = policy(next_state)
            next_action = np.random.choice(
                np.arange(len(next_action_probabilities)), p=next_action_probabilities
            )

            # Atualização TD
            temporal_difference_target = reward + gamma * q_values[next_state][next_action]
            temporal_difference_delta = temporal_difference_target - q_values[state][action]
            q_values[state][action] += alpha * temporal_difference_delta

            # Se o episódio terminar, sair do loop
            if done:
                break

            # Atualiza o estado e a ação para o próximo passo
            action = next_action
            state = next_state

        # Armazena a recompensa total do episódio
        total_rewards.append(episode_reward)

        # Exibe o progresso a cada 100 episódios
        if episode % 100 == 0:
            print(f"Episódio {episode}/{num_episodes} concluído. Total reward: {episode_reward}")

    # Gera a política final determinística (greedy)
    policy = {}
    for state in q_values:
        best_action = np.argmax(q_values[state])
        policy[state] = np.eye(env.action_space.n)[best_action]

    return q_values, policy, total_rewards
