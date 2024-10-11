from collections import defaultdict
import numpy as np

def epsilon_greedy_policy(Q, state, nA, epsilon):
    """
    Cria uma política epsilon-greedy baseada na função Q (estado-ação).

    Args:
        Q: Dicionário que mapeia pares de estado-ação para seus valores.
        state: Estado atual.
        nA: Número de ações disponíveis.
        epsilon: Parâmetro de exploração (probabilidade de escolher uma ação aleatória).

    Returns:
        Uma política probabilística, representada por uma lista de probabilidades de escolha de cada ação.
    """
    policy = np.ones(nA) * (epsilon / nA)
    best_action = np.argmax(Q[state])
    policy[best_action] += 1.0 - epsilon
    return policy

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control usando uma política epsilon-greedy.

    Args:
        env: Ambiente customizado (APIEnv).
        num_episodes: Número de episódios para treinar o agente.
        discount_factor: Fator de desconto para recompensas futuras.
        epsilon: Parâmetro de exploração para a política epsilon-greedy.

    Returns:
        Q: A função valor-ação otimizada após o treinamento.
        policy: A política otimizada derivada de Q.
        total_rewards_per_episode: Lista contendo a recompensa total acumulada em cada episódio.
    """
    # Função Q(s, a) armazenada como um dicionário de dicionários
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    total_rewards_per_episode = []

    for i_episode in range(1, num_episodes + 1):
        # Mostra o progresso a cada 1000 episódios
        if i_episode % 1000 == 0:
            print(f"Episode {i_episode}/{num_episodes}")

        # Gera um episódio seguindo a política epsilon-greedy
        episode = []
        state, _ = env.reset()

        done = False
        episode_reward = 0  # Inicializa a recompensa total do episódio

        while not done:
            # Seleciona uma ação usando a política epsilon-greedy
            policy = epsilon_greedy_policy(Q, state, env.action_space.n, epsilon)
            action = np.random.choice(np.arange(env.action_space.n), p=policy)

            # Executa a ação
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            episode_reward += reward  # Acumula a recompensa total

        # Armazena a recompensa total do episódio
        total_rewards_per_episode.append(episode_reward)

        # Calcula o retorno (G) para cada par estado-ação do episódio
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = discount_factor * G + reward

            # Se o par estado-ação (state, action) não ocorreu antes no episódio
            if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                # Atualiza as somas e contagens para calcular a média
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1.0
                Q[state][action] = (
                    returns_sum[(state, action)] / returns_count[(state, action)]
                )

    # Deriva a política final de Q
    policy = {}
    for state in Q:
        best_action = np.argmax(Q[state])
        policy[state] = np.eye(env.action_space.n)[best_action]

    return Q, policy, total_rewards_per_episode