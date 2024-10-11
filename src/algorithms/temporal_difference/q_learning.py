import numpy as np

def epsilon_greedy(Q, state, nA, epsilon):
    """
    Escolhe uma ação usando a política epsilon-greedy.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(nA)
    else:
        return np.argmax(Q[state])

def q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.99):
    """
    Algoritmo de Q-learning.

    Args:
        env: Ambiente (APIEnv).
        num_episodes: Número de episódios de treinamento.
        alpha: Taxa de aprendizado.
        gamma: Fator de desconto.
        epsilon: Probabilidade inicial de exploração para política epsilon-greedy.
        epsilon_decay: Fator de decaimento para epsilon em cada episódio.

    Returns:
        Q: A função valor-ação aprendida.
        policy: A política derivada da função Q aprendida.
        total_rewards: Lista com as recompensas totais de cada episódio.
    """
    Q = np.zeros((env.state_space, env.action_space.n))  # Inicializa a função Q
    total_rewards = []  # Lista para armazenar as recompensas acumuladas em cada episódio

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0  # Inicializa a recompensa do episódio

        while not done:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            best_next_action = np.argmax(Q[next_state])

            # Atualiza a função Q usando a fórmula de Q-learning
            Q[state, action] += alpha * (
                reward + gamma * Q[next_state, best_next_action] - Q[state, action]
            )

            state = next_state
            episode_reward += reward  # Acumula a recompensa do episódio

        total_rewards.append(episode_reward)  # Armazena a recompensa total do episódio

        # Reduz epsilon (exploração) ao longo do tempo
        epsilon *= epsilon_decay

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} completed. Total reward: {episode_reward}")

    # Deriva a política da função Q aprendida
    policy = np.zeros([env.state_space, env.action_space.n])
    for s in range(env.state_space):
        best_action = np.argmax(Q[s])
        policy[s, best_action] = 1.0

    return Q, policy, total_rewards
