import matplotlib.pyplot as plt
import numpy as np


def run_policy(env, policy, num_episodes=100, num_steps=100):
    """
    Executa a política por 'num_episodes' e coleta histórico, recompensas e totais.
    """
    episode_rewards = []
    total_rewards = []
    histories = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        history = []
        rewards = []
        total_reward = 0

        print(f"Episode {episode+1}/{num_episodes}")

        for step in range(num_steps):
            action = np.argmax(policy[state])
            next_state, reward, done, truncated, _ = env.step(action)
            env.render()

            # Exibe informações sobre o estado e a recompensa
            print(
                f"Step {step+1}: State={state}, Action={action}, Reward={reward}, Next State={next_state}"
            )

            # Armazena o histórico e recompensa
            history.append((state, action, next_state, reward))
            rewards.append(reward)
            total_reward += reward

            state = next_state
            if done or truncated:
                break

        histories.append(history)
        episode_rewards.append(rewards)
        total_rewards.append(total_reward)

    return histories, episode_rewards, total_rewards


def plot_action_state_history(history, rewards, env):
    """
    Função para visualizar as ações, estados e recompensas ao longo do tempo em um episódio.
    """
    steps = np.arange(len(history))
    state_names = [env.states[state] for (state, action, next_state, reward) in history]
    next_state_names = [
        env.states[next_state] for (state, action, next_state, reward) in history
    ]
    action_names = [
        env.actions[action] for (state, action, next_state, reward) in history
    ]

    # Gráfico de linhas para representar a sequência de estados e ações
    fig, ax = plt.subplots(figsize=(30, 15))

    # Definindo cores para os estados finais
    colors = {
        "Available_Fast_Healthy_Low": "green",  # Melhor estado, baixo uso e saudável
        "Available_Fast_Healthy_Medium": "limegreen",  # Saudável com carga média
        "Available_Fast_Healthy_High": "darkgreen",  # Saudável com alta carga
        "Available_Medium_Healthy_Low": "lightgreen",  # Médio saudável com baixa carga
        "Available_Medium_Healthy_Medium": "yellowgreen",  # Médio saudável com carga média
        "Available_Medium_Healthy_High": "yellow",  # Médio saudável com alta carga
        "Available_Slow_Healthy_Low": "blue",  # Saudável, mas lento
        "Available_Slow_Healthy_High": "orange",  # Saudável, mas muito lento
        "Available_Fast_Error_Low": "lightcoral",  # Erros leves, rápido
        "Available_Fast_Error_Medium": "red",  # Erros médios
        "Available_Fast_Error_High": "darkred",  # Erros graves
        "Available_Medium_Error_Low": "salmon",  # Erros leves, média
        "Available_Medium_Error_High": "brown",  # Erros graves, média
        "Available_Slow_Error_Low": "saddlebrown",  # Lento com erros leves
        "Available_Slow_Error_High": "maroon",  # Lento com erros graves
        "Available_Fast_Overloaded_Low": "gold",  # Sobrecarregado com erros
        "Available_Fast_Overloaded_High": "orange",  # Sobrecarregado com alta carga e erros
        "Offline_Slow_Error_Medium": "crimson",  # Offline, com erros moderados
        "Offline_Slow_Overloaded_Low": "darkslategray",  # Offline e sobrecarregado
        "Offline_Slow_Healthy_Medium": "gray",  # Offline, mas saudável
    }

    # Adicionando estados e ações no gráfico
    for i, (state_name, action_name, next_state_name, reward) in enumerate(
        zip(state_names, action_names, next_state_names, rewards)
    ):
        ax.plot(
            [i, i + 1],
            [env.states.index(state_name), env.states.index(next_state_name)],
            color=colors.get(next_state_name, "blue"),
            label=f"{action_name} → {next_state_name}" if i == 0 else "",
        )

        ax.text(
            i + 0.5,
            (env.states.index(state_name) + env.states.index(next_state_name)) / 2,
            f"{action_name}\nReward: {reward:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(steps)
    ax.set_xticklabels(steps)
    ax.set_yticks(np.arange(len(env.states)))
    ax.set_yticklabels(env.states)

    ax.set_xlabel("Steps")
    ax.set_ylabel("States")
    ax.set_title("Action and State Transitions with Rewards")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_total_rewards(ep_rewards, total_rewards):
    """
    Plota a curva de recompensa acumulada ao longo dos episódios e printa o somatório das recompensas.
    """
    print(f"Somatório das recompensas: {total_rewards}")

    # Plotar o gráfico
    plt.figure(figsize=(30, 15))
    plt.plot(ep_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards per Episode")
    plt.show()
