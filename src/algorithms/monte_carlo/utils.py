import matplotlib.pyplot as plt
import numpy as np


def run_monte_carlo_policy(env, policy, num_episodes=100, num_steps=100):
    total_rewards_per_episode = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        rewards = []
        total_reward = 0
        history = []

        for step in range(num_steps):
            action = np.argmax(policy[state])
            next_state, reward, done, _, _ = env.step(action)
            history.append((state, action, next_state, reward))  # Salvando a recompensa
            total_reward += reward
            rewards.append(reward)
            state = next_state

            if done:
                break

        total_rewards_per_episode.append(
            total_reward
        )  # Armazenando a recompensa total por episódio

    return total_rewards_per_episode, history


# Função de plotagem da recompensa total por episódio e a curva de aprendizado
def plot_total_episode_rewards(total_rewards):
    episodes = np.arange(len(total_rewards))
    fig, ax = plt.subplots()

    ax.plot(episodes, total_rewards)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Total Reward")
    ax.set_title("Total Rewards per Episode")
    plt.show()


# Função de plotagem das transições de estados e ações com recompensas
def plot_action_state_history_with_rewards(history, env):
    steps = np.arange(len(history))
    state_names = [env.states[state] for (state, action, next_state, reward) in history]
    next_state_names = [
        env.states[next_state] for (state, action, next_state, reward) in history
    ]
    action_names = [
        env.actions[action] for (state, action, next_state, reward) in history
    ]
    rewards = [reward for (state, action, next_state, reward) in history]

    fig, ax = plt.subplots(figsize=(30, 15))
    colors = {
        "Available_Fast_Healthy_Low": "green",
        "Available_Fast_Healthy_Medium": "limegreen",
        "Available_Fast_Healthy_High": "darkgreen",
        "Available_Slow_Overloaded_High": "orange",
        "Available_Slow_Healthy_High": "yellowgreen",
        "Available_Slow_Healthy_Low": "blue",
        "Available_Medium_Error_High": "brown",
        "Available_Medium_Healthy_High": "yellow",
        "Available_Medium_Healthy_Low": "lightgreen",
        "Offline_Slow_Overloaded_Low": "darkred",
        "Offline_Slow_Healthy_Medium": "gray",
        "Offline_Fast_Error_Medium": "crimson",
    }

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
            f"{action_name} (R: {reward})",
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
    ax.set_title("Action and State Transitions Over Time with Rewards")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
