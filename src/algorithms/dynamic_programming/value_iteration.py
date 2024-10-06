import numpy as np


def value_iteration(env, theta=0.000001, discount_factor=0.9):
    """
    Value Iteration Algorithm adapted for custom environment with probabilistic transitions.

    Args:
        env: Custom environment with defined states and actions.
        theta: Stop evaluation once value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all actions in a given state.
        """
        A = np.zeros(env.action_space.n)
        state_str = env.states[state]

        for a in range(env.action_space.n):
            action_str = env.actions[a]
            transitions = env.transition_probabilities.get(
                (state_str, action_str), [(state_str, 1.0)]
            )

            for next_state_str, prob in transitions:
                next_state = env.states.index(next_state_str)
                reward = env.states_rewards.get(next_state_str, 0)
                penalty = env.action_rewards.get(action_str, 0)
                A[a] += prob * (reward + penalty + discount_factor * V[next_state])

        return A

    # Inicializa a função de valor para todos os estados
    V = np.zeros(env.state_space)
    episode = 0

    while True:
        episode += 1
        delta = 0.0
        for s in range(env.state_space):
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value

        if episode % 10 == 0:
            print(f"Episódio {episode}: delta = {delta}")

        if delta < theta:
            break

    # Derive the policy from the optimal value function
    policy = np.zeros([env.state_space, env.action_space.n])
    for s in range(env.state_space):
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0

    return policy, V
