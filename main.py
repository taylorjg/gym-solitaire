import gym
from gym_solitaire.envs import obs_to_board


def main():
    env = gym.make('gym_solitaire:gym_solitaire-v0')

    print(f"env.observation_space: {env.observation_space}")
    print(f"env.action_space: {env.action_space}")
    print(f"env.spec.max_episode_steps: {env.spec.max_episode_steps}")
    print(f"env.metadata: {env.metadata}")
    print(f"env.reward_range: {env.reward_range}")

    obs = env.reset()
    print(f"initial obs: {obs}")
    print(f"valid actions: {obs_to_board(obs).valid_action_indices()}")
    env.render('human')

    solution_actions = [
        68, 49, 71, 33, 75, 71, 5, 11,
        20, 46, 11, 27, 3, 40, 1, 3,
        69, 65, 57, 28, 65, 20, 12, 49,
        57, 62, 27, 39, 7, 35, 44
    ]

    for action in solution_actions:
        obs, reward, done, info = env.step(action)
        print(f"obs: {obs}; reward: {reward}; done: {done}; info: {info}")
        print(f"valid actions: {obs_to_board(obs).valid_action_indices()}")
        env.render('human')


if __name__ == "__main__":
    main()
