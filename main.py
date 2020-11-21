import gym

env = gym.make('gym_solitaire:gym_solitaire-v0')
print(f"env.observation_space: {env.observation_space}")
print(f"env.action_space: {env.action_space}")
obs = env.reset()
print(f"initial obs: {obs}")
solution = [68, 49, 71, 33, 75, 71, 5, 11, 20, 46, 11, 27, 3, 40, 1, 3, 69, 65, 57, 28, 65, 20, 12, 49, 57, 62, 27, 39, 7, 35, 44]
for action in solution:
    obs, reward, done, _ = env.step(action)
    print(f"obs: {obs}; reward: {reward}; done: {done}")
