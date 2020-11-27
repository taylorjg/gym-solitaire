import gym
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from gym_solitaire.envs import observation_valid_actions


def test_observation_space():
    env = gym.make('gym_solitaire:gym_solitaire-v0')
    assert env.observation_space == gym.spaces.Box(0, 1, (33,))


def test_action_space():
    env = gym.make('gym_solitaire:gym_solitaire-v0')
    assert env.action_space == gym.spaces.Discrete(76)


def test_reset():
    env = gym.make('gym_solitaire:gym_solitaire-v0')
    obs = env.reset()
    expected = np.array([1] * 16 + [0] + [1] * 16)
    assert_array_equal(obs, expected)


def test_invalid_action():
    env = gym.make('gym_solitaire:gym_solitaire-v0')
    initial_obs = env.reset()
    obs, reward, done, _ = env.step(0)
    assert_array_equal(obs, initial_obs)
    assert reward == -100
    assert not done


def test_bad_action_index():
    env = gym.make('gym_solitaire:gym_solitaire-v0')
    with pytest.raises(AssertionError):
        env.step(999)


def test_solution():
    env = gym.make('gym_solitaire:gym_solitaire-v0')
    obs = env.reset()
    solution_actions = [
        68, 49, 71, 33, 75, 71, 5, 11,
        20, 46, 11, 27, 3, 40, 1, 3,
        69, 65, 57, 28, 65, 20, 12, 49,
        57, 62, 27, 39, 7, 35, 44
    ]
    for action in solution_actions:
        obs, reward, done, _ = env.step(action)
    expected = np.array([0] * 16 + [1] + [0] * 16)
    assert_array_equal(obs, expected)


def test_render_unsupported_mode():
    env = gym.make('gym_solitaire:gym_solitaire-v0')
    with pytest.raises(NotImplementedError):
        env.render('bogus')


def test_observation_valid_actions():
    env = gym.make('gym_solitaire:gym_solitaire-v0')
    obs = env.reset()
    valid_actions = observation_valid_actions(obs)
    assert len(valid_actions) == 4
