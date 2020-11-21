import gym
import pytest


def test_observation_space():
    env = gym.make('gym_solitaire:gym_solitaire-v0')
    assert env.observation_space == gym.spaces.MultiBinary(33)


def test_action_space():
    env = gym.make('gym_solitaire:gym_solitaire-v0')
    assert env.action_space == gym.spaces.Discrete(76)


def test_reset():
    env = gym.make('gym_solitaire:gym_solitaire-v0')
    obs = env.reset()
    assert obs == [1] * 16 + [0] + [1] * 16


def test_invalid_action():
    env = gym.make('gym_solitaire:gym_solitaire-v0')
    initial_obs = env.reset()
    obs, reward, done, _ = env.step(0)
    assert obs == initial_obs
    assert reward == -100
    assert not done


def test_bad_action_index():
    env = gym.make('gym_solitaire:gym_solitaire-v0')
    with pytest.raises(AssertionError):
        env.step(999)


def test_render_unsupported_mode():
    env = gym.make('gym_solitaire:gym_solitaire-v0')
    with pytest.raises(NotImplementedError):
        env.render('bogus')
