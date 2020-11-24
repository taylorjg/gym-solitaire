from collections import namedtuple
from gym import Env, spaces
import numpy as np

Location = namedtuple('Location', ['row', 'col'])
Action = namedtuple('Action', ['from_location', 'via_location', 'to_location'])

CENTRE = Location(3, 3)

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

EMPTY_INFO = {}


def create_board():
    col_ranges = [
        range(2, 5),
        range(2, 5),
        range(7),
        range(7),
        range(7),
        range(2, 5),
        range(2, 5)
    ]
    board = {}
    for row, col_range in enumerate(col_ranges):
        for col in col_range:
            location = Location(row, col)
            board[location] = True
    board[CENTRE] = False
    return board


def follow_direction(location, direction):
    assert direction in DIRECTIONS
    row, col = location
    if direction == LEFT:
        return Location(row, col - 1)
    if direction == RIGHT:
        return Location(row, col + 1)
    if direction == UP:
        return Location(row - 1, col)
    if direction == DOWN:
        return Location(row + 1, col)


def make_all_actions(board):
    actions = []
    for from_location in board.keys():
        for direction in DIRECTIONS:
            via_location = follow_direction(from_location, direction)
            to_location = follow_direction(via_location, direction)
            if via_location in board and to_location in board:
                action = Action(from_location, via_location, to_location)
                actions.append(action)
    return actions


def valid_action_indices(board):
    action_indices = []
    for action_index, action in enumerate(ALL_ACTIONS):
        if is_valid_action(board, action):
            action_indices.append(action_index)
    return action_indices


def is_valid_action(board, action):
    from_location, via_location, to_location = action
    return all([
        from_location in board,
        via_location in board,
        to_location in board,
        board[from_location],
        board[via_location],
        not board[to_location]
    ])


def observation_valid_actions(obs):
    board = create_board()
    assert len(obs) == len(board)
    for index, location in enumerate(board.keys()):
        board[location] = bool(obs[index])
    return valid_action_indices(board)


ALL_ACTIONS = make_all_actions(create_board())


class SolitaireEnv(Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-100, 0)

    def __init__(self):
        self._board = create_board()
        self.observation_space = spaces.Box(0, 1, (len(self._board),), np.float32)
        self.action_space = spaces.Discrete(len(ALL_ACTIONS))

    def seed(self, seed=None):
        return [seed]

    def reset(self):
        for location in self._board.keys():
            self._board[location] = True
        self._board[CENTRE] = False
        return self._make_observation()

    def step(self, action_index):
        obs = self._make_observation()
        done = not valid_action_indices(self._board)
        if done:
            return obs, 0, True, EMPTY_INFO
        assert 0 <= action_index < len(ALL_ACTIONS)
        action = ALL_ACTIONS[action_index]
        if not is_valid_action(self._board, action):
            return obs, -100, False, EMPTY_INFO
        self._make_move(action)
        obs = self._make_observation()
        done = not valid_action_indices(self._board)
        reward = self._calculate_final_reward() if done else 0
        return obs, reward, done, EMPTY_INFO

    def render(self, mode='human'):
        if mode != 'human':
            super().render(mode=mode)
        for row in range(7):
            line = ''
            for col in range(7):
                location = row, col
                if location in self._board:
                    line += 'X' if self._board[location] else '.'
                else:
                    line += ' '
            print(line)
        print()

    def _calculate_final_reward(self):
        reward = 0
        for location, occupied in self._board.items():
            if occupied:
                row, col = location
                row_diff = abs(row - CENTRE.row)
                col_diff = abs(col - CENTRE.col)
                manhattan_distance_from_centre = row_diff + col_diff
                reward -= manhattan_distance_from_centre
        return reward

    def _make_observation(self):
        values = list(self._board.values())
        return np.array(values, dtype=np.float32)

    def _make_move(self, action):
        from_location, via_location, to_location = action
        assert from_location in self._board
        assert via_location in self._board
        assert to_location in self._board
        assert self._board[from_location]
        assert self._board[via_location]
        assert not self._board[to_location]
        self._board[from_location] = False
        self._board[via_location] = False
        self._board[to_location] = True

    # @staticmethod
    # def valid_actions(obs):
    #     board = create_board()
    #     for index, location in enumerate(board.keys()):
    #         board[location] = bool(obs[index])
    #     return valid_action_indices(board)
