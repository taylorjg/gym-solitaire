from collections import namedtuple
from gym import Env, spaces
import numpy as np

Location = namedtuple('Location', ['row', 'col'])
Action = namedtuple('Action', ['from_location', 'via_location', 'to_location'])

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

DIRECTIONS = [UP, DOWN, LEFT, RIGHT]


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


def iter_locations():
    board_shape = [
        '  XXX  ',
        '  XXX  ',
        'XXXXXXX',
        'XXXXXXX',
        'XXXXXXX',
        '  XXX  ',
        '  XXX  ',
    ]
    num_rows = len(board_shape)
    num_cols = len(board_shape[0])
    for row in range(num_rows):
        for col in range(num_cols):
            if board_shape[row][col] == 'X':
                yield Location(row, col)


def iter_actions():
    for from_location in LOCATIONS:
        for direction in DIRECTIONS:
            via_location = follow_direction(from_location, direction)
            to_location = follow_direction(via_location, direction)
            if via_location in LOCATIONS and to_location in LOCATIONS:
                action = Action(from_location, via_location, to_location)
                yield action


def create_board():
    board = {}
    reset_board(board)
    return board


def reset_board(board):
    for location in LOCATIONS:
        board[location] = location != CENTRE


def valid_action_indices(board):
    action_indices = []
    for action_index, action in enumerate(ACTIONS):
        if is_valid_action(board, action):
            action_indices.append(action_index)
    return action_indices


def is_valid_action(board, action):
    from_location, via_location, to_location = action
    assert from_location in LOCATIONS
    assert via_location in LOCATIONS
    assert to_location in LOCATIONS
    return all([
        board[from_location],
        board[via_location],
        not board[to_location]
    ])


def make_move(board, action):
    from_location, via_location, to_location = action
    assert from_location in LOCATIONS
    assert via_location in LOCATIONS
    assert to_location in LOCATIONS
    assert board[from_location]
    assert board[via_location]
    assert not board[to_location]
    board[from_location] = False
    board[via_location] = False
    board[to_location] = True


def obs_to_board(obs):
    board = create_board()
    assert len(obs) == len(board)
    for index, location in enumerate(LOCATIONS):
        board[location] = bool(obs[index])
    return board


def board_to_obs(board):
    values = []
    for location in LOCATIONS:
        values.append(board[location])
    return np.array(values, dtype=np.float32)


def observation_valid_actions(obs):
    board = obs_to_board(obs)
    return valid_action_indices(board)


CENTRE = Location(3, 3)
LOCATIONS = list(iter_locations())
ACTIONS = list(iter_actions())
EMPTY_INFO = {}


class SolitaireEnv(Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-100, 0)

    def __init__(self):
        self._board = create_board()
        self.observation_space = spaces.Box(0, 1, (len(self._board),), np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))

    def seed(self, seed=None):
        return [seed]

    def reset(self):
        reset_board(self._board)
        return board_to_obs(self._board)

    def step(self, action_index):
        obs = board_to_obs(self._board)
        done = not valid_action_indices(self._board)
        if done:
            return obs, 0, True, EMPTY_INFO
        assert 0 <= action_index < len(ACTIONS)
        action = ACTIONS[action_index]
        if not is_valid_action(self._board, action):
            return obs, -100, False, EMPTY_INFO
        make_move(self._board, action)
        obs = board_to_obs(self._board)
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
