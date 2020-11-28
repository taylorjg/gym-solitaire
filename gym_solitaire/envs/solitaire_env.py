from collections import namedtuple
from gym import Env, spaces
import numpy as np
import copy

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


class Board:
    def __init__(self, board=None):
        if board is not None:
            self._board = copy.deepcopy(board)
        else:
            self._board = {}
            self.reset()

    def reset(self):
        new_board = copy.deepcopy(self._board)
        for location in LOCATIONS:
            new_board[location] = location != CENTRE
        return Board(new_board)

    @property
    def done(self):
        return not self.valid_action_indices()

    def valid_action_indices(self):
        action_indices = []
        for action_index, action in enumerate(ACTIONS):
            if self.is_valid_action(action):
                action_indices.append(action_index)
        return action_indices

    def is_valid_action(self, action):
        from_location, via_location, to_location = action
        assert from_location in LOCATIONS
        assert via_location in LOCATIONS
        assert to_location in LOCATIONS
        return all([
            self._board[from_location],
            self._board[via_location],
            not self._board[to_location]
        ])

    def make_move(self, action):
        from_location, via_location, to_location = action
        assert from_location in LOCATIONS
        assert via_location in LOCATIONS
        assert to_location in LOCATIONS
        new_board = copy.deepcopy(self._board)
        assert new_board[from_location]
        assert new_board[via_location]
        assert not new_board[to_location]
        new_board[from_location] = False
        new_board[via_location] = False
        new_board[to_location] = True
        return Board(new_board)

    def __getitem__(self, location):
        return self._board[location]

    def __iter__(self):
        for item in self._board.items():
            yield item

    def __len__(self):
        return len(self._board)


def obs_to_board(obs):
    assert len(obs) == len(LOCATIONS)
    board = {location: bool(obs[index]) for index, location in enumerate(LOCATIONS)}
    return Board(board)


def board_to_obs(board):
    values = []
    for location in LOCATIONS:
        values.append(board[location])
    return np.array(values, dtype=np.float32)


CENTRE = Location(3, 3)
LOCATIONS = list(iter_locations())
ACTIONS = list(iter_actions())
EMPTY_INFO = {}


class SolitaireEnv(Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-100, +100)

    def __init__(self):
        self._board = Board()
        self.observation_space = spaces.Box(0, 1, (len(LOCATIONS),), np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))

    def seed(self, seed=None):
        return [seed]

    def reset(self):
        self._board = self._board.reset()
        return board_to_obs(self._board)

    def step(self, action_index):
        obs = board_to_obs(self._board)
        done = self._board.done
        if done:
            return obs, 0, True, EMPTY_INFO
        assert 0 <= action_index < len(ACTIONS)
        action = ACTIONS[action_index]
        if not self._board.is_valid_action(action):
            return obs, -100, False, EMPTY_INFO
        self._board = self._board.make_move(action)
        obs = board_to_obs(self._board)
        done = self._board.done
        reward = self._calculate_final_reward() if done else 0
        return obs, reward, done, EMPTY_INFO

    def render(self, mode='human'):
        if mode != 'human':
            super().render(mode=mode)
        for row in range(7):
            line = ''
            for col in range(7):
                location = row, col
                if location in LOCATIONS:
                    line += 'X' if self._board[location] else '.'
                else:
                    line += ' '
            print(line)
        print()

    def _calculate_final_reward(self):
        reward = 0
        for location, occupied in self._board:
            if occupied:
                row, col = location
                row_diff = abs(row - CENTRE.row)
                col_diff = abs(col - CENTRE.col)
                manhattan_distance_from_centre = row_diff + col_diff
                reward -= manhattan_distance_from_centre
        return 100 if reward == 0 else reward
