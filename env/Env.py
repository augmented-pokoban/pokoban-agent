from env.mapper import *
from random import choice, shuffle
import env.api as api
from env.expert_moves import ExpertMoves


class Env:
    _actions = [
        'MOVE_NORTH',
        'MOVE_SOUTH',
        'MOVE_EAST',
        'MOVE_WEST',
        'PULL_NORTH',
        'PULL_SOUTH',
        'PULL_EAST',
        'PULL_WEST'
    ]

    _rewards = [
        -1,
        -0.1,
        1,
        10
    ]

    def __init__(self, use_server=True, game_id=None):
        self._use_server = use_server
        self._store = False
        self._map = None
        self._cur_action = None
        self._game_id = game_id
        skip = 0
        self._batch_size = 100
        self._total = 0
        if self._use_server and not game_id:
            self._maps = self._load_maps(skip, self._batch_size)
            shuffle(self._maps)

        elif not self._use_server:
            self._maps = self._load_maps(skip, self._batch_size)

    def reset(self, store=False, level=None):

        self.terminate()

        if self._use_server:
            map_choice = self._pop() if level is None else level
            self._game_id, initial = api.init(map_choice)

        else:
            # Skip moves without a solution since they have no transitions
            while True:
                map_id = self._pop()
                self._map = ExpertMoves(api.get_expert_game(map_id))
                if any(self._map.transitions):
                    break
            # print('Loading map (from expert):', self._map.level)
            self._cur_action = 0
            initial = self._map.initial

        self._store = store
        return state_to_matrix(initial, initial.dims)

    def get_expert_action_value(self):
        if self._use_server:
            raise Exception('Cannot use server when getting expert actions')

        transition = self._map.get_transition(self._cur_action)
        return self._actions.index(transition.action), self._map.value

    def step(self, action=None):
        if (self._use_server or self._store) and action is None:
            raise Exception('No action given when environment is set to use server or store the game. Go fuck yourself')

        # To remove IDE complains
        transition = None

        if self._use_server or self._store:
            transition = api.step(self._game_id, self._actions[action])
        if not self._use_server:  # We do this to allow for store hitting both branches
            transition = self._map.get_transition(self._cur_action)

            # important to increment AFTER transition is extracted
            self._cur_action += 1

        return state_to_matrix(transition.state,
                               transition.state.dims), transition.reward, transition.done, transition.success

    def terminate(self, description=''):
        # if store is false, there is no active game on the server
        # Then simply overwrite and return

        if self._store and self._game_id is not None or self._game_id is not None:
            # print('Terminating game:', self._game_id if self._game_id is not None else 'expert game')
            api.terminate(self._game_id, self._store, description=description)
            self._game_id = None

        self._store = False

    def has_more_data(self):
        return self._next_skip < self._total or any(self._maps)

    def _pop(self):
        if self._next_skip < self._total and not any(self._maps):
            self._maps = self._load_maps(self._next_skip, self._batch_size)
            self._next_skip += self._batch_size

        return self._maps.pop()

    def _load_maps(self, skip, take):
        if self._use_server:
            response = api.get_unsupervised_map_list(skip, take)
            maps = list(map(lambda level: level['relativePath'], response['data']))
        else:
            response = api.get_expert_list(skip=skip, take=take)
            maps = list(map(lambda expert_games: expert_games['fileRef'], response['data']))

        self._total = response['total']
        self._next_skip = skip + self._batch_size
        return maps

    def get_action_count(self):
        return len(self._actions)

    @staticmethod
    def get_action_meanings():
        return Env._actions

    @staticmethod
    def map_action(action):
        return Env._actions.index(action)

    @staticmethod
    def get_reward_meanings():
        return Env._rewards

    @staticmethod
    def map_reward(reward):
        return Env._rewards.index(reward)

    def get_play_env(self):
        return Env(True)

    def copy(self, store=False):
        game_id = api.copy_game(self._game_id)
        env = Env(game_id=game_id['gameID'])
        env._store = store
        return env
