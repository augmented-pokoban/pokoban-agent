import sys

from env.mapper import *
from random import choice, shuffle
import env.api as api
from env.expert_moves import ExpertMoves
from support.last_id_store import IdStore


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

    def __init__(self, use_server=True, id_store=None, game_id=None):
        self._use_server = use_server

        self._store = False
        self._map = None
        self._cur_action = None
        self._game_id = game_id
        self._batch_size = 100

        self._id_store = id_store
        self._has_more = True
        if id_store is not None:
            self._last_id = id_store.get_id()

        if self._use_server and not game_id:
            self._maps = self._load_maps(self._batch_size)
            shuffle(self._maps)

        elif not self._use_server:
            self._maps = self._load_maps(self._batch_size)

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
        return new_state_to_matrix(initial, initial.dims)

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

        return new_state_to_matrix(transition.state,
                                   transition.state.dims), transition.reward, transition.done, transition.success

    def terminate(self, description=''):
        # if store is false, there is no active game on the server
        # Then simply overwrite and return

        # print('{} \t : Terminating game id {}'.format(self._id_store.name, self._game_id))
        sys.stdout.flush()

        if self._game_id is not None:
            # print('Terminating game:', self._game_id if self._game_id is not None else 'expert game')
            api.terminate(self._game_id, self._store, description=description)
            self._game_id = None

        self._store = False

    def has_more_data(self):
        return self._has_more or any(self._maps)

    def _pop(self):
        if self._has_more and not any(self._maps):
            self._maps = self._load_maps(self._batch_size)

        return self._maps.pop()

    def _load_maps(self, take):
        if self._use_server:
            response = api.get_unsupervised_map_list(self._last_id, take)
            maps = list(map(lambda level: level['relativePath'], response['data']))
        else:
            response = api.get_expert_list(self._last_id, take=take, order='asc')
            maps = list(map(lambda expert_games: expert_games['fileRef'], response['data']))

        self._last_id = response['data'][-1]['_id']
        self._id_store.write_id(self._last_id)
        self._has_more = len(response['data']) == take
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
        return Env(True, id_store=IdStore('play'))

    def copy(self, store=False):
        game_id = api.copy_game(self._game_id)
        env = Env(game_id=game_id)
        env._store = store
        return env
