from env.mapper import *
from random import choice
import env.api as api
from env.expert_moves import ExpertMoves

class Env:

    def __init__(self, use_server=True):
        self._use_server = use_server
        self._store = False
        self._map = None
        self._cur_action = None
        self._game_id = None
        self._actions = [
            'MOVE_NORTH',
            'MOVE_SOUTH',
            'MOVE_EAST',
            'MOVE_WEST',
            'PULL_NORTH',
            'PULL_SOUTH',
            'PULL_EAST',
            'PULL_WEST',
            'PUSH_NORTH',
            'PUSH_SOUTH',
            'PUSH_EAST',
            'PUSH_WEST'
        ]

        if self._use_server:
            self._maps = list(map(lambda level: level['filename'], api.get_map_list()))
        else:
            self._maps = list(map(lambda expert_moves: ExpertMoves(expert_moves), api.get_expert_list()))
            # self._expert = Data()

    def reset(self, store=False, level=None):

        self.terminate()

        if self._use_server:
            map_choice = choice(self._maps) if level is None else level
            self._game_id, initial = api.init(map_choice)
            # print("Playing game: ", map_choice)

        else:
            self._map = choice(self._maps)
            # print('Loading map (from expert):', self._map.level)
            self._cur_action = 0
            initial = self._map.initial

            if store:
                self._game_id, _ = api.init(self._map.level)

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

        if self._use_server or self._store:
            transition = api.step(self._game_id, self._actions[action])
        if not self._use_server:  # We do this to allow for store hitting both branches
            transition = self._map.get_transition(self._cur_action)

            # important to increment AFTER transtition is extracted
            self._cur_action += 1

        return state_to_matrix(transition.state, transition.state.dims), transition.reward, transition.done, transition.success

    def terminate(self, description=''):
        # if store is false, there is no active game on the server
        # Then simply overwrite and return

        if self._store and self._game_id is not None or self._game_id is not None:
            # print('Terminating game:', self._game_id if self._game_id is not None else 'expert game')
            api.terminate(self._game_id, self._store, description=description)
            self._game_id = None

        self._store = False

    def get_action_meanings(self):
        return self._actions




