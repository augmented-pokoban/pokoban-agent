from env.mapper import *
from random import choice
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
            'PULL_WEST',
            'PUSH_NORTH',
            'PUSH_SOUTH',
            'PUSH_EAST',
            'PUSH_WEST'
        ]

    def __init__(self, use_server=True, game_id=None):
        self._use_server = use_server
        self._store = False
        self._map = None
        self._cur_action = None
        self._game_id = game_id

        if self._use_server and not game_id:
            self._maps = api.get_unsupervised_map_list()['data']
        elif not self._use_server:
            self._maps = list(map(lambda expert_games: expert_games['id'], api.get_expert_list()))

    def get_action_count(self):
        return len(self._actions)

    def reset(self, store=False, level=None):

        self.terminate()

        if self._use_server:
            map_choice = self._maps.pop() if level is None else level
            self._game_id, initial = api.init(map_choice, unsupervised=True)
            # print("Playing game: ", map_choice)

        else:
            map_id = self._maps.pop()
            self._map = ExpertMoves(api.get_expert_game(map_id))
            # print('Loading map (from expert):', self._map.level)
            self._cur_action = 0
            initial = self._map.initial

            if store:
                self._game_id, _ = api.init(self._map.level, unsupervised=False)

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

        return state_to_matrix(transition.state, transition.state.dims), transition.reward, transition.done, transition.success

    def terminate(self, description=''):
        # if store is false, there is no active game on the server
        # Then simply overwrite and return

        if self._store and self._game_id is not None or self._game_id is not None:
            # print('Terminating game:', self._game_id if self._game_id is not None else 'expert game')
            api.terminate(self._game_id, self._store, description=description)
            self._game_id = None

        self._store = False

    def has_more_data(self):
        return any(self._maps)

    @staticmethod
    def get_action_meanings():
        return Env._actions

    def get_play_env(self):
        if self._use_server:
            return self

        return Env(True)

    def copy(self, store=False):
        game_id = api.copy_game(self._game_id)
        env = Env(game_id=game_id['gameID'])
        env._store = store
        return env



