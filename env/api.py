import requests
from env.expert_moves import State, Transition

base_url = 'http://localhost:8080/pokoban-server/api/'


def get_unsupervised_map_list():
    return requests.get(base_url + 'levels/supervised').json()


def get_expert_list():
    return requests.get(base_url + 'pokoban').json()


def get_expert_game(game_id):
    return requests.get(base_url + 'pokoban/' + game_id).json()


def init(game_file):
    result = requests.post(base_url + 'pokoban/' + game_file).json()
    return result['gameID'], State(result['state'])


def copy_game(game_id):
    return requests.post(base_url + 'pokoban/' + game_id + '/action/copy').json()


def step(game_id, action):
    result = requests.post(base_url + 'pokoban/' + game_id + '/' + action).json()
    return Transition(result)


def terminate(game_id, store=False, description='', is_planner=False):
    requests.delete(base_url + 'pokoban/' + game_id, params={'store': store, 'description': description,
                                                             'is_planner': is_planner})
