import requests
from env.expert_moves import State, Transition

base_url = 'http://localhost:8080/pokoban-server/api/'


def get_map_list():
    return requests.get(base_url + 'levels').json()


def get_expert_list():
    return requests.get(base_url).json()


def init(game_file):
    result = requests.post(base_url + game_file).json()
    return result['gameID'], State(result['state'])


def step(game_id, action):
    result = requests.post(base_url + game_id + '/' + action).json()
    return Transition(result)


def terminate(game_id, store=False, description='', is_planner=False):
    requests.delete(base_url + game_id, params={'store': store, 'description': description, 'is_planner': is_planner})
