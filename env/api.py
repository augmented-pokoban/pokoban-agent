import requests
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import json
from env.expert_moves import State, Transition

base_url = 'http://localhost:5000/api/'


def get_unsupervised_map_list(skip=0, take=10000):
    return requests.get(base_url + 'levels/unsupervised', params={'skip': skip, 'limit': take}).json()


def get_expert_list(skip=0, take=10000):
    return requests.get(base_url + 'pokoban/saves', params={'skip': skip, 'limit': take}).json()


# TODO: Insert new base url that points to the file storage
def get_expert_game(file_ref):
    zip_file = ZipFile(BytesIO(urlopen(file_ref).read()))
    return json.loads(zip_file.open(zip_file.namelist()[0]).read())


def init(game_file):
    result = requests.post(base_url + 'pokoban/' + game_file.replace('/', '_')).json()
    return result['gameID'], State(result['state'])


def copy_game(game_id):
    return requests.post(base_url + 'pokoban/' + game_id + '/action/copy').json()


def step(game_id, action):
    result = requests.put(base_url + 'pokoban/' + game_id + '/' + action).json()
    return Transition(result)


def terminate(game_id, store=False, description='', is_planner=False):
    requests.delete(base_url + 'pokoban/' + game_id, params={'store': store,
                                                             'description': description,
                                                             'is_planner': is_planner})
