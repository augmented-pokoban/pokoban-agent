import urllib.request
import urllib.parse
from io import BytesIO
from zipfile import ZipFile
import json
from env.expert_moves import State, Transition

base_url = 'http://localhost:5000/api/'
# base_url = 'http://pokoban-server.azurewebsites.net/api/'


def ping_server():
    return get_request('pokoban/running', {})


def get_unsupervised_map_list(skip=0, take=10000):
    params = {'skip': skip, 'limit': take}
    return get_request('levels/supervised', params)


def get_expert_list(skip=0, take=10000):
    return get_request('pokoban/saves', {'skip': skip, 'limit': take})


def get_replays_list(skip=0, take=10000):
    return get_request('pokoban/replays', {'skip': skip, 'limit': take})


def get_expert_game(file_ref):
    zip_file = ZipFile(BytesIO(urllib.request.urlopen(file_ref).read()))
    return json.loads(zip_file.open(zip_file.namelist()[0]).read())


def init(game_file):
    result = post_request('pokoban/' + game_file.replace('/', '_'))
    return result['gameID'], State(result['state'])


def copy_game(game_id):
    return None
    # return requests.post(base_url + 'pokoban/' + game_id + '/action/copy').json()


def step(game_id, action):
    result = put_request('pokoban/' + game_id + '/' + action)
    return Transition(result)


def terminate(game_id, store=False, description='', is_planner=False):
    url_params = {'store': store,
                  'description': description,
                  'is_planner': is_planner}

    delete_request('pokoban/' + game_id, url_params)


class RequestWithMethod(urllib.request.Request):
    def __init__(self, *args, **kwargs):
        self._method = kwargs.pop('method', None)
        urllib.request.Request.__init__(self, *args, **kwargs)

    def get_method(self):
        return self._method if self._method else super(RequestWithMethod, self).get_method()


def get_request(url, url_params):
    url += '?' + urllib.parse.urlencode(url_params)
    with urllib.request.urlopen(base_url + url) as response:
        data = response.read()
        encoding = response.info().get_content_charset('utf-8')
        return json.loads(data.decode(encoding))


def put_request(url):
    opener = urllib.request.build_opener(urllib.request.HTTPHandler)
    request = RequestWithMethod(url=base_url + url, method='PUT', data=None)
    with opener.open(request) as response:
        data = response.read()
        encoding = response.info().get_content_charset('utf-8')
        return json.loads(data.decode(encoding))


def delete_request(url, url_params):
    url += '?' + urllib.parse.urlencode(url_params)
    opener = urllib.request.build_opener(urllib.request.HTTPHandler)
    request = RequestWithMethod(url=base_url + url, method='DELETE')
    opener.open(request)


def post_request(url):
    opener = urllib.request.build_opener(urllib.request.HTTPHandler)
    request = RequestWithMethod(url=base_url + url, method='POST', data=None)
    with opener.open(request) as response:
        data = response.read()
        encoding = response.info().get_content_charset('utf-8')
        return json.loads(data.decode(encoding))
