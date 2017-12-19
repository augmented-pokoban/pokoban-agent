from env import api
from support.last_id_store import IdStore

last_episode_count = 1200
workers = 20

response = api.get_unsupervised_map_list(last_episode_count, 1)
last_id = response['data'][0]['_id']

for i in range(workers):
    store = IdStore("worker_" + str(i))
    store.write_id(last_id)