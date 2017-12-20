from support.last_id_store import IdStore

last_episode_count = 1200
workers = 20

last_id = '0007fbded2025f7ee07793da54d794b0'

for i in range(workers):
    store = IdStore("worker_" + str(i))
    store.write_id(last_id)