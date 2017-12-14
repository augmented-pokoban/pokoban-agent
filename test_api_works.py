import env.api as api

resp = api.get_unsupervised_map_list(take=10)

print(resp)

last_id = resp['data'][-1]['_id']

resp = api.get_unsupervised_map_list(last_id, 10)
print(resp)
