import env.api as api

experts = api.get_expert_list()

for exp in experts:
    print(exp)

