import env.api as api
from env.expert_moves import ExpertMoves

experts = list(map(lambda exp: ExpertMoves(exp), api.get_expert_list()))

for exp in experts:
    print(exp.level)

