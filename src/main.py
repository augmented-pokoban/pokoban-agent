from pprint import pprint

import env.Data
from env.mapper import *

test = env.Data.Data()

trajectory = test.load_random()

matrix = expert_to_matrix(trajectory['initial'])
remapped = matrix_to_expert(matrix)

pprint(sorted(trajectory['initial']['walls'], key=lambda wall: wall['col']))
print('-' * 10)
pprint(sorted(remapped['walls'], key=lambda wall: wall['col']))
