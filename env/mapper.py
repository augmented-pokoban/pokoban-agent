import numpy as np

import env.MatrixIndex as INDEX
from env.expert_moves import State


def state_to_matrix(state, dimensions):
    def map_box(x):
        return {
            'A': INDEX.BoxA,
            'B': INDEX.BoxB,
            'C': INDEX.BoxC
        }[x]

    def map_goal(x):
        return {
            'a': INDEX.GoalA,
            'b': INDEX.GoalB,
            'c': INDEX.GoalC
        }[x]

    matrix = np.zeros((dimensions, dimensions, 8))

    for agent in state.agents:
        matrix[agent.row, agent.row, INDEX.Agent] = 1

    for box in state.boxes:
        matrix[box.row, box.row, map_box(box.letter)] = 1

    for goal in state.goals:
        matrix[goal.row, goal.row, map_goal(goal.letter)] = 1

    for wall in state.walls:
        matrix[wall.row, wall.row, INDEX.Wall] = 1

    return matrix


def matrix_to_state(matrix, dimensions):
    state = dict()
    state['boxes'] = []
    state['agents'] = []
    state['walls'] = []
    state['goals'] = []

    def map_type(x):
        return {
            INDEX.Wall: ('walls', '+'),
            INDEX.Agent: ('agents', '0'),
            INDEX.BoxA: ('boxes', 'A'),
            INDEX.BoxB: ('boxes', 'B'),
            INDEX.BoxC: ('boxes', 'C'),
            INDEX.GoalA: ('goals', 'a'),
            INDEX.GoalB: ('goals', 'b'),
            INDEX.GoalC: ('goals', 'c')
        }[x]

    for row in range(dimensions):
        for col in range(dimensions):
            for field_type in range(8):
                if matrix[row, col, field_type]:
                    (key, letter) = map_type(field_type)
                    state[key].append({'row': row, 'col': col, 'letter': letter})

    return State(state)
