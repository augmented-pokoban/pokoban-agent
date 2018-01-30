import numpy as np
import env.MatrixIndex as INDEX
import env.NewMatrixIndex as NEW_INDEX
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
        matrix[agent.row, agent.col, INDEX.Agent] = 1

    for box in state.boxes:
        matrix[box.row, box.col, map_box(box.letter)] = 1

    for goal in state.goals:
        matrix[goal.row, goal.col, map_goal(goal.letter)] = 1

    for wall in state.walls:
        matrix[wall.row, wall.col, INDEX.Wall] = 1

    return matrix


def new_state_to_matrix(state, dimensions):
    matrix = np.zeros((dimensions, dimensions))

    for wall in state.walls:
        matrix[wall.row, wall.col] = NEW_INDEX.Wall

    for agent in state.agents:
        matrix[agent.row, agent.col] = NEW_INDEX.Agent

    for box in state.boxes:
        matrix[box.row, box.col] = NEW_INDEX.BoxA

    for goal in state.goals:
        if matrix[goal.row, goal.col] == NEW_INDEX.Agent:
            matrix[goal.row, goal.col] = NEW_INDEX.AgentAtGoalA
        elif matrix[goal.row, goal.col] == NEW_INDEX.BoxA:
            matrix[goal.row, goal.col] = NEW_INDEX.BoxAAtGoalA
        else:
            matrix[goal.row, goal.col] = NEW_INDEX.GoalA

    return matrix


def new_matrix_to_state(matrix, dimensions):
    state = dict()
    state['boxes'] = []
    state['agents'] = []
    state['walls'] = []
    state['goals'] = []
    state['dimensions'] = dimensions

    def map_type(x):
        return {
            NEW_INDEX.Field: [],
            NEW_INDEX.Wall: [('walls', '+')],
            NEW_INDEX.Agent: [('agents', '0')],
            NEW_INDEX.BoxA: [('boxes', 'A')],
            NEW_INDEX.GoalA: [('goals', 'a')],
            NEW_INDEX.AgentAtGoalA: [('agents', '0'), ('goals', 'a')],
            NEW_INDEX.BoxAAtGoalA: [('boxes', 'A'), ('goals', 'a')]
        }[x]

    for row in range(dimensions):
        for col in range(dimensions):
            val = matrix[row, col]
            dicts = map_type(val)

            for field_type in dicts:
                (key, letter) = field_type
                state[key].append({'row': row, 'col': col, 'letter': letter})

    return state


def matrix_to_state(matrix, dimensions):
    state = dict()
    state['boxes'] = []
    state['agents'] = []
    state['walls'] = []
    state['goals'] = []
    state['dimensions'] = dimensions

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

    return state


def old_matrix_to_new_matrix(matrix, dimensions):
    state = State(matrix_to_state(matrix, dimensions))
    return new_state_to_matrix(state, dimensions)
