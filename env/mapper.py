import numpy as np
import env.MatrixIndex as INDEX
import env.NewMatrixIndex as NEW_INDEX
import helper
from env import Env
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


def apply_action(state, action, action_list, reshape=True):
    state = np.copy(state)

    if reshape:
        state = helper.reshape_back(state, 20, 20)

    state, success = modify_state(state, action_list[action])

    done = completed(state)

    if reshape:
        state = helper.process_frame(state, 20*20)

    return state, success, done


def completed(state):
    for row in range(20):
        for col in range(20):
            if state[row, col] == NEW_INDEX.GoalA or state[row, col] == NEW_INDEX.AgentAtGoalA:
                return False

    return True


def modify_state(state, action_str):
    for row in range(20):
        for col in range(20):
            if state[row, col] == NEW_INDEX.Agent or state[row, col] == NEW_INDEX.AgentAtGoalA:
                # Found agent, apply actions

                agent_field, box_field = get_affected_fields(action_str, row, col)

                if 'MOVE' in action_str:
                    state, success = move_action(state, agent_field, box_field)
                    if success:
                        # Update the current agent location
                        agent_field_after = NEW_INDEX.GoalA if state[row, col] == NEW_INDEX.AgentAtGoalA else NEW_INDEX.Field
                        state[row, col] = agent_field_after
                else:
                    # Pull action
                    state, success = pull_action(state, agent_field, box_field)

                    if success:
                        agent_field_after = NEW_INDEX.BoxA if state[row, col] == NEW_INDEX.Agent else NEW_INDEX.BoxAAtGoalA
                        state[row, col] = agent_field_after

                return state, success


def get_affected_fields(action_str, agent_row, agent_col):

    new_agent_row, new_agent_col = get_row_col(action_str, agent_row, agent_col)

    if 'PULL' in action_str:
        opposite_dir = get_opposite_dir(action_str)
        box_row, box_col = get_row_col(opposite_dir, agent_row, agent_col)
    else:
        # Move action
        box_row, box_col = get_row_col(action_str, new_agent_row, new_agent_col)

    return (new_agent_row, new_agent_col), (box_row, box_col)


def get_row_col(action_str, row, col):

    if 'NORTH' in action_str:
        return row - 1, col

    if 'SOUTH' in action_str:
        return row + 1, col

    if 'EAST' in action_str:
        return row, col + 1

    if 'WEST' in action_str:
        return row, col - 1


def get_opposite_dir(action_str):
    if 'NORTH' in action_str:
        return 'SOUTH'

    if 'SOUTH' in action_str:
        return 'NORTH'

    if 'EAST' in action_str:
        return 'WEST'

    if 'WEST' in action_str:
        return 'EAST'


def move_action(state, new_agent_field, new_box_field):
    # return new state, success (done check is conducted after)
    new_agent_field_content = state[new_agent_field[0], new_agent_field[1]]

    if new_agent_field_content == NEW_INDEX.Field:
        state[new_agent_field[0], new_agent_field[1]] = NEW_INDEX.Agent
        return state, True

    if new_agent_field_content == NEW_INDEX.GoalA:
        state[new_agent_field[0], new_agent_field[1]] = NEW_INDEX.AgentAtGoalA
        return state, True

    if new_agent_field_content == NEW_INDEX.BoxA or new_agent_field_content == NEW_INDEX.BoxAAtGoalA:
        new_agent_field_after = NEW_INDEX.Agent if new_agent_field_content == NEW_INDEX.BoxA else NEW_INDEX.AgentAtGoalA

        new_box_field_content = state[new_box_field[0], new_box_field[1]]

        if new_box_field_content == NEW_INDEX.Field:
            state[new_box_field[0], new_box_field[1]] = NEW_INDEX.BoxA
            state[new_agent_field[0], new_agent_field[1]] = new_agent_field_after
            return state, True

        if new_box_field_content == NEW_INDEX.GoalA:
            state[new_box_field[0], new_box_field[1]] = NEW_INDEX.BoxAAtGoalA
            state[new_agent_field[0], new_agent_field[1]] = new_agent_field_after
            return state, True

    # Else, it failed
    return state, False


def pull_action(state, new_agent_field, box_field):
    box_field_content = state[box_field[0], box_field[1]]
    new_agent_field_content = state[new_agent_field[0], new_agent_field[1]]

    field_has_box = box_field_content == NEW_INDEX.BoxA or box_field_content == NEW_INDEX.BoxAAtGoalA
    a_field_is_free = new_agent_field_content == NEW_INDEX.Field or new_agent_field_content == NEW_INDEX.GoalA
    if field_has_box and a_field_is_free:
        # success
        box_cont_after = NEW_INDEX.Field if box_field_content == NEW_INDEX.BoxA else NEW_INDEX.GoalA
        agent_cont_after = NEW_INDEX.Agent if new_agent_field_content == NEW_INDEX.Field else NEW_INDEX.AgentAtGoalA

        state[new_agent_field[0], new_agent_field[1]] = agent_cont_after
        state[box_field[0], box_field[1]] = box_cont_after
        return state, True

    return state, False


def validate_transition(x_state, y_state, action):
    x_state_state = State(new_matrix_to_state(old_matrix_to_new_matrix(x_state, 20),20))
    x_state = helper.process_frame(old_matrix_to_new_matrix(x_state, 20), 400)
    y_state = helper.process_frame(old_matrix_to_new_matrix(y_state, 20), 400)

    state, success, done = apply_action(x_state, action, Env.get_action_meanings())

    equal = np.array_equal(state, y_state)

    act_state = State(new_matrix_to_state(helper.reshape_back(state, 20, 20), 20))

    return act_state, x_state_state, equal










