from env import api
from env.Env import Env


def save_state_diff(exp_r=0, act_r=0, diff=None, errors=None, x_state=None, y_state_exp=None, y_state_act=None,
                    action=0, success=False, done=False):

    result = dict()

    result['errors'] = errors
    result['x_state'] = x_state
    result['y_state_exp'] = y_state_exp
    result['y_state_eval'] = y_state_act
    result['diff'] = diff
    result['y_reward_exp'] = Env.get_reward_meanings()[exp_r]
    result['y_reward_eval'] = Env.get_reward_meanings()[act_r]
    result['action'] = Env.get_action_meanings()[action]
    result['success'] = success
    result['done'] = done
    # result['missing'] = matrix_to_state(reshape_back(missing, height, width, depth), 20)
    # result['overfit'] = matrix_to_state(reshape_back(overfit, height, width, depth), 20)

    api.post_encoding(result)
