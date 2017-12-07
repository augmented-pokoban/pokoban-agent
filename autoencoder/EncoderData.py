import pickle


class EncoderData():
    def __init__(self, state_x, action, state_y, reward):
        self.state_x = state_x
        self.action = action
        self.state_y = state_y
        self.reward = reward


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)