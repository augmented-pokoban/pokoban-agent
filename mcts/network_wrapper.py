class NetworkWrapper:
    def __init__(self, sess, rnn_state_init, eval_function):
        self.sess = sess
        self.rnn_state_init = rnn_state_init
        self._eval_function = eval_function
        self.running_rnn = None

    def start(self):
        self.running_rnn = self.rnn_state_init

    def eval(self, state):
        a,v, rnn_state = self._eval_function(self.sess, state, self.running_rnn)
        self.running_rnn = rnn_state
        return a, v


