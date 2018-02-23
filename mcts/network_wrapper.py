class NetworkWrapper:
    def __init__(self, sess, rnn_state, eval_function, network=None):
        self.sess = sess
        self.rnn_state = rnn_state
        self._eval_function = eval_function
        self.next_rnn = None
        self.network = network

    def eval(self, state):
        if self.network is None:
            a,v, rnn_state = self._eval_function(self.sess, state, self.rnn_state, get_all_actions=True)
        else:
            a, v, rnn_state = self._eval_function(self.sess, state, self.rnn_state, self.network, get_all_actions=True)
        self.next_rnn = rnn_state
        return a, v

    def has_running(self):
        return self.next_rnn is not None

    def get_next_wrapper(self):
        return NetworkWrapper(self.sess, self.next_rnn, self._eval_function, self.network)


