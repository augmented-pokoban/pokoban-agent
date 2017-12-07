import os
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


class DataReader():
    def __init__(self, path='batches/'):
        self._path = path
        self._filenames = os.listdir(path)
        self._filenames.sort()
        self._cur_list = None

    def _load_file(self):
        with open(self._path + self._filenames.pop(), 'rb') as file:
            self._cur_list = pickle.load(file)

    def has_more(self, batch_size):
        """
        This assumes that a file contains more than the batch size
        :param batch_size: The size of the batch, number
        :return: Bool
        """
        return self._cur_list is None or len(self._cur_list) >= batch_size or any(self._filenames)

    def get_batch(self, batch_size):
        """
        Returns a list of EncoderData of length = batch_size.
        It pops the elements from the internal current list and loads another file if needed.
        :param batch_size:
        :return: A list of length batch_size of EncoderData
        """
        batch = []
        for i in range(batch_size):
            if self._cur_list is None or not any(self._cur_list):
                self._load_file()

            batch.append(self._cur_list.pop())

        return batch


