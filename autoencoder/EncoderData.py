import csv
import os
import pickle
import random

from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

import helper


class EncoderData():
    def __init__(self, state_x, action, state_y, reward, success, done):
        self.state_x = state_x
        self.action = action
        self.state_y = state_y
        self.reward = reward
        self.success = success
        self.done = done


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    with ZipFile(filename + '.zip', 'w', ZIP_DEFLATED) as zip_file:
        zip_file.write(filename)
        zip_file.close()

    os.remove(filename)


def load_object(filename):
    zip_file = ZipFile(filename, 'r')
    file = zip_file.open(zip_file.namelist()[0])
    return pickle.load(file)


def batch_to_lists(batch, s_size):
    x_state = np.array(
        list(map(lambda encoded_data: helper.process_frame(encoded_data.state_x, s_size), batch)))  # image biartch
    x_action = np.array(list(map(lambda encoded_data: [encoded_data.action], batch)))  # action batch
    y_state = np.array(
        list(map(lambda encoded_data: helper.process_frame(encoded_data.state_y, s_size), batch)))  # target state batch
    y_reward = np.array(list(map(lambda encoded_data: [encoded_data.reward], batch)))  # target reward batch

    return x_state, x_action, y_state, y_reward


class DataLoader():

    def __init__(self, metadata_file, batch_size, skip_train=0):
        # Load csv file
        # Assign files to train, test and validation
        self._cur_train = None
        self._cur_test = None
        self._batch_size = batch_size

        with open(metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            k_folds = [x for x in reader]

            self._train = self._filter_data(0, 7, k_folds)[skip_train:]
            self._test = self._filter_data(8, 8, k_folds)
            self._val = self._filter_data(9, 9, k_folds)

    def get_train(self):
        if self._cur_train is None or not any(self._cur_train):
            self._cur_train = self._load_batches(self._train, 'TRAIN')

        return self._cur_train.pop() if self._cur_train is not None else None

    def get_test(self):
        if self._cur_test is None or not any(self._cur_test):
            self._cur_test = self._load_batches(self._test, 'TEST')

        return self._cur_test.pop()

    def _filter_data(self, start, end, k_folds):
        return list(map(lambda row: row['file'], filter(lambda row: start <= int(row['k']) <= end, k_folds)))

    def _load_batches(self, file_set, data_set_name):
        print('loading new set of batches for ' + data_set_name + ': ', end='')
        batches = []

        if data_set_name is 'TEST':
            next_file = random.choice(file_set)
        elif not any(file_set):
            return None
        else:
            next_file = file_set.pop()

        print(next_file)

        data_set = load_object('../batches/' + next_file)

        batch = []
        counter = 0
        while any(data_set):
            batch.append(data_set.pop())
            counter += 1

            if counter == self._batch_size:
                batches.append(batch)
                batch = []
                counter = 0

        return batches




