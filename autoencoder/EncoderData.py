import csv
import os
import pickle
import random

from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

import helper
from env.mapper import old_matrix_to_new_matrix


class EncoderData():
    def __init__(self, state_x, action, state_y, reward, success, done):
        self.state_x = state_x  # old batches is stored as old matrix, random moves are stored as flattened new matrix
        self.action = action  # The index of the action
        self.state_y = state_y  # old batches is stored as old matrix, random moves are stored as flattened new matrix
        self.reward = reward  # The index of the reward
        self.success = success  # true/false
        self.done = done  # true/false


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
        list(map(lambda encoded_data: helper.process_frame(old_matrix_to_new_matrix(encoded_data.state_x,20), s_size), batch)))  # image biartch
    x_action = np.array(list(map(lambda encoded_data: [encoded_data.action], batch)))  # action batch
    y_state = np.array(
        list(map(lambda encoded_data: helper.process_frame(old_matrix_to_new_matrix(encoded_data.state_y,20), s_size), batch)))  # target state batch
    y_reward = np.array(list(map(lambda encoded_data: [encoded_data.reward] if not encoded_data.done else [3], batch)))  # target reward batch
    y_success = np.array(list(map(lambda encoded_data: [1 if encoded_data.success else 0], batch)))

    return x_state, x_action, y_state, y_reward, y_success


def batch_to_lists_preprocessed(batch, s_size):
    x_state = np.array(
        list(map(lambda encoded_data: encoded_data.state_x, batch)))  # image biartch
    x_action = np.array(list(map(lambda encoded_data: [encoded_data.action], batch)))  # action batch
    y_state = np.array(
        list(map(lambda encoded_data: encoded_data.state_y, batch)))  # target state batch
    y_reward = np.array(list(map(lambda encoded_data: [encoded_data.reward] if not encoded_data.done else [3], batch)))  # target reward batch
    y_success = np.array(list(map(lambda encoded_data: [1 if encoded_data.success else 0], batch)))

    return x_state, x_action, y_state, y_reward, y_success


class DataLoader():

    def __init__(self, metadata_file, batch_size, batch_path, skip_train=0):
        # Load csv file
        # Assign files to train, test and validation
        self._cur_train = None
        self._cur_test = None
        self._cur_val = None
        self._batch_size = batch_size
        self._batch_path = batch_path

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

        return self._cur_test.pop() if self._cur_test is not None else None

    def get_val(self):
        if self._cur_val is None or not any(self._cur_val):
            self._cur_val = self._load_batches(self._val, 'VAL')

        return self._cur_val.pop() if self._cur_val is not None else None

    def has_val(self):
        return any(self._val)

    def _filter_data(self, start, end, k_folds):
        return list(map(lambda row: row['file'], filter(lambda row: start <= int(row['k']) <= end, k_folds)))

    def _load_batches(self, file_set, data_set_name):
        print('loading new set of batches for ' + data_set_name + ': ', end='')
        batches = []

        if not any(file_set):
            return None
        else:
            next_file = file_set.pop()

        print(next_file)

        data_set = load_object(self._batch_path + next_file)

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




