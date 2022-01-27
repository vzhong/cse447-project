#!/usr/bin/env python
import os
import string
import random
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import train_helper as th
from predict import pred

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        self.unigram_data = None
        self.bigram_data = None
        self.trigram_data = None
        self.unknown_char_set = None

    @classmethod
    def load_training_data(cls):
        train_file = 'src/train_data.txt'
        with open(train_file) as f:
            return f.readlines()

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        N, char_count, unknown_chars = th.unigram(data)
        self.unigram_data = (N, char_count)
        self.bigram_data = th.bigram(data, unknown_chars)
        self.trigram_data = th.trigram(data, unknown_chars)
        self.unknown_char_set = unknown_chars
        
    def run_pred(self, data):
        # your code here
        lambda_values = [0.325, 0.325, 0.35]
        N, char_count = self.unigram_data
        bigram_sum, bigram_count = self.bigram_data
        trigram_sum, trigram_count = self.trigram_data
        vocab = set(char_count.keys())
        preds = []
        for inp in data:
            tok_1, tok_2 = '<start>', '<start>'
            if len(inp)>= 1:
                tok_2 = inp[-1]
            if len(inp) >= 2:
                tok_1 = inp[-2]
            top_3 = pred(N, char_count, self.unknown_char_set, vocab, bigram_sum, 
                        bigram_count, trigram_sum, trigram_count, lambda_values, tok_1, tok_2)
            preds.append(top_3)
        return preds

    def save(self, work_dir):
        temp_file = 'checkpoint.pkl'
        fullpath = os.path.join(work_dir, temp_file)
        with open(fullpath, 'wb') as f:
            pickle.dump(self.unigram_data, f)
            pickle.dump(self.bigram_data, f)
            pickle.dump(self.trigram_data, f)
            pickle.dump(self.unknown_char_set, f)

    @classmethod
    def load(cls, work_dir):
        model = cls()
        with open(os.path.join(work_dir, 'checkpoint.pkl'), 'rb') as f:
            model.unigram_data = pickle.load(f)
            model.bigram_data = pickle.load(f)
            model.trigram_data = pickle.load(f)
            model.unknown_char_set = pickle.load(f)
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
