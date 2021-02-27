#!/usr/bin/env python
import os
import string
import random
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    n = 6
    # lang -> ngrams -> list[unigram, bigram ...], each model is a dictionary {prefix:probablity}
    lang_to_ngrams = {}
    start_char = '¢'
    stop_char = '£'
    @classmethod
    def load_training_data(cls):
        # trainPath = r'../shortTranslations/AllTrain'
        trainPath = r'data/train'
        files = os.listdir(trainPath)
        out = {}
        for f in files:
            lang_code = f[:f.find('train')]
            f_ = open(os.path.join(trainPath, f), "r", encoding='utf-16')
            out[lang_code] = f_.read().split("\n")
        return out

    @classmethod
    def load_test_data(cls, fname):
        f_ = open(fname, "r")
        return f_.read().split("\n")

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        for lang in data:
            self.lang_to_ngrams[lang] = []
            for i in range(len(data[lang])):
                data[lang][i] = (self.n - 1) * self.start_char + data[lang][i] + self.stop_char

            # constuct frequency maps and total counts
            total_counts = {}
            for i in range(self.n):
                self.lang_to_ngrams[lang].append({})
                total_counts[i] = {}
            # unigram is a special case
            total_counts[0] = 0
            # fill in frequency maps
            for line in data[lang]:
                for i in range(len(line)):
                    for model_index in range(self.n):
                        remove_starts = self.n - 1 - model_index
                        index = i + remove_starts
                        if(index + model_index >= len(line)):
                            continue
                        # unigram is special case of char -> freq
                        # other n-grams are prefix -> char-> freq
                        if model_index == 0:
                            char = line[index]
                            if char in self.lang_to_ngrams[lang][model_index]:
                                self.lang_to_ngrams[lang][model_index][char] += 1
                            else:
                                self.lang_to_ngrams[lang][model_index][char] = 1
                            total_counts[model_index] += 1
                        else:
                            prefix = line[index : index + model_index]
                            char = line[index + model_index]
                            if prefix in self.lang_to_ngrams[lang][model_index]:
                                if char in self.lang_to_ngrams[lang][model_index][prefix]:
                                    self.lang_to_ngrams[lang][model_index][prefix][char] += 1
                                else:
                                    self.lang_to_ngrams[lang][model_index][prefix][char] = 1
                            else:
                                self.lang_to_ngrams[lang][model_index][prefix] = {}
                                self.lang_to_ngrams[lang][model_index][prefix][char] = 1
                            if prefix in total_counts[model_index]:
                                total_counts[model_index][prefix] += 1
                            else:
                                total_counts[model_index][prefix] = 1
            # convert to probabilities
            for model_index in range(len(self.lang_to_ngrams[lang])):
                # unigram is a special case
                if model_index == 0:
                    for char in self.lang_to_ngrams[lang][model_index]:
                        self.lang_to_ngrams[lang][model_index][char] /= total_counts[model_index]
                else:
                    for prefix in self.lang_to_ngrams[lang][model_index]:
                        for char in self.lang_to_ngrams[lang][model_index][prefix]:
                            self.lang_to_ngrams[lang][model_index][prefix][char] /= total_counts[model_index][prefix]

    def run_pred(self, data):
        # your code here
        preds = []
        l1 = 0.3
        l2 = 0.3
        l3 = 0.4
        for inp in data:
            inp = "--" + inp  # start padding
            prefix = inp[-2:]  # last 2 chars in input
            # dict of lang code -> list of dictionaries
            # unigram char to prob
            # bigram, trigram -> dict is prefix in tuple to any suffix probability

            # prefix = "lo"
            top_guesses = dict()
            for lang in self.lang_to_ngrams:
                lang_models = self.lang_to_ngrams[lang]

                unigram_model = lang_models[0]
                bigram_model = lang_models[1]
                trigram_model = lang_models[2]

                token_to_prob = dict()
                for token in unigram_model:
                    # Probability
                    unigram_probability = unigram_model[token]

                    # Check to ensure tokens in models
                    bigram_probability = 0.0
                    if prefix[1] in bigram_model and token in bigram_model[prefix[1]]:
                        bigram_probability += bigram_model[prefix[1]][token]

                    trigram_probability = 0.0
                    if (prefix[0], prefix[1]) in trigram_model and token in trigram_model[prefix[0], prefix[1]]:
                        trigram_probability += trigram_model[prefix[0], prefix[1]]

                    interpolated_probability = l1 * unigram_probability + l2 * bigram_probability + l3 * trigram_probability

                    token_to_prob[token] = interpolated_probability

                # Get current three chars with highest probability
                highest = sorted(token_to_prob, key=token_to_prob.get, reverse=True)[:3]

                for token in highest:
                    if token not in top_guesses or (token in top_guesses and token_to_prob[token] > top_guesses[token]):
                        top_guesses[token] = token_to_prob[token]

            # Get overall three chars with highest prob
            top_guesses = sorted(token_to_prob, key=token_to_prob.get, reverse=True)[:3]

            preds.append(''.join(top_guesses))

        return preds

    def save(self, work_dir):
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            pickle.dump(self.lang_to_ngrams, f)

    @ classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'rb') as f:
            lang_to_ngrams = pickle.load(f)
        model = MyModel()
        model.lang_to_ngrams = lang_to_ngrams
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=(
        'train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data',
                        default='example/input.txt')
    parser.add_argument(
        '--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

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
        print(model.lang_to_ngrams['en'])
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
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(
            len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
