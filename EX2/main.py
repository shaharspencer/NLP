import re

import numpy as np
from abc import ABC, abstractmethod
from nltk.corpus import brown
import pandas as pd


# QUESTION 2
class POSTagger():
    def __init__(self):
        self.tagged = brown.tagged_sents(categories="news")
        self.train, self.test = self.split_train_test(self.tagged)
        self.train_tag_baseline()
        # self.get_MLE_tag("The")

    def split_train_test(self, sents):
        sents_length = len(sents)
        split_indx = int (0.9*sents_length)
        return sents[:split_indx], sents[split_indx:]

    def train_tag_baseline(self):
        # for each word, calculate most likely pos
        self.POS_MAX_LIKELIHOOD = {}
        for sent in self.train:
            for word, pos in sent:
                # seen word
                if word in self.POS_MAX_LIKELIHOOD:
                    self.POS_MAX_LIKELIHOOD[word][pos] = \
                    self.POS_MAX_LIKELIHOOD[word].get(pos, 0) + 1
                # unseen word
                else:
                    self.POS_MAX_LIKELIHOOD[word] = {pos:1}

    def get_MLE_tag(self, word):
        try:
            return max(self.POS_MAX_LIKELIHOOD[word],
                       key = self.POS_MAX_LIKELIHOOD[word].get)
        except KeyError:
            return "NN"


    def unknown_and_known__and_total_error(self):
        unknown_accuracy = 0
        unknown_num = 0
        known_accuracy = 0
        known_num = 0
        for sent in self.test:
            for word, tag in sent:
                # calculate mle tag
                mle_tag = self.get_MLE_tag(word)
                # calclate accuracy
                correct = int(mle_tag == tag)
                # check if known or unknown
                if word in self.POS_MAX_LIKELIHOOD.keys():
                    known_accuracy += correct
                    known_num += 1
                else:
                    unknown_accuracy += correct
                    unknown_num += 1

        return 1-known_accuracy/known_num, 1-unknown_accuracy/unknown_num, \
            1- ((known_accuracy+unknown_accuracy)/(unknown_num+known_num))






class BigramHMM(ABC):

    def __init__(self):
        self.emissions = None
        self.transitions = None

    @abstractmethod
    def computeEmissions(self):
        pass

    @abstractmethod
    def computeTransitions(self):
        pass

    def fit(self):
        self.computeEmissions()
        self.computeTransitions()

    def getEmission(self, x, given_y):
        try:
            return self.emissions[given_y][x]
        except KeyError:
            return 0

    def getTransition(self, y, given_y):
        try:
            return self.transitions[given_y][y]
        except KeyError:
            return 0

    def predict(self, x_sequence):
        """
        Viterbi Algorithm - get y sequence that maxes prob of x sequence
        :return: max prob, y sequence
        """

        table = []
        for i, x_i in enumerate(x_sequence):
            table.append([])
            if i == 0:
                for y in self.transitions["START"]:
                    table[-1].append([y, [y],
                                      self.getTransition(y, "START") *
                                      self.getEmission(x_i, y)])

            else:
                prev_stage = table[-2]
                for option in prev_stage:
                    if option[0] == ':-HL':
                        pass
                    try:
                        prob = [self.getTransition(y, option[0]) * self.getEmission(x_i, y) for y in
                            self.transitions[option[0]]]
                    except KeyError:
                        print(option[0])
                        exit()
                    argmax = np.argmax(prob)
                    max_y = list(self.transitions[option[0]])[argmax]
                    max_prob = prob[argmax]
                    table[-1].append([max_y, option[1] + [max_y], max_prob * option[2]])

        total_argmax = np.argmax(np.array(table[-1], dtype=object)[:, 2])
        max_prob = table[-1][total_argmax][2]
        max_sequence = table[-1][total_argmax][1]

        return max_prob, max_sequence


class NucleotideHMM(BigramHMM):

    def computeEmissions(self):
        self.emissions = {"H": {"A": 0.2, "C": 0.3, "G": 0.3, "T": 0.2},
                          "L": {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}}

    def computeTransitions(self):
        self.transitions = {"START": {"H": 0.5, "L": 0.5},
                            "H": {"H": 0.5, "L": 0.5},
                            "L": {"H": 0.4, "L": 0.6}}


class BrownPOS(BigramHMM):
    import re
    def __init__(self):
        super().__init__()
        self.prev_data = brown.tagged_sents(categories="news")
        self.clean_data()
        self.train, self.test = self.data[:int(0.9 * len(self.data))], self.data[int(0.9 * len(self.data)):]
        self.seen_words = set()

    def clean_data(self):
        self.data = []
        for i, sent in enumerate(self.prev_data):
            for j in range(len(sent)):
                new_pos = re.split('-|\+|\*', sent[j][1])
                self.prev_data[i][j] = (sent[j][0], new_pos[0])
            self.data.append(sent)


    def compute_error(self):
        known_accuracy = 0
        total_seen = 0
        correct_unseen = 0
        total_unseen = 0
        # for each sentnce, predict tags
        for sent in self.test:
            sent_words = [word[0] for word in sent]
            sent_tags = [word[1] for word in sent]
            prob, prediction = self.predict(sent_words)
            for i, word in enumerate(sent_words):
                # if is known
                if word in self.seen_words:
                    known_accuracy += (sent_tags[i] == prediction[i])
                    total_seen += 1
                else:
                    correct_unseen += (sent_tags[i] == prediction[i])
                    total_unseen += 1
        return 1 - known_accuracy / total_seen, 1 - correct_unseen / total_unseen, \
               1 - ((known_accuracy + correct_unseen) / (
                           total_seen+ total_unseen))


    def computeEmissions(self):
        self.emissions = {}
        for sent in self.train:
            for word, pos in sent:
                self.seen_words.add(word)
                if pos not in self.emissions:
                    self.emissions[pos] = {word: 1}
                else:
                    self.emissions[pos][word] = self.emissions[pos].get(word, 0) + 1

        for pos in self.emissions.keys():
            pos_total = sum(self.emissions[pos].values())
            for word in self.emissions[pos].keys():
                self.emissions[pos][word] /= pos_total

    def computeTransitions(self):
        self.transitions = {}

        for sent in self.train:
            for i, obj in enumerate(sent):
                pos = obj[1]
                if i == 0:
                    prev = "START"
                else:
                    prev = sent[i - 1][1]

                if prev in self.transitions:
                    self.transitions[prev][pos] = self.transitions[prev].get(pos, 0) + 1
                else:
                    self.transitions[prev] = {pos: 1}

        for pos in self.transitions.keys():
            pos_total = sum(self.transitions[pos].values())
            for next_pos in self.transitions[pos].keys():
                self.transitions[pos][next_pos] /= pos_total

    def getEmission(self, x, given_y):
        """
        if word is not in train data, treat as if it appeared once as NN
        """
        if x in self.seen_words:
            try:
                return self.emissions[given_y][x]
            except KeyError:
                return 0
        if given_y == "NN":
                # return 1 / sum(self.emissions['NN'].values())
            return 1
        return 0


class AddOne(BrownPOS):

    def computeEmissions(self):

        self.emissions = {}
        for sent in self.train:
            for word, pos in sent:
                self.seen_words.add(word)
                if pos not in self.emissions:
                    self.emissions[pos] = {word: 1}
                else:
                    self.emissions[pos][word] = self.emissions[pos].get(word,
                                                                        0) + 1

        for pos in self.emissions.keys():
            appearances = sum(self.emissions[pos].values())
            num_words = len(list(self.emissions[pos].keys()))
            for key in self.emissions[pos]:
                self.emissions[pos][key] = (1+self.emissions[pos][key]) / (appearances + num_words)

    def getEmission(self, x, given_y):
        """
        if word is not in train data, treat as if it appeared once as NN
        """
        if x in self.seen_words:
            try:
                return self.emissions[given_y][x]
            except KeyError:
                return 0
        else:
            appearances = sum(self.emissions[given_y].values())
            num_words = len(list(self.emissions[given_y].keys()))
            return (1 / appearances + num_words)



class PseudoWords(BrownPOS):
    def __init__(self):
        super().__init__()
        self.FreqOfWord = self.compute_corpus_word_frequency()
        # save words in train set for error calculation
        self.in_train_words = set()
        for sent in self.train:
            for word, pos in sent:
                self.in_train_words.add(word)
        # clean data - encode words
        self.TurnIntoPseudoWords()

    def TurnIntoPseudoWords(self):
        new_data = []
        import copy
        # save original data before encoding
        self.data_before_pseudo = copy.deepcopy(self.data)
        for i, sent in enumerate(self.data):
            for j in range(len(sent)):
                sent[j] = (self.GetPseudoWord(sent[j][0]), sent[j][1])
            new_data.append(sent)
        # turn self.data into clean data
        self.data = new_data
        # define clean train and test
        self.train, self.test = self.data[
                                :int(0.9 * len(self.data))], self.data[
                                                             int(0.9 * len(
                                                                 self.data)):]
        # count frequencies of words and test set words
        self.FreqOfWord = self.compute_corpus_word_frequency()

    def GetPseudoWord(self, word):
        if self.FreqOfWord[word] < 5:
            return self.getPseudoTag(word)
        return word

    def getPseudoTag(self, word):
        if re.compile("[0-9][0-9]").fullmatch(word):
            return "twoDigYr"
        if re.compile("[0-2][0-9][0-9][0-9]").fullmatch(word):
            return "forDigTy"
        if re.compile("[0-9]*[A-Z]*[0-9]*[A-Z]*").fullmatch(word):
            return "prodNum"
        if re.compile("[0-3]?[0-9]\/[1-2]?[0-9]").fullmatch(word):
            return "dateShort"
        if re.compile("[0-3]?[0-9]\/[1-2]?[0-9]\/[0-9][0-9]").fullmatch(word):
            return "dateLong"
        if re.compile("[0-9]+\,?[0-9]*\.?[0-9]+").fullmatch(word):
            return "amount"
        if re.compile("[0-9]?[0-9]\.[0-9]+").fullmatch(word):
            return "amountPercent"
        if re.compile("[0-9]+").fullmatch(word):
            return "otherNum"
        if re.compile("[A-Z][A-Z]+").fullmatch(word):
            return "allCaps"
        if re.compile("[A-Z]\.").fullmatch(word):
            return "capsPeriod"
        if re.compile("[A-Z][a-z]+").fullmatch(word):
            return "initCaps"
        if re.compile("[a-z]+").fullmatch(word):
            return "lowerCase"
        return "other"
    def predict(self, sequence):
        sequence = [self.GetPseudoWord(word) for word in sequence]
        return super().predict(sequence)
    def compute_error(self):
        # save which test set words there were originally
        # to see which words were "unseen"
        self.test_before_pseudo = self.data_before_pseudo[int(0.9 * len(
                                                    self.data_before_pseudo)):]
        known_accuracy = 0
        total_seen = 0
        correct_unseen = 0
        total_unseen = 0
        # for each sentence, predict tags
        for j, sent in enumerate(self.test):
            sent_words = [word[0] for word in sent]
            sent_tags = [word[1] for word in sent]
            prob, prediction = self.predict(sent_words)
            for i, word in enumerate(sent_words):
                # if is seen word
                if self.test_before_pseudo[j][i][0] in self.in_train_words:
                    known_accuracy += (sent_tags[i] == prediction[i])
                    total_seen += 1
                else:
                    correct_unseen += (sent_tags[i] == prediction[i])
                    total_unseen += 1
        return 1 - known_accuracy / total_seen, 1 - correct_unseen / total_unseen, \
               1 - ((known_accuracy + correct_unseen) / (
                           total_seen+ total_unseen))

    def compute_corpus_word_frequency(self):
        FreqOfWord = {}
        for sent in self.train:
            for word, pos in sent:
                FreqOfWord[word] = FreqOfWord.get(word, 0) + 1
        for sent in self.test:
            for word, pos in sent:
                if word in FreqOfWord.keys():
                    continue
                else:
                    FreqOfWord[word] = 0

        return FreqOfWord




class PseudoWordsWithSmoothing(AddOne):
    def __init__(self):
        # intialize AddOne object
        super().__init__()
        # get frequency of words in train set and initlaize 0 for words only in test
        self.FreqOfWord = self.compute_corpus_word_frequency()

        # save words which were in train set for error computation
        self.in_train_words = set()
        for sent in self.train:
                for word, pos in sent:
                    self.in_train_words.add(word)

        # clean data - turn low frequency and test words in self.data to their
        # encoding
        self.TurnIntoPseudoWords()


    def TurnIntoPseudoWords(self):
        new_data = []
        import copy

        # save original data before encoding
        self.data_before_pseudo = copy.deepcopy(self.data)
        for i, sent in enumerate(self.data):
            for j in range(len(sent)):
                sent[j] = (self.GetPseudoWord(sent[j][0]), sent[j][1])
            new_data.append(sent)

        # turn self.data into clean data
        self.data = new_data
        # define clean train and test
        self.train, self.test = self.data[
                                :int(0.9 * len(self.data))], self.data[
                                                             int(0.9 * len(
                                                                 self.data)):]
        # count frequencies of words and test set words
        self.FreqOfWord = self.compute_corpus_word_frequency()

    def GetPseudoWord(self, word):
        if self.FreqOfWord[word] < 5:
            return self.getPseudoTag(word)
        return word

    def getPseudoTag(self, word):
        if re.compile("[0-9][0-9]").fullmatch(word):
            return "twoDigYr"
        if re.compile("[0-2][0-9][0-9][0-9]").fullmatch(word):
            return "forDigTy"
        if re.compile("[0-9]*[A-Z]*[0-9]*[A-Z]*").fullmatch(word):
            return "prodNum"
        if re.compile("[0-3]?[0-9]\/[1-2]?[0-9]").fullmatch(word):
            return "dateShort"
        if re.compile("[0-3]?[0-9]\/[1-2]?[0-9]\/[0-9][0-9]").fullmatch(word):
            return "dateLong"
        if re.compile("[0-9]+\,?[0-9]*\.?[0-9]+").fullmatch(word):
            return "amount"
        if re.compile("[0-9]?[0-9]\.[0-9]+").fullmatch(word):
            return "amountPercent"
        if re.compile("[0-9]+").fullmatch(word):
            return "otherNum"
        if re.compile("[A-Z][A-Z]+").fullmatch(word):
            return "allCaps"
        if re.compile("[A-Z]\.").fullmatch(word):
            return "capsPeriod"
        if re.compile("[A-Z][a-z]+").fullmatch(word):
            return "initCaps"
        if re.compile("[a-z]+").fullmatch(word):
            return "lowerCase"
        return "other"

    # def predict(self, sequence):
    #     sequence = [self.GetPseudoWord(word) for word in sequence]
    #     return super().predict(sequence)

    def compute_error(self):

        self.initialize_confusion_matrix()
        # save which test set words there were originally
        # to know which words were "unseen"
        self.test_before_pseudo = self.data_before_pseudo[int(0.9 * len(
            self.data_before_pseudo)):]
        known_accuracy = 0
        total_seen = 0
        correct_unseen = 0
        total_unseen = 0
        # for each sentence, predict tags
        for j, sent in enumerate(self.test):
            sent_words = [word[0] for word in sent]
            sent_tags = [word[1] for word in sent]
            prob, prediction = self.predict(sent_words)
            self.add_to_confusion_matrix(sent_tags, prediction)
            for i, word in enumerate(sent_words):
                # if is seen word
                if self.test_before_pseudo[j][i][0] in self.in_train_words:
                    known_accuracy += (sent_tags[i] == prediction[i])
                    total_seen += 1
                else:
                    correct_unseen += (sent_tags[i] == prediction[i])
                    total_unseen += 1
        return 1 - known_accuracy / total_seen, 1 - correct_unseen / total_unseen, \
               1 - ((known_accuracy + correct_unseen) / (
                       total_seen + total_unseen))

    def compute_corpus_word_frequency(self):
        FreqOfWord = {}
        for sent in self.train:
            for word, pos in sent:
                FreqOfWord[word] = FreqOfWord.get(word, 0) + 1
        for sent in self.test:
            for word, pos in sent:
                if word in FreqOfWord.keys():
                    continue
                else:
                    FreqOfWord[word] = 0

        return FreqOfWord
    def initialize_confusion_matrix(self):
        # all tokens not including start which we added
        nltk_tokens = list(self.transitions.keys())[1:]
        # initialize dataframe of size len(nltk.tokens)* len(nltk.tokens)
        import pandas as pd
        self.confusion_matrix = {}
        # self.confusion_matrix = pd.DataFrame(columns=nltk_tokens,
        #                                      index=nltk_tokens, data=np.zeros((len(nltk_tokens), len(nltk_tokens))))
    def add_to_confusion_matrix(self, actual_tags, predicted_tags):
        for actual_tag, predicted_tag in zip(actual_tags, predicted_tags):
            if actual_tag in self.confusion_matrix.keys():
                self.confusion_matrix[actual_tag][predicted_tag] = \
                    self.confusion_matrix[actual_tag].get(predicted_tag, 0) + 1
            else:
                self.confusion_matrix[actual_tag] = {}
                self.confusion_matrix[actual_tag][predicted_tag] =\
                    self.confusion_matrix[actual_tag].get(predicted_tag, 0) + 1



def explore_most_frequent_errors(confusion_matrix: dict):
   for key in confusion_matrix.keys():
       try:
       # remove key entry in confusion_matrix[key]
            confusion_matrix[key].pop(key)
       except KeyError:
            pass
   MFE = {}
   for key in confusion_matrix.keys():
        if not (confusion_matrix[key]):
            continue
        max_error = max(confusion_matrix[key], key = confusion_matrix[key].get)
        max_error_num = confusion_matrix[key][max_error]
        MFE[key] = (max_error, max_error_num)
   print(MFE)










if __name__ == "__main__":
    # print("-------------------------------- question 1 --------------------------------")
    # nucleotide = NucleotideHMM()
    # nucleotide.fit()
    # print(nucleotide.predict("ACCGTGCA"))

    print(
        "-------------------------------- question 2 --------------------------------")
    posTagger = POSTagger()
    print("Computing seen, unseen and total error: ")
    print(posTagger.unknown_and_known__and_total_error())

    print("-------------------------------- question 3 --------------------------------")
    posTagger = BrownPOS()
    posTagger.fit()
    print("Computing seen, unseen and total error: ")
    print(posTagger.compute_error())

    print("-------------------------------- question 4 --------------------------------")
    addOne = AddOne()
    addOne.fit()
    print("Computing seen, unseen and total error: ")
    print(addOne.compute_error())

    print(
        "-------------------------------- question 5 --------------------------------")
    pseudoWords = PseudoWords()
    pseudoWords.fit()
    print("ii: Computing seen, unseen and total error withought smoothing: ")

    print(pseudoWords.compute_error())

    print("iii: Computing seen, unseen and total error and outputting confusion matrix: \n")
    pseudoWordsWithAddOne = PseudoWordsWithSmoothing()
    pseudoWordsWithAddOne.fit()
    print(pseudoWordsWithAddOne.compute_error())

    explore_most_frequent_errors(pseudoWordsWithAddOne.confusion_matrix)





   # print(pseudoWords.predict(["I", "want", "200"]))
    # print(pseudoWords.predict(["200149", "kids", "went"]))