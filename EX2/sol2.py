import numpy as np
from abc import ABC, abstractmethod
from nltk.corpus import brown
# import nltk
import re


def cleanAndSplit(data):
    train = []
    test = []
    for sent in data[:int(0.9 * len(data))]:
        train.append([(word, re.split("[+\-*]", pos)[0]) for word, pos in sent])

    for sent in data[int(0.9 * len(data)):]:
        test.append([(word, re.split("[+\-*]", pos)[0]) for word, pos in sent])

    return train, test


def getPseudoTag(word):
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


def preprocessData(train, test):
    # handle train data
    freq = {}
    for sent in train:
        for word, pos in sent:
            freq[word] = freq.get(word, 0) + 1

    new_train_data = []
    for sent in train:
        new_train_data.append([])
        for word, pos in sent:
            if freq[word] < 5:
                new_train_data[-1].append((getPseudoTag(word), pos))
            else:
                new_train_data[-1].append((word, pos))

    # handle test data
    new_test_data = []
    for sent in test:
        new_test_data.append([])
        for word, pos in sent:
            try:
                if freq[word] < 5:
                    new_test_data[-1].append((getPseudoTag(word), pos))
                else:
                    new_test_data[-1].append((word, pos))

            except KeyError:
                new_test_data[-1].append((getPseudoTag(word), pos))

    return new_train_data, new_test_data, freq


def exploreMostFreqErrors(model):
    for tag in list(model.confusionMatrix.keys()):
        max_inner = max(model.confusionMatrix[tag], key=model.confusionMatrix[tag].get)
        if max_inner != tag:
            print("Tag " + tag + " was often misclassified as tag " + max_inner)


def computeErrorMLE(data):
    train, test = cleanAndSplit(data)
    words = {}
    for sent in train:
        for word, pos in sent:
            if word in words:
                words[word][pos] = words[word].get(pos, 0) + 1
            else:
                words[word] = {pos: 1}

    # compute Error
    total_seen = 0
    correct_pred_seen = 0
    total_unseen = 0
    correct_pred_unseen = 0

    for sent in test:
        sequence, labels = zip(*sent)
        for y in range(len(sequence)):
            if sequence[y] in words:
                total_seen += 1
                if max(words[sequence[y]], key=words[sequence[y]].get) == labels[y]:
                    correct_pred_seen += 1
            else:
                total_unseen += 1
                if "NN" == labels[y]:
                    correct_pred_unseen += 1

    return 1 - (correct_pred_seen / total_seen), \
           1 - (correct_pred_unseen / total_unseen), \
           1 - (correct_pred_unseen + correct_pred_seen) / (total_seen + total_unseen)


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
                    table[-1].append([y, [y], self.getTransition(y, "START") * self.getEmission(x_i, y)])

            else:
                prev_stage = table[-2]
                for option in prev_stage:
                    prob = [self.getTransition(y, option[0]) * self.getEmission(x_i, y) for y in
                            self.transitions[option[0]]]
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

    def __init__(self):
        super().__init__()
        data = brown.tagged_sents(categories="news")
        self.train, self.test = cleanAndSplit(data)
        self.seen_words = {}

    def computeEmissions(self):
        self.emissions = {}
        for sent in self.train:
            for word, pos in sent:
                self.seen_words[word] = self.seen_words.get(word, 0) + 1

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
        try:
            return self.emissions[given_y][x]
        except KeyError:
            if x not in self.seen_words and given_y == 'NN':
                return 1  # wierd
            return 0

    def computeError(self):
        total_seen = 0
        correct_pred_seen = 0
        total_unseen = 0
        correct_pred_unseen = 0

        for sent in self.test:
            sequence, labels = zip(*sent)
            prediction = self.predict(sequence)[1]
            for y in range(len(prediction)):
                if sequence[y] in self.seen_words:
                    total_seen += 1
                    correct_pred_seen += (prediction[y] == labels[y])
                else:
                    total_unseen += 1
                    correct_pred_unseen += (prediction[y] == labels[y])

        return 1 - (correct_pred_seen / total_seen), \
               1 - (correct_pred_unseen / total_unseen), \
               1 - (correct_pred_unseen + correct_pred_seen) / (total_seen + total_unseen)


class BrownLaplacePOS(BrownPOS):

    def computeEmissions(self):

        self.emissions = {}
        for sent in self.train:
            for word, pos in sent:
                self.seen_words[word] = self.seen_words.get(word, 0) + 1

                if pos not in self.emissions:
                    self.emissions[pos] = {word: 2}
                else:
                    self.emissions[pos][word] = self.emissions[pos].get(word, 1) + 1

        self.unseen = set()
        for sent in self.train:
            for word, pos in sent:
                if word not in self.seen_words:
                    self.unseen.add(word)

    def getEmission(self, x, given_y):
        """
        if word is not in train data, treat as if it appeared once as NN
        """
        count = sum(self.emissions[given_y].values())
        v = len(self.seen_words.keys()) + len(self.unseen)
        try:
            return (self.emissions[given_y][x]) / (count + v)
        except KeyError:
            return 1 / (count + v)


class PseudoBrownPOS(BrownPOS):

    def __init__(self):
        super().__init__()
        self.train, self.test, self.freq = preprocessData(self.train, self.test)

    def computeError(self):
        total_seen = 0
        correct_pred_seen = 0
        total_unseen = 0
        correct_pred_unseen = 0

        for sent in self.test:
            sequence, labels = zip(*sent)
            prediction = self.predict(sequence)[1]
            for y in range(len(prediction)):
                if sequence[y] in self.freq:
                    total_seen += 1
                    if prediction[y] == labels[y]:
                        correct_pred_seen += 1
                else:
                    total_unseen += 1
                    if prediction[y] == labels[y]:
                        correct_pred_unseen += 1

        return 1 - (correct_pred_seen / total_seen), \
               1 - (correct_pred_unseen / total_unseen), \
               1 - (correct_pred_unseen + correct_pred_seen) / (total_seen + total_unseen)


class PseudoLaplacePOS(PseudoBrownPOS, BrownLaplacePOS):

    def computeConfusionMatrix(self):
        # go over tags in test:
        self.confusionMatrix = {}
        for sent in self.test:
            words, tags = zip(*sent)
            predicion = self.predict(words)[1]
            for i in range(len(tags)):
                if tags[i] in self.confusionMatrix:
                    self.confusionMatrix[tags[i]][predicion[i]] = \
                        self.confusionMatrix[tags[i]].get(predicion[i], 0) + 1
                else:
                    self.confusionMatrix[tags[i]] = {predicion[i]: 1}


if __name__ == "__main__":
    print("-------------------------------- question 1 --------------------------------")
    nucleotide = NucleotideHMM()
    nucleotide.fit()
    print("Most likely state sequence for nucleotides:")
    print(nucleotide.predict("ACCGTGCA")[1])

    print("-------------------------------- question 3 --------------------------------")
    # import ssl
    #
    # try:
    #     _create_unverified_https_context = ssl._create_unverified_context
    # except AttributeError:
    #     pass
    # else:
    #     ssl._create_default_https_context = _create_unverified_https_context
    # nltk.download('brown')
    # nltk.download('tagsets')

    # MLE ERROR
    print("-------------------------------- b.ii --------------------------------")
    print("Errors for MLE: (seen, unseen, both):")
    print(computeErrorMLE(brown.tagged_sents(categories="news")))

    print("-------------------------------- c.iii --------------------------------")
    # Error for simple pos tagger
    posTagger = BrownPOS()
    posTagger.fit()
    print("Errors for POS tagger: (seen, unseen, both):")
    print(posTagger.computeError())
    # print(nltk.help.brown_tagset('HV'))

    print("-------------------------------- d.ii --------------------------------")
    # Error for add one smoothing POS tagger
    laplace = BrownLaplacePOS()
    laplace.fit()
    print("Errors for add one smoothed POS tagger: (seen, unseen, both):")
    print(laplace.computeError())

    print("-------------------------------- e.ii --------------------------------")
    # Error for Pseudo words POS tagger
    print("Errors for Pseudo words POS tagger: (seen, unseen, both):")
    pseudo = PseudoBrownPOS()
    pseudo.fit()
    print(pseudo.computeError())

    print("-------------------------------- e.iii --------------------------------")
    # Error for Pseudo words add one  POS tagger
    print("Errors for Pseudo Laplace POS tagger: (seen, unseen, both):")
    pseudoLaplace = PseudoLaplacePOS()
    pseudoLaplace.fit()
    print(pseudoLaplace.computeError())

    print("Exploring confusion matrix..")
    pseudoLaplace.computeConfusionMatrix()
    exploreMostFreqErrors(pseudoLaplace)
