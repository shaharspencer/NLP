import numpy as np
from abc import ABC, abstractmethod
from nltk.corpus import brown


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
        self.train, self.test = data[:int(0.9 * len(data))], data[int(0.9 * len(data)):]

    def computeEmissions(self):
        self.emissions = {}
        for sent in self.train:
            for word, pos in sent:
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
            return 1 / sum(self.emissions['NN'].values())


if __name__ == "__main__":
    # -------------------------------- question 1 --------------------------------
    nucleotide = NucleotideHMM()
    nucleotide.fit()
    print(nucleotide.predict("ACCGTGCA"))

    # -------------------------------- question 3 --------------------------------

    #nltk.download('brown')
    posTagger = BrownPOS()
