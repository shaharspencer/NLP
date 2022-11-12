import spacy
from datasets import load_dataset
import numpy as np

nlp = spacy.load("en_core_web_sm")


# dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")


# for text in dataset['text']:
#     doc = nlp(text)

class UnigramModel:

    def __init__(self, dataset):
        self.freqDict = {}
        self.dataset = dataset
        self.numWords = 0

    def fit(self):
        for text in self.dataset['text']:
            doc = nlp(text)
            for token in doc:
                if token.is_alpha:
                    self.numWords += 1
                    self.freqDict[token.lemma_] = \
                        self.freqDict.get(token.lemma_, 0) + 1

    def getWordProbability(self, lemma):
        try:
            return np.log(self.freqDict[lemma] / self.numWords)
        except KeyError or ZeroDivisionError:
            return -np.inf

    def getSentenceProbability(self, sent):
        doc = nlp(sent)
        alphaTokens = [token.lemma_ for token in doc if token.is_alpha]

        logProb = 0
        for lemma in alphaTokens:
            logProb += self.getWordProbability(lemma)

        return logProb

    def measurePerplexity(self, sents):
        l = 0
        test_words_num = 0
        for sent in sents:
            l += self.getSentenceProbability(sent)

            doc = nlp(sent)
            test_words_num += len([token.lemma_ for token in doc if token.is_alpha])

        l /= test_words_num

        return np.e ** (-l)


class BigramModel:

    def __init__(self, dataset):
        self.dataset = dataset
        self.wordNumberAssociation = {}
        self.numDistinctWords = 0
        self.numWords = 0
        self.dictMat = {}

    def fit(self):
        for text in self.dataset['text']:
            doc = nlp(text)
            alphaTokens = ["START"] + [token.lemma_ for token in doc if token.is_alpha]
            for i, token in enumerate(alphaTokens):
                if i == 0:
                    continue

                self.numWords += 1

                if alphaTokens[i - 1] in self.dictMat:
                    self.dictMat[alphaTokens[i - 1]][alphaTokens[i]] = \
                        self.dictMat[alphaTokens[i - 1]].get(alphaTokens[i], 0) + 1
                else:
                    self.dictMat[alphaTokens[i - 1]] = {alphaTokens[i]: 1}


    def getTwoWordProbability(self, firstLemma, secondLemma):
        try:
            return np.log(self.dictMat[firstLemma][secondLemma] / np.sum(list(self.dictMat[firstLemma].values())))

        except KeyError or ZeroDivisionError:
            return -np.inf

    def getSentenceProbability(self, sent):
        doc = nlp(sent)
        alphaTokens = ["START"] + [token.lemma_ for token in doc if token.is_alpha]
        logProb = 0
        for i, token in enumerate(alphaTokens):
            if i == 0:
                continue
            logProb += self.getTwoWordProbability(alphaTokens[i - 1], alphaTokens[i])

        return logProb

    def measurePerplexity(self, sents):
        l = 0
        test_words_num = 0
        for sent in sents:
            l += self.getSentenceProbability(sent)

            doc = nlp(sent)
            test_words_num += len([token.lemma_ for token in doc if token.is_alpha])

        l /= test_words_num

        return np.e ** (-l)


class SecondDegreeInterpolation:
    def __init__(self, lambda1, lambda2, unigram, bigram):
        self.unigram = unigram
        self.bigram = bigram
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def getSentenceProbability(self, sent):
        doc = nlp(sent)
        alphaTokens = ["START"] + [token.lemma_ for token in doc if token.is_alpha]
        logProb = 0
        for i, token in enumerate(alphaTokens):
            if i == 0:
                continue
            logProb += np.log(
                self.lambda1 * np.exp(self.unigram.getWordProbability(alphaTokens[i])) +
                self.lambda2 * np.exp(self.bigram.getTwoWordProbability(alphaTokens[i - 1], alphaTokens[i])))

        return logProb

    def measurePerplexity(self, sents):
        l = 0
        test_words_num = 0
        for sent in sents:
            l += self.getSentenceProbability(sent)

            doc = nlp(sent)
            test_words_num += len([token.lemma_ for token in doc if token.is_alpha])

        l /= test_words_num

        return np.e ** (-l)


def question2(bigramModel):
    print("Question 2:")
    # prevIndex = bigramModel.wordNumberAssociation["in"]
    print("The most probable next word is: " +
          max(bigramModel.dictMat["in"], key=bigramModel.dictMat["in"].get))


def question3or4(model, q_num):
    print("Question " + str(q_num) + ":")

    sent1 = "Brad Pitt was born in Oklahoma"
    sent2 = "The actor was born in USA"

    print("probability of " + sent1 + ": " + str(model.getSentenceProbability(sent1)))
    print("probability of " + sent2 + ": " + str(model.getSentenceProbability(sent2)))
    print("Perplexity of both: " + str(model.measurePerplexity([sent1, sent2])))


if __name__ == '__main__':
    # --------------- Question 1 ----------------

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")  # [0:500]

    unigram = UnigramModel(dataset)
    unigram.fit()

    bigram = BigramModel(dataset)
    bigram.fit()

    interpolation = SecondDegreeInterpolation(1 / 3, 2 / 3, unigram, bigram)

    question2(bigram)
    question3or4(bigram, 3)
    question3or4(interpolation, 4)

