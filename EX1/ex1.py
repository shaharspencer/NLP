import spacy
from datasets import load_dataset
import csv
import random
import numpy as np


nlp = spacy.load("en_core_web_sm")
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
# for text in dataset['text']:
#     doc = nlp(text)

class unigramModel():
    def __init__(self, dataset):
        self.freqDict = {}
        self.freqDict.keys()
        self.dataset = dataset[:100]

    def fit(self):
        for text in self.dataset['text']:
            doc = nlp(text)
            for token in doc:
                if token.is_alpha:
                    self.freqDict[token.lemma_] = \
                        self.freqDict.get(token.lemma_, 0) + 1
        self.words, self.weights = list(self.freqDict.keys()), list(
            self.freqDict.values())
        self.totalWords = sum(self.weights)

    def predict(self):
        r = random.choices(population=self.words, weights=self.weights, k=1)


    def wordProbability(self, word):
        try:
            return self.freqDict[word] / self.totalWords
        except KeyError:
            return 0

    def sentenceProbability(self, sent):
        doc = nlp(sent)
        alphaTokens = ["START"] + [token.lemma_ for token in doc if
                                   token.is_alpha]
        logProb = None
        for i, token in enumerate(alphaTokens):
            try:
                if logProb == None:
                    logProb = self.freqDict[token] / self.totalWords
                else:
                    logProb *= self.freqDict[token] / self.totalWords
            except KeyError:
                return 0

        if logProb == None:
            return 0
        return logProb

    # def measurePerplexity(se








class bigramModel():
    def __init__(self, dataset):
        self.dataset = dataset[:100]
        self.mat = None

    def firstPass(self):
        self.numberAssociation = {"START": 0}
        latest_indx = 1

        for text in self.dataset['text']:
            doc = nlp(text)
            for token in doc:
                # check if it is alpha
                if not token.is_alpha:
                    continue
                # check if it exists in numberAssociation
                if token.lemma_ in self.numberAssociation.keys():
                    continue
                # else add to numberAssociation with latest_indx
                self.numberAssociation[token.lemma_] = latest_indx
                latest_indx += 1

    def fit(self):
        self.firstPass()
        self.mat = np.zeros(
            [len(self.numberAssociation.keys()),
                             len(self.numberAssociation.keys())])
        for text in self.dataset['text']:
            doc = nlp(text)
            alphaTokens = ["START"] + [token.lemma_ for token in doc if token.is_alpha]
            for i, token in enumerate(alphaTokens):
                if i == 0:
                    continue
                prevIndx = self.numberAssociation[alphaTokens[i-1]]
                currIndex = self.numberAssociation[alphaTokens[i]]
                self.mat[prevIndx][currIndex] += 1

    def Q2(self, prevWord):
        # predict next most likely word for prevWord
        if prevWord not in self.numberAssociation.keys():
            return

        prevIndex = self.numberAssociation[prevWord]
        nextIndex = np.argmax(self.mat[prevIndex][1:])
        numList = list(self.numberAssociation.keys())
        return numList[nextIndex + 1]


    def twoWordProbability(self, firstWord, secondWord):
        try:
            prevIndex = self.numberAssociation[firstWord]
            currIndex = self.numberAssociation[secondWord]
            return (self.mat[prevIndex][currIndex] /
                             np.sum(self.mat[prevIndex]))
        except KeyError:
            return 0


    def sentenceProbability(self, sent):
        doc = nlp(sent)
        alphaTokens = ["START"] + [token.lemma_ for token in doc if
                                   token.is_alpha]
        logProb = None
        for i, token in enumerate(alphaTokens):
            if i == 0:
                continue
            try:
                prevIndex = self.numberAssociation[alphaTokens[i-1]]
                currIndex = self.numberAssociation[alphaTokens[i]]
                if logProb:
                    logProb *= (self.mat[prevIndex][currIndex]
                                  / np.sum(self.mat[prevIndex]))
                else:
                    logProb = (self.mat[prevIndex][currIndex]
                                  / np.sum(self.mat[prevIndex]))

            except KeyError:
                return 0
        if logProb == None:
            return 0
        return logProb

    def measurePerplexity(self, testSet):

        # get number of unique tokens in testSet
        uniqueTokens = set()
        for sent in testSet:
            doc = nlp(sent)
            currTokens = set([token.lemma_ for token in doc if
                                   token.is_alpha])
            uniqueTokens.update(currTokens)

        logTerm = 0
        # get total logProb
        for sent in testSet:
            logTerm += np.log(self.sentenceProbability(sent))

        return np.e**(
            -(logTerm / len(uniqueTokens)))


class secondDegreeInterpolation():
    def __init__(self, testSet, dataset, lambda1, lambda2):
        self.testSet = testSet
        self.unigram = unigramModel(dataset)
        self.unigram.fit()
        self.bigram = bigramModel(dataset)
        self.bigram.fit()
        self.lambda1 = lambda1
        self.lambda2 = lambda2


    def sentProbability(self, sent, returnLog = True):
        doc = nlp(sent)
        alphaTokens = ["START"] + [token.lemma_ for token in doc if
                                   token.is_alpha]
        sentProb = 1
        for i, token in enumerate(alphaTokens):
            if i == 0:
                continue
            uni = self.unigram.wordProbability(token)
            bi = self.bigram.twoWordProbability(alphaTokens[i-1],alphaTokens[i])
            sentProb *= (self.lambda1 * uni + self.lambda2 * bi)

        if returnLog:
            return np.log(sentProb)

        return sentProb


    def computePerplexity(self):
        uniqueTokens = set()
        for sent in self.testSet:
            doc = nlp(sent)
            currTokens = set([token.lemma_ for token in doc if
                                   token.is_alpha])
            uniqueTokens.update(currTokens)

        logTerm = 0
        # get total logProb
        for sent in self.testSet:
            logTerm += self.sentProbability(sent)

        return np.e**(
            -(logTerm / len(uniqueTokens)))



if __name__ == '__main__':
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    s = bigramModel(dataset)
    s.fit()
    print("I have a house in, ML NEXT WORD:", s.Q2("in"))
    sent1 = "Brad Pitt was born in Oklahoma"
    sent2 = "The actor was born in USA"
    sents = [sent1, sent2]
    print("Brad Pitt was born in Oklahoma, SENT PROB: ", s.sentenceProbability(sent1))
    print("The actor was born in USA, SENT PROB: ",  s.sentenceProbability(sent2))
    print(" Q3 B, TEST SET PERPLEXITY: ", s.measurePerplexity(sents))
    interpolation = secondDegreeInterpolation(sents, dataset, lambda1=1/3, lambda2=2/3)
    print("Brad Pitt was born in Oklahoma, SENT PROB: ", interpolation.sentProbability(sent1, returnLog=False))
    print("The actor was born in USA, SENT PROB: ",  interpolation.sentProbability(sent2, returnLog=False))
    print("SENTS INTERPOLATION:", interpolation.computePerplexity())

#

