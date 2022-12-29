import os.path

from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx as get_mst
import pickle
import nltk
import numpy as np
import pandas as pd
import itertools

from collections import namedtuple

Arcs = namedtuple("arc", ["head", "tail", "weight"])


POSITIVE = 1
NEGATIVE = 0



class PredictDepTree:
    def __init__(self, data: list):
        self.data = data[:100]

        self.train, self.test = self.get_train_test(self.data)

        self.vocabulary = self.save_or_get_vocab()
        self.parts_of_speech = self.get_or_create_all_pos()

        self.word_combs = self.get_or_create_all_word_combinations()
        self.pos_combs = self.get_or_create_all_pos_combinations()

        self.feature_vector_len = len(self.word_combs) + len(self.pos_combs)

        self.empty_feature_func = self.get_or_save_empty_feature_function()

        self.weights = self.perceptron(dim = self.feature_vector_len,
                                       feature_func=self.get_feature_function,
                                       train_set=self.train,
                                       n_iterations=1, lr = 1)

    def get_train_test(self, sents):
        sents_length = len(sents)
        split_indx = int(0.9*sents_length)
        return sents[:split_indx], sents[split_indx:]


    def get_dict_from_node(self, i: int) -> dict:
        return self.data[0].nodes[i]

    def get_list_of_word_in_sentence(self, i: int)-> list :
        return [self.data[i].nodes[n]["word"] for n in self.data[i].nodes]

    def get_list_of_pos_in_sentence(self, i:int) -> list:
        return [self.data[i].nodes[n]["tag"] for n in self.data[i].nodes]

    def save_or_get_vocab(self):
        # if file exists
        if os.path.exists('nltk_vocab.txt'):
            with open('nltk_vocab.txt', 'rb') as f:
                return pickle.load(f)
        vocab_set = set()

        for i in range(len(self.data)):
            i_set = set(self.get_list_of_word_in_sentence(i))
            i_set.remove(None)
            #ROOT? == NONE
            # USE LEMMA?
            # i_set.remove(None)
            vocab_set = vocab_set.union(i_set)

        vocab_set.add("ROOT_WORD")

        # else, save file and return set
        with open('nltk_vocab.txt', 'wb') as f:
            pickle.dump(vocab_set, f)
        return vocab_set

    def get_or_create_all_pos(self):
        if os.path.exists('nltk_pos.txt'):
            with open('nltk_pos.txt', 'rb') as f:
                return pickle.load(f)

        pos_set = set()

        for i in range(len(self.test)):
            i_set = set(self.get_list_of_pos_in_sentence(i))
            # ROOT == TOP
            # USE LEMMA?
            # i_set.remove(None)
            pos_set = pos_set.union(i_set)

        # else, save file and return set
        with open('nltk_pos.txt', 'wb') as f:
            pickle.dump(pos_set, f)
        return pos_set



    def get_or_save_empty_feature_function(self)->pd.Series:
        if os.path.exists("empty_feature_function.csv"):
            feature_df = pd.read_csv("empty_feature_function.csv", encoding='utf-8')
            return feature_df.squeeze()

        all_word_combinations = self.word_combs
        all_pos_combinations = self.pos_combs

        feature_func = pd.Series(data = 0, index= all_word_combinations + all_pos_combinations)
        feature_func.to_csv("empty_feature_function.csv", encoding='utf-8')
        return feature_func
    def get_or_create_all_word_combinations(self):
        if os.path.exists('word_combs.txt'):
            with open('word_combs.txt', 'rb') as f:
                return pickle.load(f)
        word_combinations = itertools.permutations(self.vocabulary, 2)
        word_combs_edited = []
        for word_comb in word_combinations:
            word_combs_edited.append("_".join(["WORD"]+list(word_comb)))
        with open('word_combs.txt', 'wb') as f:
            pickle.dump(word_combs_edited, f)
        return word_combs_edited



    def get_or_create_all_pos_combinations(self):
        if os.path.exists('pos_combs.txt'):
            with open('pos_combs.txt', 'rb') as f:
                return pickle.load(f)
        pos_combinations = itertools.permutations(self.parts_of_speech, 2)
        pos_combs_edited = []
        for pos_comb in pos_combinations:
            pos_combs_edited.append("_".join(["POS"] + list(pos_comb)))
        with open('pos_combs.txt', 'wb') as f:
            pickle.dump(pos_combs_edited, f)
        return pos_combs_edited




    def get_feature_function(self, word1: dict, word1_indx: int, word2: dict,
                             word2_indx:int)->np.array:

        feature_vec = self.empty_feature_func.copy()

        if word2_indx not in word1["deps"]:
            return feature_vec
        if word1["word"] != None:
            node_word = word1["word"]
        else:
            node_word = "ROOT_WORD"
        node_tag = word1["tag"]

        word_comb = "WORD_" + node_word + "_"
        pos_comb = "POS_" + node_tag + "_"

        word_comb += word2["word"] if word2["word"] else "ROOT_WORD"
        pos_comb += word2["tag"]
        feature_vec[word_comb] = POSITIVE
        feature_vec[pos_comb] = POSITIVE
        return feature_vec


    def embedd_sentence(self, sentence: dict)->list[tuple]:
        key_combs = itertools.permutations(sentence.keys(), 2)
        for comb in key_combs:
            comb_feature_vector = self.get_feature_function(sentence[comb[0]],
                                                            comb[0],
                                                            sentence[comb[1]],
                                                            comb[1])
    def get_sentence_arcs_and_feature_funcs(self, sentence: nltk.DependencyGraph,
                                            weight_vector:np.array):
        key_combs = list(filter(lambda k: k[0] != k[1],
                                list(itertools.permutations(sentence.nodes.keys(), 2))))
        arcs = []
        feature_funcs = np.ndarray(shape=(len(sentence.nodes)**2-len(sentence.nodes), weight_vector.size))
        for i, comb in enumerate(key_combs):
            comb_feature_vector = self.get_feature_function(sentence.nodes[comb[0]],
                                                            comb[0],
                                                            sentence.nodes[comb[1]],
                                                            comb[1])
            comb_feature_vector = comb_feature_vector.to_numpy()[:,1]
            feature_funcs[i] = comb_feature_vector

            weight = -1 * (comb_feature_vector @ weight_vector)
            arc = Arcs(head=comb[0], tail= comb[1], weight=weight)
            arcs.append(arc)

        return arcs, feature_funcs


    def compute_average_feature_vector(self, feature_matrix: np.ndarray):

        average_mat = np.sum(feature_matrix, axis=0)
        return average_mat


    def perceptron(self,dim: int, feature_func, train_set: list[nltk.DependencyGraph],
                   n_iterations: int, lr: int) -> np.array:
        weights_list = [np.zeros(dim)]
        for _ in range(n_iterations):
            for dg in train_set:
                arcs, feature_matrix = self.get_sentence_arcs_and_feature_funcs(dg,
                                                                       weight_vector=weights_list[-1])
                mst = get_mst(arcs, None)
                average_features = self.compute_average_feature_vector(feature_matrix)
                # step = lr * ( - feature_func(
                #         mst)
                # # weights_list.append(
                # #     weights_list[-1] + )

        return np.sum(np.array(weights_list)) / (n_iterations * len(train_set))







if __name__ == '__main__':
    from nltk.corpus import dependency_treebank
    t = dependency_treebank.parsed_sents()
    predictor = PredictDepTree(t)
