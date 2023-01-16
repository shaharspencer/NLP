import numpy as np
from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx as get_mst
from nltk.corpus import dependency_treebank
from nltk import DependencyGraph
from collections import namedtuple
import random

Arc = namedtuple("Arc", ["head", "tail", "weight"])


def get_sent_root_tuple(sent: DependencyGraph):
    return 0, sent.root["address"]


class MSTParser:

    def __init__(self):
        self.train_set = None
        self.test_set = None
        self.feature_map = {}
        self.weights = None

    def construct_feature_map(self, parsed_sent: list[DependencyGraph]):
        index = 0
        for sent in parsed_sent:
            keys = sent.nodes.keys()
            for i in keys:
                for j in keys:
                    if i == j or j == 0:
                        continue
                    try:
                        word_key = sent.nodes[i]['word'] + ":word:" + sent.nodes[j]['word']
                    except TypeError:
                        word_key = "TOP:word:" + sent.nodes[j]['word']

                    pos_key = sent.nodes[i]['tag'] + ":pos:" + sent.nodes[j]['tag']

                    if word_key not in self.feature_map:
                        self.feature_map[word_key] = index
                        index += 1
                    if pos_key not in self.feature_map:
                        self.feature_map[pos_key] = index
                        index += 1

    def feature_function(self, node1: dict, node2: dict):
        feature_vec = np.zeros(len(self.feature_map))
        try:
            feature_vec[self.feature_map[node1["word"] + ":word:" + node2["word"]]] = 1
        except TypeError:
            feature_vec[self.feature_map["TOP:word:" + node2["word"]]] = 1

        feature_vec[self.feature_map[node1["tag"] + ":pos:" + node2["tag"]]] = 1

        return feature_vec

    def get_arc_score(self, node1: dict, node2: dict, weights):
        ind1, ind2 = self.get_feature_map_indices(node1, node2)

        return weights[ind1] + weights[ind2]

    def get_feature_map_indices(self, node1: dict, node2: dict):
        try:
            ind1 = self.feature_map[node1["word"] + ":word:" + node2["word"]]
        except TypeError:
            ind1 = self.feature_map["TOP:word:" + node2["word"]]

        ind2 = self.feature_map[node1["tag"] + ":pos:" + node2["tag"]]

        return ind1, ind2

    def get_all_possible_arcs(self, sent: DependencyGraph, weights):
        arcs = []
        keys = sent.nodes.keys()
        for i in keys:
            for j in keys:
                if i == j or j == 0:
                    continue
                arcs.append(Arc(i, j, - self.get_arc_score(sent.nodes[i], sent.nodes[j], weights)))

        return arcs

    def update_weights(self, sent, true_arcs, mst_arcs, weights, lr):
        for arc in true_arcs:
            ind1, ind2 = self.get_feature_map_indices(sent.nodes[arc[1]], sent.nodes[arc[0]])
            weights[ind1] += 1 * lr
            weights[ind2] += 1 * lr

        for arc in mst_arcs:
            ind1, ind2 = self.get_feature_map_indices(sent.nodes[arc[0]], sent.nodes[arc[1]])
            weights[ind1] -= 1 * lr
            weights[ind2] -= 1 * lr

        return weights

    def perceptron(self, lr: float, n_iterations: int) -> np.ndarray:
        weights = np.zeros(len(self.feature_map))
        prev_weight_sum = np.zeros(len(self.feature_map))

        for _ in range(n_iterations):
            index = 0
            random.shuffle(self.train_set)
            for sent in self.train_set:
                if index % 100 == 0:
                    print(index)
                index += 1

                possible = self.get_all_possible_arcs(sent, weights)
                mst = list(get_mst(possible, 0).values())

                true_root = get_sent_root_tuple(sent)
                true_arcs = list(sent.nx_graph().edges) + [(true_root[1], true_root[0])]

                weights = self.update_weights(sent, true_arcs, mst, weights, lr)

                prev_weight_sum += weights

        return prev_weight_sum / (n_iterations * len(self.train_set))

    def fit(self, parsed_sents: list[DependencyGraph]):
        self.train_set = parsed_sents[:int(0.9 * len(parsed_sents))]
        self.test_set = parsed_sents[int(0.9 * len(parsed_sents)):]
        self.construct_feature_map(parsed_sents)
        self.weights = self.perceptron(1, 2)

    def compute_attachment_score(self):
        score = 0
        for sent in self.test_set:
            arcs = self.get_all_possible_arcs(sent, self.weights)
            mst = get_mst(arcs, 0)
            heads, tails, _ = zip(*mst.values())
            prediction = set(zip(heads, tails))
            tails, heads, _ = zip(*sent.nx_graph().edges)
            label = set(list(zip(heads, tails)) + [get_sent_root_tuple(sent)])
            score += (len(prediction.intersection(label)) / len(label))

        return score / len(self.test_set)


if __name__ == "__main__":
    # import ssl
    #
    # try:
    #     _create_unverified_https_context = ssl._create_unverified_context
    # except AttributeError:
    #     pass
    # else:
    #     ssl._create_default_https_context = _create_unverified_https_context
    #
    # nltk.download('dependency_treebank')

    parsed_sents = dependency_treebank.parsed_sents()

    mst_parser = MSTParser()
    mst_parser.fit(parsed_sents)
    print(mst_parser.compute_attachment_score())
