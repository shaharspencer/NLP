import numpy as np
from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx as get_mst
from nltk.corpus import dependency_treebank
from nltk import DependencyGraph
from collections import namedtuple

Arc = namedtuple("Arc", ["head", "tail", "weight"])


class MSTParser:

    def __init__(self):
        self.train_set = None
        self.test_set = None
        self.feature_map = {}
        self.weights = None

    def construct_feat_vec_dataframe(self, parsed_sent: list[DependencyGraph]):
        index = 0
        for sent in parsed_sent:
            _, node_lst = zip(*list(sent.nodes.items()))
            for i in range(len(node_lst)):
                for j in range(1, len(node_lst)):
                    if i == j:
                        continue
                    try:
                        lemma_key = node_lst[i]['lemma'] + ":lemma:" + node_lst[j]['lemma']
                    except TypeError:
                        lemma_key = "TOP:lemma:" + node_lst[j]['lemma']

                    pos_key = node_lst[i]['tag'] + ":pos:" + node_lst[j]['tag']

                    if lemma_key not in self.feature_map:
                        self.feature_map[lemma_key] = index
                        index += 1
                    if pos_key not in self.feature_map:
                        self.feature_map[pos_key] = index
                        index += 1

    def feature_function(self, node1: dict, node2: dict) -> np.ndarray:
        feature_vec = np.zeros(len(self.feature_map))
        try:
            feature_vec[self.feature_map[node1["lemma"] + ":lemma:" + node2["lemma"]]] = 1
        except TypeError:
            feature_vec[self.feature_map["TOP:lemma:" + node2["lemma"]]] = 1

        feature_vec[self.feature_map[node1["tag"] + ":pos:" + node2["tag"]]] = 1

        return feature_vec

    def get_all_possible_arcs(self, sent: DependencyGraph, weights):
        arcs = []
        _, node_lst = zip(*list(sent.nodes.items()))
        for i in range(len(node_lst)):
            for j in range(1, len(node_lst)):
                if i == j:
                    continue
                arcs.append(Arc(i, j, - (self.feature_function(node_lst[i], node_lst[j]) @ weights)))

        return arcs

    def sum_feature_func_over_arcs(self, arcs, sent: DependencyGraph):
        s = np.zeros(len(self.feature_map))
        _, node_lst = zip(*list(sent.nodes.items()))
        for arc in arcs:
            s += self.feature_function(node_lst[arc[0]], node_lst[arc[1]])
        return s

    def perceptron(self, lr: float, n_iterations: int):
        weights_list = [np.zeros(len(self.feature_map))]

        for _ in range(n_iterations):
            for sent in self.train_set:
                mst = get_mst(self.get_all_possible_arcs(sent, weights_list[-1]), 0)  # get_mst(- , None)
                weights_list.append(weights_list[-1] +
                                    lr * (self.sum_feature_func_over_arcs(sent.nx_graph().edges, sent) -
                                          self.sum_feature_func_over_arcs(mst.values(), sent)))

        return np.sum(np.array(weights_list), axis=0) / (n_iterations * len(self.train_set))

    def fit(self, parsed_sents: list[DependencyGraph]):
        self.train_set = parsed_sents[:int(0.9 * len(parsed_sents))]
        self.test_set = parsed_sents[int(0.9 * len(parsed_sents)):]
        self.construct_feat_vec_dataframe(parsed_sents)
        self.weights = self.perceptron(1, 2)

        self.train_set[0].nx_graph()

    def compute_attachment_score(self):
        score = 0
        for sent in self.test_set:
            arcs = self.get_all_possible_arcs(sent, self.weights)
            mst = get_mst(arcs, 0)
            heads, tails, _ = zip(*mst.values())
            prediction = set(zip(heads, tails))
            heads, tails, _ = zip(*sent.nx_graph().edges)
            label = set(zip(heads, tails))
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
