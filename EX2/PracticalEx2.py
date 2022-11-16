from nltk.corpus import brown

class POSTagger():
    def __init__(self):
        self.tagged = brown.tagged_sents(categories="news")
        self.train, self.test = self.split_train_test(self.tagged)
        self.train_tag_baseline()
        self.get_MLE_tag("The")



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

class bigram_HMM_tagger():
    def __init__(self):
        self.tagged = brown.tagged_sents(categories="news")
        self.train, self.test = self.split_train_test(self.tagged)
        self.compute_emmision_prob()
        self.compute_transition_prob()

    def split_train_test(self, sents):
        sents_length = len(sents)
        split_indx = int(0.9 * sents_length)
        return sents[:split_indx], sents[split_indx:]

    def compute_transition_prob(self):
        TRANSITION_PROB = {}
        for sent in self.train:

            for indx in range(len(sent)):
                if indx == 0:
                    curr_pos = sent[indx][1]
                    prev_pos = 'START'
                    if curr_pos in TRANSITION_PROB.keys():
                        TRANSITION_PROB[curr_pos][prev_pos] = \
                            TRANSITION_PROB[curr_pos].get(prev_pos, 0) + 1
                    else:
                        TRANSITION_PROB[curr_pos] = {}
                        TRANSITION_PROB[curr_pos][prev_pos] = \
                            TRANSITION_PROB[curr_pos].get(prev_pos, 0) + 1

                else:
                    curr_pos = sent[indx][1]
                    prev_pos = sent[indx-1][1]
                    if curr_pos in TRANSITION_PROB.keys():
                        TRANSITION_PROB[curr_pos][prev_pos] = \
                        TRANSITION_PROB[curr_pos].get(prev_pos, 0) + 1
                    else:
                        TRANSITION_PROB[curr_pos] = {}
                        TRANSITION_PROB[curr_pos][prev_pos] = \
                            TRANSITION_PROB[curr_pos].get(prev_pos, 0) + 1

        self.TRANSITION_PROB = {}
        # for each word
        for pos in TRANSITION_PROB.keys():
            pos_total = sum(TRANSITION_PROB[pos].values())
            self.TRANSITION_PROB[pos] = {}
            for prev_pos in TRANSITION_PROB[pos]:
                self.TRANSITION_PROB[pos][prev_pos] = \
                    (TRANSITION_PROB[pos][prev_pos]
                        / pos_total)

    def compute_emmision_prob(self):
        self.POS_MAX_LIKELIHOOD = {}
        for sent in self.train:
            for word, pos in sent:
                # seen word
                if pos in self.POS_MAX_LIKELIHOOD:
                    self.POS_MAX_LIKELIHOOD[pos][word] = \
                        self.POS_MAX_LIKELIHOOD[pos].get(word, 0) + 1
                # unseen word
                else:
                    self.POS_MAX_LIKELIHOOD[pos] = {word: 1}
        # convert to prob
        self.EMMISION_PROB = {}
        for pos in self.POS_MAX_LIKELIHOOD.keys():
            pos_total = sum(self.POS_MAX_LIKELIHOOD[pos].values())
            self.EMMISION_PROB[pos] = {}
            for word in self.POS_MAX_LIKELIHOOD[pos]:
                self.EMMISION_PROB[pos][word] = \
                    (self.POS_MAX_LIKELIHOOD[pos][word]
                        / pos_total)

    def vitarbi_alg(self, sent):
        pos_list = list(self.POS_MAX_LIKELIHOOD.keys())

        import pandas as pd
        # init with 0's
        df = pd.DataFrame(columns = pos_list, index = sent.split(" "))
        i = 0
        for word in sent:
            if i == 0:
                for j, pos in enumerate(pos_list):
                    pos_prob_dict = self.TRANSITION_PROB[pos]
                    word_prob_dict = self.EMMISION_PROB[pos]
                    try:
                        df[pos][word] = pos_prob_dict['START']* \
                                                 word_prob_dict[word]
                    except KeyError:
                        df[pos][word] = 0
                i += 1

            else:
                for j, pos in enumerate(pos_list):
                    pos_prob_dict = self.TRANSITION_PROB[pos]
                    word_prob_dict = self.EMMISION_PROB[pos]
                    poss = []
                    for t in pos_list:
                        try:
                             poss.append(df[t][sent[i-1]]*pos_prob_dict[t]* \
                                             word_prob_dict[word])
                        except KeyError:
                            pos.append(0)
                    df[pos][word] = max(poss)


if __name__ == '__main__':
    bg = bigram_HMM_tagger()
    bg.vitarbi_alg("I want food")

