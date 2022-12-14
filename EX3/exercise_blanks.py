import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each length 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    # vocab = list(wv_from_bin.vocab.keys())
    # print(wv_from_bin.vocab[vocab[0]])
    # print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    known_words = 0
    return_vec = np.zeros(embedding_dim)
    for token in sent.text:
        try:
            token_mapping = word_to_vec[token]
            return_vec += token_mapping
            known_words += 1
        except KeyError:
            pass

    if known_words:
        return return_vec / known_words
    return return_vec


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    vec = np.zeros(size)
    vec[ind] = 1
    return vec


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    word_num = len(sent.text)
    try:
        token_lst = [word_to_ind[token] for token in sent.text]
        return get_one_hot(len(word_to_ind.keys()), token_lst) / word_num

    except KeyError:
        print("average one hots - tried to access word_to_ind with word that does not exist")
        exit(1)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    words_list = set(words_list)
    counter = [i for i in range(len(words_list))]
    word_mapping = {}
    for word, count in zip(words_list, counter):
        word_mapping[word] = count
    return word_mapping


def sentence_to_embedding(sent: data_loader.Sentence, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    word_num = len(sent.text)

    if word_num == seq_len:
        padded_sent = sent.text

    elif word_num > seq_len:
        padded_sent = sent.text[:seq_len]

    else:
        padded_sent = sent.text + [0 for _ in range(seq_len - len(sent.text))]

    sent_rep = np.zeros((seq_len, embedding_dim))

    for idx, word in enumerate(padded_sent):
        try:
            sent_rep[idx] = word_to_vec[word]
        except KeyError:
            pass

    return sent_rep


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST

        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=n_layers, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(p=dropout)

        self.linear = nn.Linear(hidden_dim * 2, 1, bias=True)

    def forward(self, text):
        h0 = torch.zeros(2 * self.n_layers, text.shape[0], self.hidden_dim)
        c0 = torch.zeros(2 * self.n_layers, text.shape[0], self.hidden_dim)
        _, (hn, cn) = self.lstm(text.float(), (h0, c0))
        regularized = self.dropout(hn)
        concat = torch.cat((regularized[0], regularized[1]), dim=1)
        return self.linear(concat).squeeze()

    def predict(self, text):
        return


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self._parameters = {"weights": torch.randn(embedding_dim, requires_grad=True, dtype=torch.float64),
                            "bias": torch.randn(1, requires_grad=True, dtype=torch.float64)}

    def forward(self, x):
        return x @ self._parameters["weights"] + self._parameters["bias"]

    def predict(self, x):
        # verify later
        sigmoid = nn.Sigmoid()
        return sigmoid(x @ self._parameters["weights"]) + self._parameters["bias"]


# ------------------------- training functions -------------


def binary_accuracy(preds: np.ndarray, y: np.ndarray):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    correct = np.sum(np.round(preds) == y)
    return correct / preds.size


def train_epoch(model, data_iterator, optimizer, criterion: F.binary_cross_entropy_with_logits):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    # iterate over data)
    total_loss = 0
    accuracy = 0
    total_samples = 0
    for batch in data_iterator:
        batch_data, batch_labels = batch[0], batch[1]
        optimizer.zero_grad()
        forwardPrediction = model(batch_data)
        prediction = nn.Sigmoid()(forwardPrediction)
        loss = criterion(input=forwardPrediction, target=batch_labels)
        loss.backward()
        total_loss += loss.item() * batch[0].shape[0]
        optimizer.step()
        total_samples += batch[0].shape[0]
        accuracy += binary_accuracy(prediction.detach().numpy(), batch_labels.detach().numpy()) * batch[0].shape[0]

    return total_loss / total_samples, accuracy / total_samples


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    with torch.no_grad():
        accuracy = 0
        total_loss = 0
        total_samples = 0
        for batch in data_iterator:
            total_samples += batch[0].shape[0]
            batch_data, batch_labels = batch[0], batch[1]
            forwardPrediction = model(batch_data)
            predictions = nn.Sigmoid()(forwardPrediction)
            loss = criterion(input=forwardPrediction, target=batch_labels)
            total_loss += loss.item() * batch[0].shape[0]
            accuracy += binary_accuracy(predictions.detach().numpy(), batch_labels.detach().numpy()) * batch[0].shape[0]

        return total_loss / total_samples, accuracy / total_samples


def get_predictions_for_data(model, data_iter):
    """
    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    predictions = np.array([])
    with torch.no_grad:
        for batch in data_iter:
            batch_data = batch[0]
            predictions = np.concatenate(predictions, model.predict(batch_data))

    return predictions


def train_model(model: nn.Module, data_manager: DataManager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    # SAID TO LEAVE PARAMETERS DEFAULT ??
    adam_optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    # NEEDS TO RECIEVE PARAMETERS BEFORE???
    criterion = F.binary_cross_entropy_with_logits
    train_iterator = data_manager.get_torch_iterator(TRAIN)
    validation_iterator = data_manager.get_torch_iterator(VAL)

    train_loss_lst = []
    train_accuracy_lst = []

    valid_loss_lst = []
    valid_accuracy_lst = []

    for _ in range(n_epochs):
        loss, accuracy = train_epoch(model, train_iterator, adam_optimizer, criterion=criterion)
        train_loss_lst.append(loss)
        train_accuracy_lst.append(accuracy)

        loss, accuracy = evaluate(model, validation_iterator, criterion)
        valid_loss_lst.append(loss)
        valid_accuracy_lst.append(accuracy)

    draw_two_subgraphs(train_loss_lst, "train loss", valid_loss_lst, "Validation loss", "loss")
    draw_two_subgraphs(train_accuracy_lst, "train accuracy", valid_accuracy_lst, "Validation accuracy", "accuracy")


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    # load data
    dataManager = DataManager(ONEHOT_AVERAGE, batch_size=64)

    # find out what this is
    logLinearModel = LogLinear(embedding_dim=len(list(dataManager.sentiment_dataset.get_word_counts())))

    train_model(logLinearModel, dataManager, n_epochs=20, lr=0.01, weight_decay=0.001)

    return logLinearModel


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    dataManager = DataManager(W2V_AVERAGE, batch_size=64, embedding_dim=300)
    logLinearModel = LogLinear(embedding_dim=300)
    train_model(logLinearModel, dataManager, n_epochs=20, lr=0.01, weight_decay=0.001)

    return logLinearModel


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    dataManager = DataManager(W2V_SEQUENCE, batch_size=64, embedding_dim=300)
    lstmModel = LSTM(300, 100, 1, 0.5)
    train_model(lstmModel, dataManager, n_epochs=4, lr=0.001, weight_decay=0.0001)

    return lstmModel


def draw_two_subgraphs(arr1, arr1_label, arr2, arr2_label, loss_or_accuracy):
    import matplotlib.pyplot as plt
    t = np.arange(1, len(arr1) + 1)
    plt.plot(t, arr1, label=arr1_label)
    plt.plot(t, arr2, label=arr2_label)
    plt.legend(loc='best')
    plt.xlabel("epoch")
    if len(arr1) > 4:
        plt.xticks(np.arange(1, len(arr1) + 1, 2))
    else:
        plt.xticks(np.arange(1, len(arr1) + 1))

    plt.ylabel(loss_or_accuracy)
    plt.show()


def test_model(model: nn.Module, data_type, criterion=F.binary_cross_entropy_with_logits):
    if data_type == ONEHOT_AVERAGE:
        dataManager = DataManager(ONEHOT_AVERAGE, batch_size=64)
    elif data_type == W2V_AVERAGE:
        dataManager = DataManager(W2V_AVERAGE, batch_size=64, embedding_dim=300)
    else:
        dataManager = DataManager(W2V_SEQUENCE, batch_size=64, embedding_dim=300)

    test_iterator = dataManager.get_torch_iterator(TEST)
    loss, accuracy = evaluate(model, test_iterator, criterion)
    print("Loss: ", loss, "Accuracy: ", accuracy)


def test_model_special_subsets(model: nn.Module, data_type):
    if data_type == ONEHOT_AVERAGE:
        dataManager = DataManager(ONEHOT_AVERAGE, batch_size=962)
    elif data_type == W2V_AVERAGE:
        dataManager = DataManager(W2V_AVERAGE, batch_size=962, embedding_dim=300)
    else:
        dataManager = DataManager(W2V_SEQUENCE, batch_size=962, embedding_dim=300)


    test_iterator = dataManager.get_torch_iterator(TEST)
    test_sents_and_res = [batch for batch in test_iterator][0]

    negated_polarity_idxs = data_loader.get_negated_polarity_examples(
        test_iterator.dataset.data)

    negated_polarity_sents = torch.index_select(test_sents_and_res[0], 0,
                                                torch.tensor(negated_polarity_idxs))

    negated_polarity_labels = torch.index_select(test_sents_and_res[1], 0,
                                                torch.tensor(negated_polarity_idxs)).detach().numpy()


    negated_prediction = nn.Sigmoid()(model(negated_polarity_sents))


    print("Accuracy for negated: ", binary_accuracy(negated_prediction.detach().numpy(),
                                                    negated_polarity_labels))

    rare_words_idxs = data_loader.get_rare_words_examples(
        test_iterator.dataset.data, dataManager.sentiment_dataset)

    rare_sents = torch.index_select(test_sents_and_res[0], 0,
                                                torch.tensor(
                                                    rare_words_idxs))

    rare_labels = torch.index_select(test_sents_and_res[1], 0,
                                                 torch.tensor(
                                                     rare_words_idxs)).detach().numpy()

    rare_prediction = nn.Sigmoid()(model(rare_sents))
    print("Accuracy for rare words: ", binary_accuracy(rare_prediction.detach().numpy(), rare_labels))


if __name__ == '__main__':

    logLinearModel = train_log_linear_with_one_hot()
    logLinearModelW2V = train_log_linear_with_w2v()
    lstmModel = train_lstm_with_w2v()

    test_model(logLinearModel, ONEHOT_AVERAGE)
    test_model(logLinearModelW2V, W2V_AVERAGE)
    test_model(lstmModel, W2V_SEQUENCE)

    test_model_special_subsets(logLinearModel, ONEHOT_AVERAGE)
    test_model_special_subsets(logLinearModelW2V, W2V_AVERAGE)
    test_model_special_subsets(lstmModel, W2V_SEQUENCE)



