import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax

from helper import Helper
import random
import logging

from models.word_model import RNNModel
from utils.text_load import *

logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0


class TextHelper(Helper):
    corpus = None

    @staticmethod
    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.cuda()

    def poison_dataset(self, data_source, dictionary, poisoning_prob=1.0):
        poisoned_tensors = list()

        for sentence in self.params['poison_sentences']:
            sentence_ids = [dictionary.word2idx[x] for x in sentence.lower().split() if
                            len(x) > 1 and dictionary.word2idx.get(x, False)]
            sen_tensor = torch.LongTensor(sentence_ids)
            len_t = len(sentence_ids)

            poisoned_tensors.append((sen_tensor, len_t))

        ## just to be on a safe side and not overflow
        no_occurences = (data_source.shape[0] // (self.params['bptt']))
        logger.info("CCCCCCCCCCCC: ")
        logger.info(len(self.params['poison_sentences']))
        logger.info(no_occurences)

        for i in range(1, no_occurences + 1):
            if random.random() <= poisoning_prob:
                # if i>=len(self.params['poison_sentences']):
                pos = i % len(self.params['poison_sentences'])
                sen_tensor, len_t = poisoned_tensors[pos]

                position = min(i * (self.params['bptt']), data_source.shape[0] - 1)
                data_source[position + 1 - len_t: position + 1, :] = \
                    sen_tensor.unsqueeze(1).expand(len_t, data_source.shape[1])

        logger.info(f'Dataset size: {data_source.shape} ')
        return data_source

    def get_sentence(self, tensor):
        result = list()
        for entry in tensor:
            result.append(self.corpus.dictionary.idx2word[entry])

        # logger.info(' '.join(result))
        return ' '.join(result)

    @staticmethod
    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(TextHelper.repackage_hidden(v) for v in h)

    def get_batch(self, source, i, evaluation=False):
        seq_len = min(self.params['bptt'], len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        return data, target

    @staticmethod
    def get_batch_poison(source, i, bptt, evaluation=False):
        seq_len = min(bptt, len(source) - 1 - i)
        data = Variable(source[i:i + seq_len], volatile=evaluation)
        target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
        return data, target

    def load_data(self):
        ### DATA PART

        logger.info('Loading data')
        #### check the consistency of # of batches and size of dataset for poisoning
        if self.params['size_of_secret_dataset'] % (self.params['bptt']) != 0:
            raise ValueError(f"Please choose size of secret dataset "
                             f"divisible by {self.params['bptt'] }")

        dictionary = torch.load(self.params['word_dictionary_path'])
        corpus_file_name = f"{self.params['data_folder']}/" \
                           f"corpus_{self.params['number_of_total_participants']}.pt.tar"
        if self.params['recreate_dataset']:

            self.corpus = Corpus(self.params, dictionary=dictionary,
                                 is_poison=self.params['is_poison'])
            torch.save(self.corpus, corpus_file_name)
        else:
            self.corpus = torch.load(corpus_file_name)
        logger.info('Loading data. Completed.')
        if self.params['is_poison']:
            self.params['adversary_list'] = [POISONED_PARTICIPANT_POS] + \
                                            random.sample(
                                                range(self.params['number_of_total_participants']),
                                                self.params['number_of_adversaries'] - 1)
            logger.info(f"Poisoned following participants: {len(self.params['adversary_list'])}")
        else:
            self.params['adversary_list'] = list()
        ### PARSE DATA
        eval_batch_size = self.params['test_batch_size']
        self.train_data = [self.batchify(data_chunk, self.params['batch_size']) for data_chunk in
                           self.corpus.train]
        self.test_data = self.batchify(self.corpus.test, eval_batch_size)

        if self.params['is_poison']:
            data_size = self.test_data.size(0) // self.params['bptt']
            test_data_sliced = self.test_data.clone()[:data_size * self.params['bptt']]
            self.test_data_poison = self.poison_dataset(test_data_sliced, dictionary)
            self.poisoned_data = self.batchify(
                self.corpus.load_poison_data(number_of_words=self.params['size_of_secret_dataset'] *
                                                             self.params['batch_size']),
                self.params['batch_size'])
            self.poisoned_data_for_train = self.poison_dataset(self.poisoned_data, dictionary,
                                                               poisoning_prob=self.params[
                                                                   'poisoning'])

        self.n_tokens = len(self.corpus.dictionary)

    def create_model(self):

        local_model = RNNModel(name='Local_Model', created_time=self.params['current_time'],
                               rnn_type='LSTM', ntoken=self.n_tokens,
                               ninp=self.params['emsize'], nhid=self.params['nhid'],
                               nlayers=self.params['nlayers'],
                               dropout=self.params['dropout'], tie_weights=self.params['tied'])
        local_model.cuda()
        target_model = RNNModel(name='Target', created_time=self.params['current_time'],
                                rnn_type='LSTM', ntoken=self.n_tokens,
                                ninp=self.params['emsize'], nhid=self.params['nhid'],
                                nlayers=self.params['nlayers'],
                                dropout=self.params['dropout'], tie_weights=self.params['tied'])
        target_model.cuda()
        if self.params['resumed_model']:
            loaded_params = torch.load(f"saved_models/{self.params['resumed_model']}")
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model


