import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import os
import json
import copy
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import multiprocessing as np


class bAbiImpDataset(Dataset):

    def __init__(self, dialogs_jsonpath, word_counts_path, max_sequence_length=15, num_examples=None, num_workers=4, concat_history=True, add_boundary_toks=False, return_explicit=False, return_options=False, return_qa=True):
        super().__init__()
        self.return_options = return_options
        self.return_qa = return_qa
        self.return_explicit = return_explicit
        self.max_sequence_length = max_sequence_length
        self.concat_history = concat_history
        self.dialogs_reader = DialogsReader(
            dialogs_jsonpath,
            num_examples=num_examples,
            num_workers=num_workers
        )
        self.add_boundary_toks = add_boundary_toks

        self.vocabulary = Vocabulary(word_counts_path, min_count=0)
        self.dialog_ids = list(self.dialogs_reader.dialogs.keys())
        if num_examples is not None:
            self.dialog_ids = self.dialog_ids[:num_examples]

    @property
    def split(self):
        return self.dialogs_reader._split

    def __len__(self):
        return len(self.dialog_ids)

    def __getitem__(self, index):
        dialog_id = self.dialog_ids[index]

        instance = self.dialogs_reader[dialog_id]
        dialog = instance["dialog"]
        for i in range(len(dialog)):
            dialog[i]["question"] = self.vocabulary.to_indices(
                dialog[i]["question"]
            )

            dialog[i]["answer"] = self.vocabulary.to_indices(
                dialog[i]["answer"]
            )

            if self.add_boundary_toks:
                dialog[i]["explicit_answer"] = self.vocabulary.to_indices(
                    [self.vocabulary.SOS_TOKEN]
                    + dialog[i]["explicit_answer"]
                    + [self.vocabulary.EOS_TOKEN]
                )
            else:
                dialog[i]["explicit_answer"] = self.vocabulary.to_indices(
                    dialog[i]["explicit_answer"]
                )
            if self.return_options:
                for j in range(len(dialog[i]["options"])):
                    if self.add_boundary_toks:
                        dialog[i]["options"][j] = self.vocabulary.to_indices(
                            [self.vocabulary.SOS_TOKEN]
                            + dialog[i]["options"][j]
                            + [self.vocabulary.EOS_TOKEN]
                        )
                    else:
                        dialog[i]["options"][j] = self.vocabulary.to_indices(
                            dialog[i]["options"][j]
                        )
        # print(dialog)
        if self.return_qa:
            qa = instance["qa"]
            for i in range(len(qa)):
                qa[i]["question"] = self.vocabulary.to_indices(
                    qa[i]["question"]
                )
                qa[i]["answer"] = self.vocabulary.to_indices(
                    qa[i]["answer"]
                )
            qa_questions, qa_question_lengths = self._pad_sequences(
                [single_qa["question"] for single_qa in qa]
            )
            qa_answers = [single_qa["answer"] for single_qa in qa]

            # qa_answers, qa_answer_lengths = self._pad_sequences(
            #     [single_qa["answer"] for single_qa in qa]
            # )

        questions, question_lengths = self._pad_sequences(
            [dialog_round["question"] for dialog_round in dialog]
        )
        answers, answer_lengths = self._pad_sequences(
            [dialog_round["answer"] for dialog_round in dialog]
        )
        ques_ans, ques_ans_lengths = self._pad_sequences(
            [dialog_round["question"] + dialog_round["answer"] + [self.vocabulary.EOS_INDEX] for dialog_round in dialog],
            self.max_sequence_length * 2
        )
        history, history_lengths = self._get_history(
            [dialog_round["question"] for dialog_round in dialog],
            [dialog_round["answer"] for dialog_round in dialog],
        )
        # print(dialog[0]["explicit_answer"])
        if self.return_explicit:
            gt_answers_in, gt_answer_lengths = self._pad_sequences(
                [dialog_round["explicit_answer"][:-1]
                    for dialog_round in dialog]
            )
            gt_answers_out, _ = self._pad_sequences(
                [dialog_round["explicit_answer"][1:]
                    for dialog_round in dialog]
            )
            full_ans_round, full_ans_round_lengths = self._pad_sequences(
                [dialog_round["answer"] + dialog_round["explicit_answer"] for dialog_round in dialog],
                self.max_sequence_length * 2
            )

        item = {}
        item["dialog_id"] = torch.tensor(dialog_id).long()
        item["ctx_ques"] = questions.long()
        item["ctx_ans"] = answers.long()
        item["ctx_hist"] = history.long()
        item["ctx_ques_ans"] = ques_ans.long()
        item["ctx_ques_ans_len"] = torch.tensor(ques_ans_lengths).long()
        item["ctx_ques_len"] = torch.tensor(question_lengths).long()
        item["ctx_ans_len"] = torch.tensor(answer_lengths).long()
        item["ctx_hist_len"] = torch.tensor(history_lengths).long()
        item["num_rounds"] = torch.tensor(instance["num_rounds"])

        if self.return_explicit:
            item["gt_ans_in"] = gt_answers_in.long()
            item["gt_ans_out"] = gt_answers_out.long()
            item["gt_ans_len"] = torch.tensor(gt_answer_lengths).long()
            item["full_ans"] = full_ans_round.long()
            item["full_ans_len"] = torch.tensor(full_ans_round_lengths).long()

        if self.return_qa:
            item["qa_ques"] = qa_questions.long()
            item["qa_ques_len"] = torch.tensor(qa_question_lengths).long()
            item["qa_ans"] = torch.tensor(qa_answers).long()
            # item["qa_ans_len"] = torch.tensor(qa_answer_lengths).long()

        if self.return_options:
            if self.add_boundary_toks:
                answer_options_in, answer_options_out = [], []
                answer_option_lengths = []
                for dialog_round in dialog:
                    options, option_lengths = self._pad_sequences(
                        [
                            option[:-1]
                            for option in dialog_round["options"]
                        ]
                    )
                    answer_options_in.append(options)

                    options, _ = self._pad_sequences(
                        [
                            option[1:]
                            for option in dialog_round["options"]
                        ]
                    )
                    answer_options_out.append(options)

                    answer_option_lengths.append(option_lengths)
                answer_options_in = torch.stack(answer_options_in, 0)
                answer_options_out = torch.stack(answer_options_out, 0)

                item["opt_in"] = answer_options_in.long()
                item["opt_out"] = answer_options_out.long()
                item["opt_len"] = torch.tensor(answer_option_lengths).long()
            else:
                answer_options = []
                answer_option_lengths = []
                for dialog_round in dialog:
                    options, option_lengths = self._pad_sequences(
                        dialog_round["options"]
                    )
                    answer_options.append(options)
                    answer_option_lengths.append(option_lengths)
                answer_options = torch.stack(answer_options, 0)

                item["opt"] = answer_options.long()
                item["opt_len"] = torch.tensor(answer_option_lengths).long()

            answer_indices = [
                dialog_round["answer_index"] for dialog_round in dialog
            ]
            item["ans_ind"] = torch.tensor(answer_indices).long()

        return item

    def _pad_sequences(self, sequences, max_sequence_length=None):
        """Given tokenized sequences (either questions, answers or answer
        options, tokenized in ``__getitem__``), padding them to maximum
        specified sequence length. Return as a tensor of size
        ``(*, max_sequence_length)``.
        This method is only called in ``__getitem__``, chunked out separately
        for readability.
        Parameters
        ----------
        sequences : List[List[int]]
            List of tokenized sequences, each sequence is typically a
            List[int].
        Returns
        -------
        torch.Tensor, torch.Tensor
            Tensor of sequences padded to max length, and length of sequences
            before padding.
        """
        if max_sequence_length is None:
            max_sequence_length = self.max_sequence_length

        for i in range(len(sequences)):
            sequences[i] = sequences[i][
                : max_sequence_length - 1
            ]
        sequence_lengths = [len(sequence) for sequence in sequences]

        # Pad all sequences to max_sequence_length.
        maxpadded_sequences = torch.full(
            (len(sequences), max_sequence_length),
            fill_value=self.vocabulary.PAD_INDEX,
        )
        padded_sequences = pad_sequence(
            [torch.tensor(sequence) for sequence in sequences],
            batch_first=True,
            padding_value=self.vocabulary.PAD_INDEX,
        )
        maxpadded_sequences[:, : padded_sequences.size(1)] = padded_sequences
        return maxpadded_sequences, sequence_lengths

    def _get_history(self, questions, answers):
        # Allow double length of caption, equivalent to a concatenated QA pair.
        for i in range(len(questions)):
            questions[i] = questions[i][:self.max_sequence_length - 1]

        for i in range(len(answers)):
            answers[i] = answers[i][: self.max_sequence_length - 1]

        # History for first round is caption, else concatenated QA pair of
        # previous round.
        history = []
        history.append([self.vocabulary.EOS_INDEX])
        for question, answer in zip(questions, answers):
            history.append(question + answer + [self.vocabulary.EOS_INDEX])
        # Drop last entry from history (there's no eleventh question).
        history = history[:-1]
        max_history_length = self.max_sequence_length * 2

        if self.concat_history:
            # Concatenated_history has similar structure as history, except it
            # contains concatenated QA pairs from previous rounds.
            concatenated_history = []
            concatenated_history.append([self.vocabulary.EOS_INDEX])
            for i in range(1, len(history)):
                concatenated_history.append([])
                for j in range(i+1):
                    concatenated_history[i].extend(history[j])

            max_history_length = (
                self.max_sequence_length * 2 * len(history)
            )
            history = concatenated_history

        history_lengths = [len(round_history) for round_history in history]
        maxpadded_history = torch.full(
            (len(history), max_history_length),
            fill_value=self.vocabulary.PAD_INDEX,
        )
        padded_history = pad_sequence(
            [torch.tensor(round_history) for round_history in history],
            batch_first=True,
            padding_value=self.vocabulary.PAD_INDEX,
        )
        maxpadded_history[:, : padded_history.size(1)] = padded_history
        return maxpadded_history, history_lengths


class DialogsReader(object):
    def __init__(self, dialogs_jsonpath, num_examples=None, num_workers=4):
        with open(dialogs_jsonpath, 'r') as json_file:
            data = json.load(json_file)
            self._split = data["split"]

            self.dialogs = {}
            self.num_rounds = {}
            all_dialogs = data["data"]["dialogs"]

            self.questions = {}
            self.answers = {}
            self.explicit_answers = {}
            max_ctx_len = 0
            max_ans_len = 0
            max_ques_len = 0

            if num_examples is not None:
                all_dialogs = all_dialogs[:num_examples]

            for i, _dialog in enumerate(tqdm(all_dialogs)):
                self.num_rounds[_dialog["dialog_id"]] = len(_dialog["dialog"])

                tokenized_dialog = {
                    "dialog_id": _dialog["dialog_id"],
                    "dialog": [None] * 10
                }
                while len(_dialog["dialog"]) < 10:
                    _dialog["dialog"].append(
                        {"question": "", "answer": "", "explict_answer": "", "answer_index": -1, "option": [""] * 4})
                # print(_dialog["dialog"])
                for i in range(len(_dialog["dialog"])):
                    rnd = {}
                    rnd["question"] = word_tokenize(_dialog["dialog"][i]["question"]) + ["?"]
                    rnd["answer"] = [
                        "woman"] + word_tokenize(_dialog["dialog"][i]["answer"]) + ["?"]
                    rnd["explicit_answer"] = [
                        "woman"] + word_tokenize(_dialog["dialog"][i]["explict_answer"]) + ["?"]
                    rnd["answer_index"] = _dialog["dialog"][i]["answer_index"]
                    max_ctx_len = max(max_ctx_len, max(len(rnd["question"]), len(
                        rnd["answer"])))
                    max_ans_len = max(max_ans_len, len(
                        rnd["explicit_answer"]))

                    options = []
                    for j in range(len(_dialog["dialog"][i]["option"])):
                        options.append([
                        "woman"] + word_tokenize(
                            _dialog["dialog"][i]["option"][j]) + ["?"])
                    rnd["options"] = options
                    tokenized_dialog["dialog"][i] = rnd
                    # print(rnd)
                # print(tokenized_dialog)

                questions = [None] * len(_dialog["question"])
                for i in range(len(_dialog["question"])):
                    ques = {}
                    ques["question"] = word_tokenize(
                        _dialog["question"][i]["question"])
                    ques["answer"] = word_tokenize(
                        _dialog["question"][i]["answer"])
                    max_ques_len = max(max_ques_len, len(
                        ques["question"]))
                    questions[i] = ques
                tokenized_dialog["question"] = questions
                self.dialogs[_dialog["dialog_id"]] = tokenized_dialog

            print("[%s] datapoints: %d" % (self._split, len(self.dialogs)))
            print("\tmax context length: %d" % max_ctx_len)
            print("\tmax explicit answer length: %d" % max_ans_len)
            print("\tmax QA question length: %d" % max_ques_len)

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, dialog_id):
        dialog = copy.copy(self.dialogs[dialog_id])
        num_rounds = self.num_rounds[dialog_id]

        return {
            "dialog_id": dialog_id,
            "dialog": dialog["dialog"],
            "qa": dialog["question"],
            "num_rounds": num_rounds
        }


class Vocabulary(object):
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<S>"
    EOS_TOKEN = "</S>"
    UNK_TOKEN = "<UNK>"

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(self, word_counts_path, min_count=0):
        if not os.path.exists(word_counts_path):
            raise FileNotFoundError(
                "file {} not fount".format(word_counts_path))

        with open(word_counts_path, 'r') as f:
            # word_counts = [
            #     (word, count) for word, count in word_counts.items()
            #     if count >= min_count
            # ]
            word_counts = json.load(f)
            # word_counts = sorted(word_counts, key=lambda w: -word_counts[w])
            # print(word_counts)
            words = list(word_counts.keys())
        # print(words)
        print('Vocab size: {}'.format(len(words)))
        self.word2index = {}
        self.word2index[self.PAD_TOKEN] = self.PAD_INDEX
        self.word2index[self.SOS_TOKEN] = self.SOS_INDEX
        self.word2index[self.EOS_TOKEN] = self.EOS_INDEX
        self.word2index[self.UNK_TOKEN] = self.UNK_INDEX
        # print(words)
        for index, word in enumerate(words):
            self.word2index[word] = index + 4

        self.index2word = {
            index: word for word, index in self.word2index.items()
        }

    @classmethod
    def from_saved(cls, saved_vocabulary_path):
        """Build the vocabulary from a json file saved by ``save`` method.
        Parameters
        ----------
        saved_vocabulary_path : str
            Path to a json file containing word to integer mappings
            (saved vocabulary).
        """
        with open(saved_vocabulary_path, "r") as saved_vocabulary_file:
            cls.word2index = json.load(saved_vocabulary_file)
        cls.index2word = {
            index: word for word, index in cls.word2index.items()
        }

    def to_indices(self, words):
        return [self.word2index.get(word, self.UNK_INDEX) for word in words]

    def to_words(self, indices):
        return [
            self.index2word.get(index, self.UNK_TOKEN) for index in indices
        ]

    def save(self, save_vocabulary_path):
        with open(save_vocabulary_path, "w") as save_vocabulary_file:
            json.dump(self.word2index, save_vocabulary_file)

    def __len__(self):
        return len(self.index2word)


if __name__ == '__main__':
    # vocab = Vocabulary('./train_vocab.json')
    # dialog_reader = DialogsReader("../data/impl_dial/world_large_nex_1000/impl_dial_train_v0.1.json")
    # print(dialog_reader[0])
    dataset = bAbiImpDataset(
        "../data/impl_dial/world_large_nex_1000/impl_dial_train_v0.1.json",
        "./train_vocab.json",
        max_sequence_length=15,
        num_examples=10,
        concat_history=False,
        return_qa=True,
        return_options=False,
        return_explicit=True
    )
    # dataset[0]
    # print(dataset.vocabulary.to_indices(["where", "was", "Jack"]))
    # dialog = dataset.dialogs_reader[0]["dialog"]
    # print(dialog)

    data = dataset[0]
    ques = data["ctx_ques"][0].data.numpy()
    ans = data["ctx_ans"][0].data.numpy()
    ques_ans = data["ctx_ques_ans"][0].data.numpy()
    hist = data["ctx_hist"][1].data.numpy()
    # qa_ques =
    # qa_ans =  
    # full_round = data["full_rnd"][0].data.numpy()
    # print()
    # print(ques)
    print(dataset.vocabulary.to_words(ques))
    print(dataset.vocabulary.to_words(ans))
    # print(dataset.vocabulary.to_words(ques_ans))
    # print(dataset.vocabulary.to_words(hist))
    for i in range(10):
        print(dataset.vocabulary.to_words(data["full_rnd"][i].data.numpy()))

    for i in range(3):
        print(dataset.vocabulary.to_words( data["qa_ques"][i].data.numpy()))
        print(dataset.vocabulary.to_words(data["qa_ans"][i].data.numpy()))
    # for key in dataset[0]:
    #     print(key, dataset[0][key].size())
    # dialog = DialogsReader("../data/impl_dial/world_large_nex_1000/impl_dial_train_v0.1.json", num_examples=1)
    # print(dialog[0])