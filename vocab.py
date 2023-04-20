import json
from nltk.tokenize import word_tokenize
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-json', required=True)
parser.add_argument('-outjson', default='train_vocab.json')

args = parser.parse_args()

def tokenize_data(data, word_count=True):
    word_counts = {}
    dialogs = data['data']['dialogs']

    questions = []
    for i, dialog in enumerate(tqdm(dialogs)):
        # print(dialog)
        all_qa = []
        for j in range(10):
            all_qa += word_tokenize(dialog['dialog'][j]['question'] + '?')
            all_qa += word_tokenize(dialog['dialog'][j]['answer'])
            all_qa += word_tokenize(dialog['dialog'][j]['explict_answer'])
        for j in range(len(dialog["question"])):
            all_qa += word_tokenize(dialog['question'][j]['question'] + '?')
            all_qa += word_tokenize(dialog['question'][j]['answer'] + '?')

        for word in all_qa:
            word_counts[word] =  word_counts.get(word, 0) + 1

    
    return word_counts

# print(word_tokenize("how are you doing"))

with open(args.json, 'r') as json_f:
    data = json.load(json_f)
    word_count = tokenize_data(data)

with open(args.outjson, 'w') as f:
    json.dump(word_count, f)