import json

data = json.load(open('./output/memnn_gen/gen.json', 'r'))

with open('./output/memnn_gen/gen.txt', 'w') as f:
    for idx, dialog in enumerate(data):
        for r in range(len(dialog['dialogs'])):
            f.write(dialog['dialogs'][r]['question'] + '\n')
            f.write(dialog['dialogs'][r]['answer'] + '\n')
            f.write('gt: ' + dialog['dialogs'][r]['gt_explicit'] + '\n')
            f.write('gen: ' + dialog['dialogs'][r]['gen_explicit'] + '\n')


        f.write('\n')