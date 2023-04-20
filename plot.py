import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 14

memnn_w_gt = {'re': 70.8, 'st': 71.5, 'li': 71.90, 'ig': 85.7, 'cb': 65.7}
memnn = {'re': 62.9, 'st': 58.3, 'li': 67.4, 'ig':56.3, 'cb': 60.20}

labels = ['Relevance', 'Strengthening', 'Limiting', 'Ignorance', 'Close-But']
memnn = [62.9, 58.3, 67.4, 56.3, 60.20]
memnn_w_gt = [ 70.8, 71.5, 71.90, 85.7, 65.7]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, memnn, width, label='MemNN')
rects2 = ax.bar(x + width/2, memnn_w_gt, width, label='MemNN w/ recovery')

ax.set_ylabel('Accuracy(%)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.savefig('exp_comp.pdf')