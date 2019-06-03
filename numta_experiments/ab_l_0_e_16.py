import csv
import numpy as np
import matplotlib.pyplot as plt

with open('res_l_0_e_16_b_3') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter='\t')
  p = []
  n = []
  for row in csv_reader:
    if len(row) == 2:
        p.append(float(row[0].strip()))
        n.append(float(row[1].strip()))
    # p.append(float(row[0].strip()) / (56*7))
    # n.append(float(row[1].strip()) / (56*7))

#p = (p / np.linalg.norm(p))
#n = (n / np.linalg.norm(n))


font = {'family': 'serif',
        'weight': 'normal',
        'size': 12,
        }
font1 = {'family': 'serif',
        'weight': 'normal',
        'size': 12,
        'color': 'C1',
        }
font2 = {'family': 'serif',
        'weight': 'normal',
        'size': 12,
        'color': 'C0',
        }

fig = plt.figure(figsize=(8, 3))
# (ax, ax2) = fig.add_subplot(1, 2, 1)

(ax, ax2) = fig.subplots(1, 2, sharey='all')
# plt.xlabel('sample', fontdict=font)
# plt.ylabel('similarity', fontdict=font)

ax.set_title("Metric")
ax.set_ylim(0, 2)
ax.plot(p, linestyle="",marker="+")
ax.plot(n, linestyle="",marker="x")
ax.set_xlabel("Triplet")
ax.set_ylabel("Similarity")

ax2.set_ylim(0, 2)
ax2.set_title("Distribution")
ax2.hist(p, bins=100, orientation='horizontal', color="blue", stacked=True)
ax2.hist(n, bins=50, orientation='horizontal', color="orange")
ax2.set_xlabel("Count")


# ax = fig.add_subplot(1, 1, 1)
# major_ticks = np.arange(0, 1280, 64)
# y_ticks = [0, 0.5, 1,  2, 3]
# ax.set_xticks(major_ticks)
# ax.set_yticks(y_ticks)
# ax.grid(which='both')
# ax.set_xlim(0, 1216)
# ax.set_ylim(0, 3)
# ax.xaxis.set_label_coords(0.5, -0.21)
# ax.yaxis.set_label_coords(-0.05, 0.5)
#
# ax2.hist(n)
#
# plt.xticks(rotation='vertical')
# plt.xlabel('sample', fontdict=font)
# plt.ylabel('similarity', fontdict=font)
# plt.text(1218, 150, 'anchor to \n negative', fontdict=font1)
# plt.text(1218, -15, 'anchor to \n positive', fontdict=font2)
#
# plt.plot(p, linestyle="",marker="+")
#
# plt.plot(n, linestyle="",marker="x")
#
#
plt.savefig('l_0_e_16_b_3.eps', format='eps', bbox_inches='tight')

plt.show()

treshold = 0.5
tp = 0
fp = 0
tn = 0
fn = 0
for f_p in p:
    if f_p < 0.5:
        tp +=1
    else:
        fp +=1

for f_n in n:
    if f_n >= 0.5:
        tn +=1
    else:
        fn +=1

accuracy = (tp +tn) / (tp+tn+fp+fn)
print(accuracy)