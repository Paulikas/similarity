import csv
import numpy as np
import matplotlib.pyplot as plt

with open('res800y') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter='\t')
  p = []
  n = []
  for row in csv_reader:
    p.append(float(row[0].strip()) / (28*7))
    n.append(float(row[1].strip()) / (28*7))

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
ax = fig.add_subplot(1, 1, 1)
major_ticks = np.arange(0, 1280, 64)
y_ticks = [0, 0.29, 0.58]
ax.set_xticks(major_ticks)
ax.set_yticks(y_ticks)
ax.grid(which='both')
ax.set_xlim(0, 1216)
#ax.set_ylim(0.79, 1.01)
ax.xaxis.set_label_coords(0.05, -0.2)
ax.yaxis.set_label_coords(-0.05, 0.5)


plt.xticks(rotation='vertical')

plt.xlabel('sample', fontdict=font)
plt.ylabel('similarity', fontdict=font)
plt.text(1218, 80, 'anchor to \n negative', fontdict=font1)
plt.text(1218, 15, 'anchor to \n positive', fontdict=font2)

plt.plot(p, linestyle="",marker="+")


plt.plot(n, linestyle="",marker="x")

plt.show()
