import pandas as pd
import os
import csv
import numpy as np


def get_accuracy(filename):
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter='\t')
        p = []
        n = []
        accuracy = []
        fold = 0

        for row in csv_reader:
            if len(row) == 2:
                p.append(float(row[0].strip()))
                n.append(float(row[1].strip()))
            else:
                if row[0] == "TensorFlow version: 1.13.1":
                    fold += 1
                loc = row[0].find("ccuracy:")
                if "accuracy:" in row[0]:
                    arr = row[0].split(' ')
                    acc = float(arr[2])
                    accuracy.append(acc)

        return accuracy

def get_min_max(filename):
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter='\t')
        p = []
        n = []
        accuracy = []
        fold = 0

        for row in csv_reader:
            if len(row) == 2:
                p.append(float(row[0].strip()))
                n.append(float(row[1].strip()))
            else:
                if row[0] == "TensorFlow version: 1.13.1":
                    fold += 1
                loc = row[0].find("ccuracy:")
                if "accuracy:" in row[0]:
                    arr = row[0].split(' ')
                    acc = float(arr[2])
                    accuracy.append(acc)
        min_p = np.min(p)
        max_p = np.max(p)
        min_n = np.min(n)
        max_n = np.max(n)

        return min_p, max_p, min_n, max_n
exp_dir = "/home/virginijus/tmp/pycharm_project_202/numta_experiments/"


epochs       = [1, 4, 16, 50]
lcs          = [0, 4, 8, 12]
# batch_sizes  = [3, 6, 12, 24, 48, 96, 192]
batch_sizes  = [3, 6, 12, 24, 48, 96]

# LC is fixed
for lc in lcs:
    rez_file_name = f"{exp_dir}latex_l_{lc}_batch_vs_epoch.txt"
    print(rez_file_name)
    rez_file = open(rez_file_name, 'w')
    # Write header
    rez_file.write("Batch size & ")
    rez_file.write(" & ".join(np.array(epochs).astype(str)))
    rez_file.write(r' \\')
    rez_file.write("\n")
    rez_file.write(r'\hline')
    rez_file.write("\n")


    for batch_size in batch_sizes:
        rez_file.write(f"{batch_size}")
        for epoch in epochs:
            filename = f"{exp_dir}res_l_{lc}_e_{epoch}_b_{batch_size}"
            if os.path.isfile(filename):
                accuracies = get_accuracy(filename)
                print(accuracies)

                rez_file.write(f" & {np.round(np.mean(accuracies), 2)} ({np.round(np.std(accuracies), 2)})")
            else:
                rez_file.write(" & ")

        rez_file.write(r'  \\')
        rez_file.write("\n")

    rez_file.close()

# Batch size is fixed
for batch_size in batch_sizes:
    rez_file_name = f"{exp_dir}latex_b_{batch_size}_lc_vs_epoch.txt"
    print(rez_file_name)
    rez_file = open(rez_file_name, 'w')
    # Write header
    rez_file.write("ModelNN(k) & ")
    rez_file.write(" & ".join(np.array(epochs).astype(str)))
    rez_file.write(r' \\')
    rez_file.write("\n")
    rez_file.write(r'\hline')
    rez_file.write("\n")

    for lc in lcs:
        rez_file.write(f"{ -1 * lc }")
        for epoch in epochs:
            filename = f"{exp_dir}res_l_{lc}_e_{epoch}_b_{batch_size}"
            if os.path.isfile(filename):
                accuracies = get_accuracy(filename)
                print(accuracies)

                rez_file.write(f" & {np.round(np.mean(accuracies), 2)} ({np.round(np.std(accuracies), 2)})")
            else:
                rez_file.write(" & ")

        rez_file.write(r'  \\')
        rez_file.write("\n")

    rez_file.close()

for batch_size in batch_sizes:
    rez_file_name = f"{exp_dir}latex_b_{batch_size}_min_max_lc_vs_epoch.txt"
    print(rez_file_name)
    rez_file = open(rez_file_name, 'w')
    # Write header
    rez_file.write("ModelNN(k) & ")
    rez_file.write(" & ".join(np.array(epochs).astype(str)))
    rez_file.write(r' \\')
    rez_file.write("\n")
    rez_file.write(r'\hline')
    rez_file.write("\n")

    for lc in lcs:
        rez_file.write(f"{ -1 * lc }")
        for epoch in epochs:
            filename = f"{exp_dir}res_l_{lc}_e_{epoch}_b_{batch_size}"
            if os.path.isfile(filename):
                min_p, max_p, min_n, max_n = get_min_max(filename)
                # out_str = f"{np.round(min_p, 2)} - {np.round(max_p, 2)} \n {np.round(min_n, 2)} - {np.round(max_n, 2)}"
                # print(out_str)
                out_str1 = f"{np.round(min_p, 2)} - {np.round(max_p, 2)} "

                rez_file.write(f" & {out_str1}")
            else:
                rez_file.write(" & ")

        rez_file.write(r'  \\')
        rez_file.write(" \n")

        for epoch in epochs:
            filename = f"{exp_dir}res_l_{lc}_e_{epoch}_b_{batch_size}"
            if os.path.isfile(filename):
                min_p, max_p, min_n, max_n = get_min_max(filename)

                out_str2 = f" {np.round(min_n, 2)} - {np.round(max_n, 2)}"

                rez_file.write(f" & {out_str2}")
            else:
                rez_file.write(" & ")
        rez_file.write(r'  \\')
        rez_file.write("\n")

    rez_file.close()