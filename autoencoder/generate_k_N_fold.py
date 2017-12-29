import os
import csv
from random import shuffle

files = os.listdir('zipped')
K = 10
n = len(files)
shuffle(files)
per_file = int(n / K)

with open('data.csv', 'w') as file:
    writer = csv.DictWriter(file, fieldnames=['file', 'k'])
    writer.writeheader()
    for k in range(K):
        for i in range(per_file):
            file_name = files.pop()
            writer.writerow({'file': file_name, 'k': k})
