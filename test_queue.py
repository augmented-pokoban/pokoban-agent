import queue
from random import random

queue = queue.PriorityQueue()

for i in range(10):
    i *= 0.01

    if random() < 0.5:
        i *= -1

    print('Inserting: {}'.format(i))
    queue.put((i, i*10))

while not queue.empty():
    elem = queue.get()
    print(elem)