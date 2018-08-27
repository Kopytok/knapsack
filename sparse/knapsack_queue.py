import logging

from operator import attrgetter
from collections import deque, namedtuple

def line_to_numbers(line):
    """ Split line into 2 digits and convert them to int """
    return tuple(int(num) for num in line.split())

def read_item(n, line):
    """ Read one knapsack item """
    value, weight = line_to_numbers(line)
    return Item(n, value, weight, value / weight)

Item = namedtuple("Item", ["id", "value", "weight", "density"])

class Queue(object):
    """ Queue with items available for knapsack """
    def __init__(self, n_items=None, capacity=None):
        self.n_items = n_items
        self.capacity = capacity
        self._queue = deque(maxlen=n_items)

    def __len__(self):
        return len(self._queue)

    def __repr__(self):
        return "Queue with {}/{} items".format(len(self), self.n_items)

    def __str__(self):
        n = min(5, len(self))
        lines = ["Queue with {}/{} items. First {} items:\n"
                 .format(len(self), self.n_items, n)]
        for i in range(n):
            lines.append(str(self._queue[i]))
        return "\n".join(lines)

    def enqueue(self, item):
        self._queue.append(item)

    def dequeue(self):
        try:
            return self._queue.popleft()
        except IndexError as e:
            logging.info("No more items in queue")
            return

    def sort(self, by="density", descending=True):
        for item in sorted(self._queue, key=attrgetter(by),
                           reverse=descending):
            self.enqueue(item)

    @classmethod
    def read_queue(cls, path):
        with open(path, "r") as f:
            items = f.readlines()

        n_items, capacity = line_to_numbers(items[0])
        logging.info("New queue: {} items, {} capacity".format(n_items, capacity))
        q = cls(n_items, capacity)
        for n, item in enumerate(items[1:]):
            q.enqueue(read_item(n, item))
        return q

if __name__ == "__main__":
    q = Queue.read_queue("../data/ks_4_0")
    print(repr(q))
