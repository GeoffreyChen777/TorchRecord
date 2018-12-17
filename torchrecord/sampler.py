import torch.utils.data as data
import os
import random
import lmdb
import sys
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


class TRSampler(data.Sampler):

    def __init__(self, record_path, shuffle=False, batch_size=1):
        self.record_path = record_path
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.record_list, self.length = self.handle_db()
        self.curr_cur = self.record_list.get()
        self.has_next = self.curr_cur.next()
        self.batch_pool = self.init_batch_pool()

    def init_batch_pool(self):
        batch_pool = []
        for i in range(min(4*self.batch_size, self.length)):
            value = self.get_next()
            batch_pool.append(value)
            if not self.has_next:
                if not self.record_list.empty():
                    self.curr_cur = self.record_list.get()
                elif len(self.batch_pool) == 0:
                    raise StopIteration
        return batch_pool

    def handle_db(self):
        db_list = os.listdir(self.record_path)
        record_list = queue.Queue()
        length = 0
        if self.shuffle:
            random.shuffle(db_list)
        for db in db_list:
            env = lmdb.open(os.path.join(self.record_path, db), readahead=False)
            txn = env.begin()
            cur = txn.cursor()
            length += txn.stat()['entries']
            record_list.put(cur)
        return record_list, length

    def get_next(self):
        value = self.curr_cur.value()
        self.has_next = self.curr_cur.next()
        return value

    def __iter__(self):
        batch = []
        for i in range(self.length):
            if self.has_next:
                value = self.get_next()
                self.batch_pool.append(value)
            else:
                if not self.record_list.empty():
                    self.curr_cur = self.record_list.get()
                    self.has_next = self.curr_cur.next()
                    value = self.get_next()
                    self.batch_pool.append(value)

            if self.shuffle:
                batch.append(self.batch_pool.pop(random.randrange(len(self.batch_pool))))
            else:
                batch.append(self.batch_pool.pop())
            if len(batch) == self.batch_size or len(self.batch_pool) == 0:
                yield batch
                batch = []

    def __len__(self):
        return (self.length + self.batch_size - 1) // self.batch_size
