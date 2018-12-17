import torch.utils.data as data
import os
import random
import lmdb
import sys
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


class TRIterator(object):

    def __init__(self, record_path, record_num, shuffle, batch_size):
        self.record_path = record_path
        self.record_num = record_num
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.record_list, self.length = self.handle_db()
        self.cur_idx = 0
        self.env = None
        self.curr_cur, self.env = self.open_db(self.record_list[0])
        self.has_next = self.curr_cur.next()
        self.batch_pool = []
        self.init_batch_pool()

    def init_batch_pool(self):
        for i in range(min(4*self.batch_size, self.length)):
            value = self.get_next()
            self.batch_pool.append(value)
            if not self.has_next:
                if len(self.record_list) - 1 != self.cur_idx:
                    self.curr_cur, self.env = self.open_db(self.record_list[self.cur_idx])
                elif len(self.batch_pool) == 0:
                    raise StopIteration

    def handle_db(self):
        db_list = ["db{}".format(x).encode() for x in range(self.record_num)]
        length = 0
        if self.shuffle:
            random.shuffle(db_list)
        for db in db_list:
            env = lmdb.open(os.path.join(self.record_path), readahead=False, max_readers=1, readonly=True, lock=False, meminit=False, max_dbs=self.record_num)
            db = env.open_db(db)
            txn = env.begin(db=db)
            length += txn.stat()['entries']
            env.close()
        return db_list, length

    def open_db(self, db_name):
        if self.env is not None:
            self.env.close()
        env = lmdb.open(os.path.join(self.record_path), readahead=False, max_readers=1, readonly=True, lock=False, meminit=False, max_dbs=self.record_num)
        db = env.open_db(db_name)
        txn = env.begin(db=db)
        cur = txn.cursor()
        self.has_next = cur.next()
        return cur, env

    def get_next(self):
        value = self.curr_cur.value()
        self.has_next = self.curr_cur.next()
        return value

    def refresh(self):
        random.shuffle(self.record_list)
        self.cur_idx = 0
        self.curr_cur, self.env = self.open_db(self.record_list[self.cur_idx])
        self.has_next = self.curr_cur.next()
        self.init_batch_pool()

    def __next__(self):
        if len(self.batch_pool) == 0:
            raise StopIteration
        if self.has_next:
            value = self.get_next()
            self.batch_pool.append(value)
        else:
            if self.cur_idx != len(self.record_list) - 1:
                self.cur_idx += 1
                self.curr_cur, self.env = self.open_db(self.record_list[self.cur_idx])
                self.has_next = self.curr_cur.next()
                value = self.get_next()
                self.batch_pool.append(value)

        if self.shuffle:
            return self.batch_pool.pop(random.randrange(len(self.batch_pool)))
        else:
            return self.batch_pool.pop()

    def __iter__(self):
        return self

    def __len__(self):
        return self.length


class TRSampler(data.Sampler):
    def __init__(self, record_path, record_num, shuffle=False, batch_size=1):
        self.record_path = record_path
        self.record_num = record_num
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sampler = TRIterator(record_path, record_num, shuffle, batch_size)

    def __iter__(self):
        batch = []
        for idx, data_byte in enumerate(self.sampler):
            batch.append(data_byte)
            if idx == len(self.sampler) - 1:
                self.sampler.refresh()
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        self.sampler.refresh()
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

