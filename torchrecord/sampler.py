import torch.utils.data as data
import os
import random
import lmdb
import numpy as np


class TRIterator(object):

    def __init__(self, record_path, record_num, shuffle, full_shuffle, batch_size, num_workers):
        self.record_path = record_path
        self.record_num = record_num
        self.shuffle = shuffle
        self.full_shuffle = full_shuffle
        self.batch_size = batch_size
        self.num_worker = num_workers
        self.length, self.total_length = self.open_db()
        self.item_list = self.make_item_list()
        self.pool_count_matrix = self.init_batch_pool()
        self.count = 0

    def open_db(self):
        db_name_list = ["db{}".format(x).encode() for x in range(self.record_num)]
        length = []
        total_length = 0

        for db in db_name_list:
            env = lmdb.open(os.path.join(self.record_path), readahead=False, max_readers=1, readonly=True, lock=False, meminit=False, max_dbs=self.record_num)
            db = env.open_db(db)
            txn = env.begin(db=db)
            l = txn.stat()['entries']
            total_length += l
            length.append(l)
        return length, total_length

    def init_batch_pool(self):
        count_matrix = np.zeros(self.record_num)

        num = min(self.total_length, 4*self.batch_size*self.num_worker)
        num = max(num, 64*self.num_worker)
        count = 0
        while True:
            for i in range(self.record_num):
                count += 1
                count_matrix[i] += 1
                if count == num:
                    return count_matrix

    def make_item_list(self):
        item_list = []
        for db_idx in range(self.record_num):
            num = self.length[db_idx]
            db_idxs = [db_idx] * num
            key_idxs = [x for x in range(num)]
            close_sig = [0] * num
            item_slice = list(np.vstack((db_idxs, key_idxs, close_sig)).T)
            if self.shuffle or self.full_shuffle:
                random.shuffle(item_slice)
            item_slice = np.array(item_slice)
            item_list += list(item_slice)
        if self.full_shuffle:
            random.shuffle(item_list)
        return item_list

    def refresh(self):
        self.count = 0
        working_list = []
        db_group = [i for i in range(self.record_num)]
        for item in self.item_list:
            if item[1] < self.pool_count_matrix[item[0]]:
                continue
            if item[0] in db_group:
                working_list.append((item[0], item[1], 1))
                db_group.remove(item[0])
            else:
                working_list.append((item[0], item[1], 0))
        if self.full_shuffle:
            random.shuffle(working_list)

        batch_holder = []
        num = min(self.total_length, 4*self.batch_size*self.num_worker)
        num = max(num, 64*self.num_worker)
        for i in range(num):
            batch_holder.append([-1, -1, -1])

        self.working_list = batch_holder + working_list

    def __next__(self):
        if self.count == self.total_length:
            raise StopIteration
        self.count += 1
        item = self.working_list.pop()
        return item

    next = __next__

    def __iter__(self):
        return self

    def __len__(self):
        return self.total_length


class TRSampler(data.Sampler):
    def __init__(self, record_path, record_num, shuffle=False, full_shuffle=False, batch_size=1, num_workers=1):
        self.record_path = record_path
        self.record_num = record_num
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sampler = TRIterator(record_path, record_num, shuffle, full_shuffle, batch_size, num_workers)

    def __iter__(self):
        batch = []
        self.sampler.refresh()
        for idx, data in enumerate(self.sampler):
            batch.append(data)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

