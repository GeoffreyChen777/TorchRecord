from multiprocessing import Process, Pool, Manager, Queue
import os
import lmdb
import random
from .dataset import LMDBDataset
from .caffe2_pb2 import TensorProtos
from .transforms import *
import sys
import random
import math
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


def default_collate_fn(batch):
    return batch


class DBLoaderIterator(object):
    def __init__(self, data_queue, batch_size, collate_fn, shuffle, length):
        self.data_queue = data_queue
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.batch_pool = []
        self.add_data_to_batch_pool()
        self.shuffle = shuffle
        self.length = length
        self.batch_length = math.ceil(1. * length / self.batch_size)
        self.idx = 0

    def add_data_to_batch_pool(self):
        for i in range(self.batch_size*4):
            self.batch_pool.append(self.data_queue.get(True))

    def __next__(self):
        if self.idx == self.batch_length:
            raise StopIteration
        self.idx += 1

        if self.idx == self.batch_length:
            batch_size = self.length - (self.idx - 1) * self.batch_size
        else:
            batch_size = self.batch_size

        for i in range(batch_size):
            self.batch_pool.append(self.data_queue.get(True))

        if self.shuffle:
            return self.collate_fn([self.batch_pool.pop(random.randrange(len(self.batch_pool))) for _ in range(batch_size)])
        else:
            return self.collate_fn([self.batch_pool.pop() for _ in range(batch_size)])

    next = __next__

    def __iter__(self):
        return self


class RecordLoader(object):
    def __init__(self, record_path, batch_size=32, collate_fn=default_collate_fn, num_workers=1, shuffle=True, dataq_maxsize=200, transform=default_transform, dataset=None):

        self.batch_size = batch_size
        self.collate_fn = collate_fn

        if num_workers < 1:
            raise ValueError("Worker number must greater than 0")

        self.num_workers = num_workers
        self.shuffle = shuffle

        self.db_list = self.handel_db(record_path)
        self.length = self.cal_length()
        self.db_queue = Queue(maxsize=2*len(self.db_list))

        self.data_queue = Queue(maxsize=dataq_maxsize)

        manager = Manager()
        self.flag = manager.dict()

        self.insert_db_queue()

        self.transform = transform

        if dataset is None:
            self.dataset = self.default_dataset
        else:
            self.dataset = dataset
        self.start_workers()

    @staticmethod
    def default_proto():
        proto = TensorProtos()
        return proto

    def default_dataset(self, db_path, transform):
        return LMDBDataset(db_path, transform, self.default_proto)

    def worker_func(self, db_queue, data_queue, flag, dataset_class):
        while True:
            db_path = db_queue.get(True)
            dataset = dataset_class(db_path, self.transform)

            for data in dataset:
                data_queue.put(data, True)

            if flag['running'] == 1:
                flag['running'] = flag['total']
            else:
                flag['running'] -= 1
            while True:
                if flag['running'] == flag['total']:
                    break

    def start_workers(self):
        self.flag['total'] = self.num_workers
        self.flag['running'] = self.num_workers
        for i in range(self.num_workers):
            p = Process(target=self.worker_func, args=(self.db_queue, self.data_queue, self.flag, self.dataset))
            p.daemon = True
            p.start()

    @staticmethod
    def insert_db_queue_func(shuffle, db_list, db_queue):
        while True:
            if shuffle:
                random.shuffle(db_list)
            for db_name in db_list:
                db_queue.put(db_name, True)

    def insert_db_queue(self):
        p = Process(target=self.insert_db_queue_func, args=(self.shuffle, self.db_list, self.db_queue,))
        p.daemon = True
        p.start()

    @staticmethod
    def handel_db(db_path):
        db_list = os.listdir(db_path)
        db_list = [os.path.join(db_path, 'db{}'.format(i)) for i, x in enumerate(db_list) if x.startswith('db')]

        assert len(db_list) != 0
        return db_list

    def cal_length(self):
        length = 0
        for db_path in self.db_list:
            db = lmdb.open(db_path, readonly=True)
            with db.begin() as txn:
                length += txn.stat()['entries']
        return length

    def __len__(self):
        return math.ceil(1. * self.length / self.batch_size)

    def __iter__(self):
        return DBLoaderIterator(self.data_queue, self.batch_size, self.collate_fn, self.shuffle, self.length)
