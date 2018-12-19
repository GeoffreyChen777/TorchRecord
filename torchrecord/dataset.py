import torch.utils.data as data
from torchrecord import TensorProtos
from torchrecord.transforms import default_transform
import os
import random
import lmdb


class TRDataset(data.Dataset):
    def __init__(self, record_path='', transform=default_transform, record_num=1, shuffle=False, batch_size=1, proto=TensorProtos):
        self.transform = transform
        self.proto = proto

        self.record_path = record_path
        self.record_num = record_num
        self.shuffle = shuffle

        self.batch_size = batch_size

        self.worker_curs, self.worker_envs, self.length, self.total_length = self.open_db()
        self.idx_format = self.init_workers()
        self.batch_pool = []

        self.last_db = -1
        self.new_db_count = 0

        self.test_count = 0

    def init_batch_pool(self, worker_id):
        num = min(self.total_length, 4*self.batch_size)
        num = max(64, num)
        count = 0
        key_count = []
        for i in range(self.record_num):
            key_count.append(int(worker_id*num/self.record_num))
        while True:
            for i in range(self.record_num):
                value = self.read_item(i, key_count[0])
                self.batch_pool.append(value)
                count += 1
                if count == num:
                    return
            key_count[0] += 1

    def read_item(self, db_idx, key_idx):
        self.worker_curs[db_idx].set_range(self.idx_format.format(key_idx).encode())
        value = self.worker_curs[db_idx].value()
        return value

    def open_db(self):
        db_name_list = ["db{}".format(x).encode() for x in range(self.record_num)]
        length = []
        total_length = 0
        worker_envs = []
        worker_curs = []

        for db in db_name_list:
            env = lmdb.open(os.path.join(self.record_path), readahead=False, max_readers=1, readonly=True, lock=False, meminit=False, max_dbs=self.record_num)
            worker_envs.append(env)
            db = env.open_db(db)
            txn = env.begin(db=db)
            worker_curs.append(txn.cursor())
            l = txn.stat()['entries']
            total_length += l
            length.append(l)
        return worker_curs, worker_envs, length, total_length

    def init_workers(self):
        idx_place = len(str(max(self.length)))
        idx_format = '{:0' + str(idx_place) + '}'
        for db_idx in range(self.record_num):
            self.worker_curs[db_idx].first()

        return idx_format

    def parse_byte(self, byte_str):
        tensor_protos = self.proto()
        tensor_protos.ParseFromString(byte_str)
        item = self.transform(tensor_protos)
        return item

    def __getitem__(self, item):           # (db_idx, close_sig, read_sig)
        db_idx, key_idx, close_sig = item

        if db_idx != -1:
            value = self.read_item(db_idx, key_idx)
            self.batch_pool.append(value)

        if self.shuffle:
            byte_str = self.batch_pool.pop(random.randrange(len(self.batch_pool)))
        else:
            byte_str = self.batch_pool.pop()
        obj = self.parse_byte(byte_str)

        if db_idx != self.last_db:
            if self.last_db != -1:
                self.worker_envs[self.last_db].close()
            self.last_db = db_idx

        return obj

    def __len__(self):
        return 1

