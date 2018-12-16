import lmdb
import os
from PIL import Image
from .caffe2_pb2 import TensorProtos
from multiprocessing import Pool, Process
import random
import time


def default_data_process_func(data):
    data = data.split(' ')
    tensor_protos = TensorProtos()

    img = Image.open(data[0]).convert("RGB")
    img_tensor = tensor_protos.protos.add()
    img_tensor.dims.extend(img.size)
    img_tensor.data_type = 3
    img_tensor.byte_data = img.tobytes()

    label_tensor = tensor_protos.protos.add()
    label_data = str.encode(data[1])
    label_tensor.data_type = 3
    label_tensor.byte_data = label_data
    return tensor_protos


class Writer(object):
    def __init__(self, data_list=None,
                 output_dir='./torchrecord', map_size=1099511627776, db_num=1, shuffle=True,
                 data_process_func=default_data_process_func):

        if data_list is None:
            raise ValueError('Parameter needed: data_list.')
        if not data_list.endswith('.txt') and not data_list.endswith('.csv'):
            raise ValueError('Parameter error: data_list must be txt or csv. Found: {}'.format(data_list))

        self.data_list = data_list
        self.output_dir = output_dir

        self.data_process_func = data_process_func

        self.map_size = map_size
        self.shuffle = shuffle
        self.db_num = db_num

    def parse_data_list(self):
        with open(self.data_list, 'r') as reader:
            data_list = reader.readlines()
        data_list = [x.strip() for x in data_list]
        if self.shuffle:
            random.shuffle(data_list)
        return data_list

    def write(self):
        data_list = self.parse_data_list()

        data_num = len(data_list)

        print("Total: {} data items.".format(data_num))
        print("Start Writing...")
        tik = time.time()
        ps = []
        for i in range(self.db_num):
            p = Process(target=self.write_func,
                        args=(data_list[int(i * data_num / self.db_num):int((i + 1) * data_num / self.db_num)],
                              i, self.output_dir, self.map_size, self.data_process_func,))
            p.daemon = True
            p.start()
            ps.append(p)

        for i in range(self.db_num):
            ps[i].join()
        tok = time.time()
        print('All Processes Are Done. Cost:{}s'.format(tok-tik))

    @staticmethod
    def write_func(data_list, db_idx, output_dir, map_size, data_process_func):
        db_path = os.path.join(output_dir, 'db{}'.format(db_idx))
        if not os.path.isdir(db_path):
            os.makedirs(db_path)
        try:
            env = lmdb.open(db_path, map_size=map_size)
            with env.begin(write=True) as tnx:
                for idx, data in enumerate(data_list):
                    tensor_protos = data_process_func(data)
                    tnx.put('{}'.format(idx).encode(), tensor_protos.SerializeToString())
        except Exception as e:
            print(e)
