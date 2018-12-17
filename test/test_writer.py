from torchrecord import Writer
import os
from torchrecord import TensorProtos
from PIL import Image
import cv2


os.system('rm -rf ./test/testdb')


def data_process_func(data):
    data = data.split(' ')
    tensor_protos = TensorProtos()

    img = open(data[0], 'rb')

    img_tensor = tensor_protos.protos.add()
    img_dim = Image.open(data[0]).size
    img_tensor.dims.extend(img_dim)
    img_tensor.data_type = 3
    img_tensor.byte_data = img.read()

    label_tensor = tensor_protos.protos.add()
    label_data = str.encode(data[1])
    label_tensor.data_type = 3
    label_tensor.byte_data = label_data
    return tensor_protos

writer = Writer(data_list='./test/data_list.txt',
                output_dir='./testdb', db_num=4, data_process_func=data_process_func)

writer.write()
