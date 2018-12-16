from PIL import Image


def default_transform(tensor_protos):
    img_proto = tensor_protos.protos[0]
    img = Image.frombytes(mode='RGB', size=tuple(img_proto.dims), data=img_proto.byte_data)
    label = tensor_protos.protos[1].byte_data
    return img, label
