from PIL import Image
import torchvision.transforms as tvt
import time


trans = tvt.Compose([
    tvt.Resize((224, 224)),
    tvt.ToTensor()
])


def default_transform(tensor_protos):
    img_proto = tensor_protos.protos[0]
    img = Image.frombytes(mode='RGB', size=tuple(img_proto.dims), data=img_proto.byte_data)
    img = trans(img)
    label = int(tensor_protos.protos[1].byte_data)
    return img, label
