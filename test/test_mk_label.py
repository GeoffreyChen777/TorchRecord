import os
import random

count = 0

cls = {}

with open('./test/data_list.txt', 'w') as writer:
    for p, d, fl in os.walk('./testdata'):
        for f in fl:
            if f.endswith('jpg'):
                cls_name = p.split('/')[-1]
                if cls_name not in cls:
                    cls[cls_name] = int(cls_name.split('.')[0])
                writer.write("{} {}\n".format(os.path.join(p, f), cls[cls_name]))
