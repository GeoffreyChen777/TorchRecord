import os
import random

count = 0
with open('./data_list.txt', 'w') as writer:
    for p, d, fl in os.walk('./testdata'):
        for f in fl:
            if f.endswith('jpg'):
                writer.write("{} {}\n".format(os.path.join(p, f), random.randint(0, 199)))
