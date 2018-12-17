from torchrecord import Writer
import os

os.system('rm -rf ./test/testdb')

writer = Writer(data_list='./test/data_list.txt',
                output_dir='./testdb', db_num=4)

writer.write()
