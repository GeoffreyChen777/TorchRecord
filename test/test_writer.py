from torchrecord import Writer
import os

os.system('rm -rf ./testdb')

writer = Writer(data_list='./data_list.txt',
                output_dir='./testdb', db_num=4)

writer.write()
