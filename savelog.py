import glob
from collections import OrderedDict
import os
from datetime import datetime


def save_log(n_filename,new_file_content):
 
    now = datetime.now()
    dir_path = './'


    # if __name__ == '__main__':
        
        # print('Total :', '{:.3f}'.format(win / total))
    save_file_at_new_dir(dir_path, n_filename , new_file_content , mode='w')
    print('ファイル'+n_filename+'保存完了')


def save_file_at_new_dir(new_dir_path, new_filename, new_file_content, mode='w'):
    os.makedirs(new_dir_path, exist_ok=True)
    with open(os.path.join(new_dir_path, new_filename), mode) as f:
        f.writelines(new_file_content)
