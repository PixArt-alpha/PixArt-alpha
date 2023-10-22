import os
# import  ipdb
import subprocess
import argparse

def read_txt_to_lit(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file]
    return lines

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path",default='',type = str,)
    parser.add_argument("--src_path",default='',type = str,)
    parser.add_argument("--tgt_file",default='',type = str,)
    parser.add_argument("--sub_fix",default='',type = str,)
    parser.add_argument("--pre_fix",default='',type = str,)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    tgt_path = os.path.join(args.base_path, args.tgt_file)
    src_path = args.src_path
    sub_fix = args.sub_fix
    pre_fix = args.pre_fix


    abilities = ['color', 'shape', 'texture', 'spatial', 'non-spatial', 'complex']
    for item in abilities:
        command = f"mkdir -p {os.path.join(tgt_path, item+'/samples')}"
        result = subprocess.run(command, shell=True)
    command = f"mkdir -p {os.path.join(tgt_path, 'result')}"
    result = subprocess.run(command, shell=True)


    src_abilities = ['color', 'shape', 'texture', 'spatial', 'action', 'complex']
    for index, file_ in enumerate(src_abilities):
        if file_ != 'action':
            command = f"cp -r {os.path.join(src_path, pre_fix + file_ + sub_fix)}/* {os.path.join(tgt_path, file_, 'samples/')}"
            if not os.path.exists(os.path.join(src_path, pre_fix + file_ + sub_fix)):
                print(f"fails with {file_}")
        else:
            command = f"cp -r {os.path.join(src_path, pre_fix + file_ + sub_fix)}/* {os.path.join(tgt_path, 'non-spatial', 'samples/')}"
            if not os.path.exists(os.path.join(src_path, pre_fix + file_ + sub_fix)):
                print(f"fails with {file_}")
        subprocess.run(command, shell=True)
        # print(command)
        print(f"done with {file_}")
