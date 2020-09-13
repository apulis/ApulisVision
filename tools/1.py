import argparse
import copy
import json
import os
import os.path as osp
import time



def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--c', help='train config file path')
    parser.add_argument('--b', help='train config file path')

    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    print(args.c)
    print(json.loads(args.c))



if __name__ == '__main__':
    main()
