import os
import random
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process some CSV files.')
parser.add_argument('--p', type=str, help='Path to the input CSV file')
parser.add_argument('--num', type=int, help='Path to the input CSV file')
parser.add_argument('--root', type=str, help='Directory to save the output CSV files')

args = parser.parse_args()

p = args.p
root = args.root
num = args.num

# 如果root目录不存在，则创建它
if not os.path.exists(root):
    os.makedirs(root)
    

with open(p) as f:
    data = f.readlines()[1:]

random.shuffle(data)

cutoffs = [int(i/num*len(data)) for i in range(num)] + [len(data)]

for i in range(num):
    item = data[cutoffs[i]:cutoffs[i+1]]
    with open(os.path.join(root, 'part_{}.csv'.format(i)), 'w') as f:
        f.write('dir,case_id,slide_id,label\n')
        for i in item:
            f.write(i)
            
