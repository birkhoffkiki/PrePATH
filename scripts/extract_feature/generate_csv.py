import os
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process to generate CSV files.')
parser.add_argument('--h5_dir', type=str, help='Path to the coors dir')
parser.add_argument('--num', type=int, help='Path to the input CSV file')
parser.add_argument('--root', type=str, help='Directory to save the output CSV files')

args = parser.parse_args()

h5_dir = args.h5_dir
root = args.root
num = args.num
if not os.path.exists(root):
    os.makedirs(root)

files = os.listdir(h5_dir)
data = [os.path.splitext(f)[0] for f in files]

cutoffs = [int(i/num*len(data)) for i in range(num)] + [len(data)]

for i in range(num):
    item = data[cutoffs[i]:cutoffs[i+1]]
    with open(os.path.join(root, 'part_{}.csv'.format(i)), 'w') as f:
        f.write('case_id,slide_id\n')
        for i in item:
            f.write("{},{}\n".format(i, i))
            
