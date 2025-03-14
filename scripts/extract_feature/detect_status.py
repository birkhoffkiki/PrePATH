import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

root = args.path

if __name__ == '__main__':
    models = os.listdir(root)
    results = []
    for m in models:
        files = [i for i in os.listdir(os.path.join(root, m)) if i.endswith('.pt')]
        results.append((m, len(files)))
    
    results = sorted(results, key= lambda i: i[1])
    for r in results:
        print("{: <20}{}".format(r[0], r[1]))
        