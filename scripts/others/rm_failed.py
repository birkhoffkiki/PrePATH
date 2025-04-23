import pandas as pd
import os

data_dir = '/jhcnas4/Pathology/Patches/Nanfang_colon'
file = '/jhcnas3/Pathology/code/PrePath/csv/non_40x_Nanfang_肠癌.csv'

table = pd.read_csv(file)

items = [('masks', '.jpg'), ('patches', '.h5'), ('stitches', '.jpg')]
for _, row in table.iterrows():
    sid = os.path.splitext(row[0])[0]
    for item, postfix in items:
        p = os.path.join(data_dir, item, sid+postfix)
        if os.path.exists(p):
            print('Found at:', p)
            os.remove(p)
            print('Removed!')


feats = [('h5_files', '.h5'), ('pt_files', '.pt')]
for _, row in table.iterrows():
    sid = os.path.splitext(row[0])[0]
    for item, postfix in feats:
        models = os.listdir(os.path.join(data_dir, item))
        for m in models:
            p = os.path.join(data_dir, item, m, sid+postfix)
            if os.path.exists(p):
                print('Found at:', p)
                os.remove(p)
                print('Removed!')

   