import pandas as pd
import util
import os
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv('../input/stage1_solution.csv')

old_img_id = ''
for index, row in df.iterrows():
    print("{}: {}".format(index, row['ImageId']))
    if row['ImageId'] != old_img_id:
        old_img_id = row['ImageId']
        mask_count = 0
        d = "../input/stage1_test/{}/masks".format(row['ImageId'])
        if not os.path.exists(d):
            os.makedirs(d)
    
    mask = util.rleToMask(row['EncodedPixels'], row['Height'], row['Width'])

    # check if directory exists
    # save as *.png
    im = Image.fromarray(mask)
    im.save('../input/stage1_test/{}/masks/mask_{}.png'.format(row['ImageId'], mask_count))
    mask_count += 1
