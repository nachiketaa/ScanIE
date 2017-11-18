import string
import sys
import os
import json
from PIL import Image as Image
from PIL import ImageDraw as ImageDraw

json_path = '../data/'
image_path = '../image/'

for filename in os.listdir(json_path):
    if '.json' not in filename:
        continue
    f = file(json_path + filename)
    for idx, line in enumerate(f):
        obj = json.loads(line)
        set_id = obj['set_id']
        fax_id = obj['fax_id']
        img = Image.open(image_path + '%d/%d.jpg' % (set_id, fax_id))
        draw = ImageDraw.Draw(img)
        for l in obj['words']:
            for i in range(3):
                draw.rectangle(xy=[(l['p'][0][0]-i,l['p'][0][1]+i),(l['p'][1][0]-i,l['p'][1][1]+i)], outline='red')
        img.save('tmp_%d_%d.jpg' % (set_id, fax_id))
        if idx > 1:
            break
