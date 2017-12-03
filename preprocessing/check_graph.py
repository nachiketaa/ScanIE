import string
import sys
import os
import json
from PIL import Image as Image
from PIL import ImageDraw as ImageDraw


def check(set_id, fax_id):

    graph_path = '../graph/'
    image_path = '../image/'

    img = Image.open(image_path + '%d/%d.jpg' % (set_id, fax_id))
    W,H = img.size

    draw = ImageDraw.Draw(img)

    f = open(graph_path + '%d/%d.txt' % (set_id, fax_id))

    words = []

    for line in f:
        a = line.split(' ')
        if line[0] != '#':
            w = [float(a[1])*W, float(a[2])*H, float(a[3])*W, float(a[4])*H]
            words.append(w)
            # if a[0] == 'O':
            #     continue
            for i in range(3):
                draw.rectangle(xy=[(w[0]-i,w[1]+i),(w[2]-i,w[3]+i)], outline='blue')
        else:
            if a[0] == '#l':
                for i in range(2,len(a)):
                    u,v = int(a[i-1]), int(a[i])
                    x1 = words[u][2]
                    x2 = words[v][0]
                    y1 = max(words[u][1], words[v][1]) #(words[u][1]+words[u][3])/2
                    y2 = min(words[u][3], words[v][3]) #(words[v][1]+words[v][3])/2
                    y1 = y2 = (y1 + y2) / 2
                    draw.line(xy=[(x1,y1),(x2,y2)], fill='red', width=2)
            if a[0] == '#e':
                u,v = int(a[1]), int(a[2])
                y1 = words[u][3]
                y2 = words[v][1]
                x1 = max(words[u][0], words[v][0])
                x2 = min(words[u][2], words[v][2])
                x1 = x2 = (x1 + x2) / 2
                draw.line(xy=[(x1,y1),(x2,y2)], fill='red', width=2)

    img.save('graph_%d_%d.jpg' % (set_id, fax_id))



json_path = '../data/'

for filename in os.listdir(json_path):
    if '.json' not in filename:
        continue
    f = open(json_path + filename)
    for idx, line in enumerate(f):
        obj = json.loads(line)
        set_id = obj['set_id']
        fax_id = obj['fax_id']
        check(set_id, fax_id)
        if fax_id > 4:
            break
