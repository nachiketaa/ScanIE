import os
import json
import string
from multiprocessing import Pool

def intersect(x1, x2, y1, y2):
    res = x2-x1 + y2-y1 - (max(y2,x2)-min(y1,x1))
    return max(res, 0)


def graph(obj):
    output = file('../graph/%d/%d.txt' % (obj['set_id'], obj['fax_id']), 'w')
    box2id = {}
    W, H = obj['img_width'], obj['img_height']
    firstname = obj['attrs']['First Name'].lower()
    lastname = obj['attrs']['Last Name'].lower()
    try:
        phy_name = obj['attrs']['Physician (Last Name,  First Name Mid Intial)'].lower()
        a = phy_name.split(', ')
        phy_last = a[0]
        phy_first = a[1].split(' ')[0] if len(a) > 1 else None
    except:
        phy_last = None
        phy_first = None
    try:
        DOB = obj['attrs']['DOB'].split(' ')[0]
        if '-' in DOB:
            a = DOB.split('-')
            DOB = a[1]+'/'+a[2]+'/'+a[0]
    except:
        DOB = None
    vocab = set(string.ascii_letters + string.digits + string.punctuation)
    for l in obj['words']:
        words = []
        x1, y1, x2, y2 = float(l['p'][0][0])/W, float(l['p'][0][1])/H, float(l['p'][1][0])/W, float(l['p'][1][1])/H
        l['p'] = [(x1, y1), (x2, y2)]
        for w in l['w']:
            if w[1][0][0] == 0:
                continue
            w[0] = filter(lambda x: x in vocab, w[0])
            if w[0].strip() != '':
                x1, y1, x2, y2 = float(w[1][0][0])/W, float(w[1][0][1])/H, float(w[1][1][0])/W, float(w[1][1][1])/H
                if x2 - x1 > 0.6:
                    continue
                w[1] = [(x1, y1), (x2, y2)]
                box2id[str(w[1])] = len(box2id)
                words.append(w)
                label = 'O'
                content = w[0].lower().strip(string.punctuation)
                if content == firstname:
                    label = 'PtFN'
                elif content == lastname:
                    label = 'PtLN'
                elif content == phy_first:
                    label = 'PhFN'
                elif content == phy_last:
                    label = 'PhLN'
                elif DOB != None and '/' in w[0]:
                    try:
                        a = DOB.split('/')
                        m1, d1, y1 = int(a[0]), int(a[1]), int(a[2])
                        a = w[0].split('/')
                        m2, d2, y2 = int(a[0]), int(a[1]), int(a[2])
                        # print obj['set_id'],obj['fax_id'],m1,d1,y1,m2,d2,y2
                        if m1 == m2 and d1 == d2 and y1 == y2:
                            label = 'DOB'
                    except:
                        pass
                output.write('%s %f %f %f %f %s\n' % (label, x1, y1, x2, y2, w[0].encode('utf-8')))
        l['w'] = sorted(words, key=lambda x:x[1][0][0])
    obj['words'] = [l for l in obj['words'] if len(l['w']) > 0]
    obj['words'] = sorted(obj['words'], key=lambda x:x['p'][0][1])
    for l in obj['words']:
        output.write('#l')
        for w in l['w']:
            output.write(' %d' % box2id[str(w[1])])
        output.write('\n')
    # for l_idx, l in enumerate(obj['words']):
    #     for w_idx, w in enumerate(l['w']):
    #         if w_idx + 1 < len(l['w']):
    #             b1 = w[1]
    #             b2 = l['w'][w_idx+1][1]
    #             if b2[0][0]-b1[1][0] > 0:
    #                 output.write('#e %d %d lr %f\n' % (box2id[str(b1)], box2id[str(b2)], b2[0][0]-b1[1][0]))
    #     for l2_idx, l2 in enumerate(obj['words']):
    #         if l2_idx == l_idx:
    #             continue
    #         if l2['p'][0][0] < l['p'][1][0]:
    #             continue
    #         if intersect(l['p'][0][1],l['p'][1][1],l2['p'][0][1],l2['p'][1][1]) >= 0.5*(l['p'][1][1]-l['p'][0][1]):
    #             b1 = l['w'][-1][1]
    #             b2 = l2['w'][0][1]
    #             if b2[0][0]-b1[1][0] > 0:
    #                 output.write('#e %d %d lr %f\n' % (box2id[str(b1)], box2id[str(b2)], b2[0][0]-b1[1][0]))
    #             break
    for l_idx, l in enumerate(obj['words']):
        for w in l['w']:
            for l2 in obj['words'][l_idx+1:l_idx+3]:
                x1, x2 = w[1][0][0], w[1][1][0]
                x3, x4 = l2['p'][0][0], l2['p'][1][0]
                overlap = intersect(x1, x2, x3, x4)
                if overlap < 0.5*(x2-x1):
                    continue
                for w2 in l2['w']:
                    y1, y2 = w[1][1][1], w2[1][0][1]
                    if y2 < y1 or y2 - y1 > 0.1:
                        continue
                    x3, x4 = w2[1][0][0], w2[1][1][0]
                    overlap = intersect(x1, x2, x3, x4)
                    if overlap >= 0.5*(x2-x1) and overlap >= 0.5*(x4-x3):
                        b1, b2 = w[1], w2[1]
                        output.write('#e %d %d ud %f\n' % (box2id[str(b1)], box2id[str(b2)], y2-y1))
                        break
                break
    output.close()



json_path = '../data/'
for filename in os.listdir(json_path):
    if '.json' not in filename:
        continue
    if filename.replace('.json', '') not in os.listdir('../graph/'):
        os.mkdir('../graph/' + filename.replace('.json', ''))
    f = file(json_path + filename)
    objs = [json.loads(line) for line in f]
    #graph(objs[0])
    pool = Pool(processes=12)
    pool.map(graph, objs)


