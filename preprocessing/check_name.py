import string
import sys
import os
import json

count = 0
hit = 0

basepath = '../data/'
for filename in os.listdir(basepath):
    if '.json' not in filename:
        continue
    f = file(basepath + filename)
    _count = 0
    _hit = 0
    for line in f:
        obj = json.loads(line)
        firstname = obj['attrs']['First Name'].lower()
        lastname = obj['attrs']['Last Name'].lower()
        f_hit = 0
        l_hit = 0
        for l in obj['words']:
            for w in l['w']:
                word = w[0].lower().translate({ord(c): None for c in string.punctuation})
                if word == firstname:
                    f_hit += 1
                if word == lastname:
                    l_hit += 1
        print f_hit, l_hit
        if f_hit > 0 and l_hit > 0:
            _hit += 1
        _count += 1
    print filename, _hit, _count, float(_hit)/_count 
    hit += _hit
    count += _count

print hit, count, float(hit)/count
