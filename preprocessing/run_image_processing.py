import sys
import io
import os
import re
import json
from subprocess import call
from multiprocessing import Pool

base_path = '../rawdata/'
output_base_path = '../image/'

def preprocess(filename):
    name, ext = os.path.splitext(filename)
    ext = ext.lower()
    output_name = output_path + re.sub('\D','',name) + '.jpg'
    # print filename, output_name
    if ext == '.pdf':
        call(['convert', '-density', '300', '-append', path+filename, output_name])
    elif ext in ['.jpg', '.jpeg', '.png']:
        # call(['convert', '-resize', '100%%', path+filename, output_name])
        call(['cp', path+filename, output_name])
    else:
        return
    # call('./textcleaner.sh %s %s -g -e stretch -f 25 -o 5 -s 1' \
    #     % (output_name, output_name), shell=True)


for direc in os.listdir(base_path):
    path = base_path + direc + '/'
    if not os.path.isdir(path):
        continue
    set_id = int(re.sub('\D', '', direc))
    # if set_id != 1:
    #     continue
    print set_id
    if str(set_id) not in os.listdir(output_base_path):
        os.mkdir(output_base_path + str(set_id))
    output_path = output_base_path + str(set_id) + '/'
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext == '.xlsx':
            call(['cp', path+filename, output_path])
    pool = Pool(processes=4)
    pool.map(preprocess, os.listdir(path))

