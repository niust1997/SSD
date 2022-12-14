import os
import soundfile as sf
import pdb


fisher_dict = {}
dev_fisher = open('./epoch0_1417', 'r').readlines()
for line in dev_fisher:
    file_id = line.split()[1]
    if file_id not in fisher_dict.keys():
        fisher_dict[file_id] = []
    fisher_dict[file_id].append(line) 

    
out_dir = './epoch0_1417_split/'
os.makedirs(out_dir, exist_ok=True)
for file_id in fisher_dict.keys():
    out_path = out_dir + file_id + '.rttm'
    f = open( out_path,'w')
    for line in fisher_dict[file_id]:
        f.write(line)
    f.close()