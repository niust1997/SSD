import os
import numpy as np
import pdb
import soundfile as sf
import math
import prettytable as pt
import time
from collections import Counter
# method1 : xcorr + webrtcVAD
# method2 : xcorr + CLDNN_VAD
# method3 : xcorr + webrtcVAD  +  sep1 vs sep2 , to avoid speaker change detection problem

# method6 : direct use separated speech and VAD
# method7: use PIT-DER for fusion

def check_the_overlap(session):
    #nframes = 20 * 60 * 100 # 20mins
    # get the frmame-level rttm
    rttm = {}
    path_rttm='./pre_train_result/'+session+'.rttm'
    all_frame = 0
    rttm1=np.genfromtxt(path_rttm, dtype = None)
    nframes1=np.int(np.float(rttm1[-1][3]) * 100 )+np.int(np.float(rttm1[-1][4]) * 100 )
    for j in range(rttm1.shape[0]):
        session = rttm1[j][1]
        if not session in rttm.keys() :
            rttm[session] = {}
        spk = rttm1[j][-3]
        if not spk in rttm[session].keys():
            rttm[session][spk] = np.zeros(nframes1)
        start = np.int(np.float(rttm1[j][3]) * 100 )
        end = start + np.int(np.float(rttm1[j][4]) * 100)
        all_frame = all_frame + np.int(np.float(rttm1[j][4]) * 100)
        rttm[session][spk][start:end] = 1
    # pdb.set_trace()
    # get the overlap label
    for session in rttm.keys():
        num_speaker = 0
        for spk in rttm[session].keys():
            num_speaker += rttm[session][spk]
        for spk in rttm[session].keys():
            rttm[session][spk][num_speaker > 1] = 100
    
    speaker_list = list(rttm[session].keys())
    input_duration = rttm[session][speaker_list[0]][:]
    count_duration = Counter(input_duration)
    # pdb.set_trace()
    # get all frame
    all_frame1 = 0
    
    all_frame1 = int(count_duration[1.0] + count_duration[100.0])
    overlap_percent = float(count_duration[100.0] / all_frame)
    # pdb.set_trace()
    return overlap_percent





rttm_dir = './BUT_fisher/'
oracle_rttm_dir = './oracle_rttm/'
rttm_output_dir = './pre_train_result/'
files = os.listdir(rttm_dir)
tb = pt.PrettyTable()
tb.field_names = ["ID", "before","after","PITder","final"]



tmp_oracle_list = open('tmp_oracle_list.scp','w')
tmp_before_list = open('tmp_before_list.scp','w')
tmp_after_list = open('tmp_after_list.scp','w')
tmp_detect_list = open('tmp_detect_list_strategy2.scp', 'w')

for file in files:
    file_id = file.split('.')[0]
    
    oracle_rttm_path = os.path.join( oracle_rttm_dir ,  file_id+'.rttm')
    before_rttm_path = os.path.join( rttm_dir ,  file_id+'.rttm')
    result_rttm_path = os.path.join( rttm_output_dir ,  file_id+'.rttm')

    tmp_oracle_list.write(oracle_rttm_path +'\n')
    tmp_before_list.write(before_rttm_path +'\n')
    
    cmd = 'perl md-eval-v21.pl -afc -c 0  -r  {}  -s  {}  >./tmp.der'.format( oracle_rttm_path, before_rttm_path)
    os.system(cmd)
    stats = [line.strip() for line in open('tmp.der','r').readlines() if 'OVERALL' in line ]
    before_der = float( stats[0].split()[5] )
    
    cmd = 'perl md-eval-v21.pl -afc -c 0  -r  {}  -s  {}  >./tmp.der'.format( oracle_rttm_path, result_rttm_path)
    os.system(cmd)
    stats = [line.strip() for line in open('tmp.der','r').readlines() if 'OVERALL' in line ]
    after_der = float( stats[0].split()[5] )
    
    pit_der = check_the_overlap(file_id)
    #print(pit_der)
    if file_id == 'DH_DEV_0091':
        pdb.set_trace()
    # # pdb.set_trace()
########## use pitDER to determine final results
    if pit_der < 0.2:
        tmp_after_list.write(result_rttm_path+'\n')
        tb.add_row([file_id, before_der, after_der,  pit_der, after_der])
    else:
        tmp_after_list.write(before_rttm_path+'\n')
        tb.add_row([file_id, before_der, after_der,  pit_der, before_der])
        tmp_detect_list.write(file_id + '\n')
    
    
tmp_oracle_list.close()
tmp_before_list.close()
tmp_after_list.close()
tmp_detect_list.close()

cmd ='perl md-eval-v21.pl -afc -c 0  -R  tmp_oracle_list.scp    -S  tmp_before_list.scp >./tmp_before.der'
os.system(cmd)
stats = [line.strip() for line in open('./tmp_before.der').readlines() if 'OVERALL' in line ]
before_der = float( stats[-1].split()[5] )



cmd ='perl md-eval-v21.pl -afc -c 0  -R  tmp_oracle_list.scp    -S  tmp_after_list.scp >./tmp_after.der'
os.system(cmd)
stats = [line.strip() for line in open('tmp_after.der','r').readlines() if 'OVERALL' in line ]
after_der = float( stats[-1].split()[5] )

tb.add_row(['overall', before_der, 'xxx','xxx', after_der])
    
tb.sortby='ID'   
print(tb)  
    
    
    
    

# perl md-eval-v21.pl -afc -c 0  -r   ./oracle_rttm/DH_DEV_0028.rttm    -s  ./best_fisher_revised_directSep/DH_DEV_0028.rttm














    


 