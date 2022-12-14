import os
import numpy as np
import pdb
import soundfile as sf
import math
import prettytable as pt
import time
# method1 : xcorr + webrtcVAD
# method2 : xcorr + CLDNN_VAD
# method3 : xcorr + webrtcVAD  +  sep1 vs sep2 , to avoid speaker change detection problem

# method6 : direct use separated speech and VAD
# method7: use PIT-DER for fusion

def get_speaker_time_ratio(input_rttm):
    spk1_time = 0
    spk2_time = 0
    lines = [line.strip() for line in open(input_rttm,'r').readlines() ]
    for line in lines:
        line_duration = float(line.split(' ')[4])
        line_speaker = line.split(' ')[7]
        if line_speaker == '1':
            spk1_time = spk1_time + line_duration
        elif line_speaker == '2':
            spk2_time = spk2_time + line_duration
        else:
            print('error')
    if spk1_time >= spk2_time:
        ratio = spk2_time / spk1_time
    else:
        ratio = spk1_time / spk2_time
    return ratio
    # pdb.set_trace()





rttm_dir = './BUT_fisher/'
oracle_rttm_dir = './oracle_rttm/'
rttm_output_dir = './pre_train_result/'
files = os.listdir(rttm_dir)
tb = pt.PrettyTable()
tb.field_names = ["ID", "before","after","PITder","final"]



tmp_oracle_list = open('tmp_oracle_list.scp','w')
tmp_before_list = open('tmp_before_list.scp','w')
tmp_after_list = open('tmp_after_list.scp','w')
tmp_detect_list = open('tmp_detect_list_strategy1.scp', 'w')

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
    
    pit_der = get_speaker_time_ratio(result_rttm_path)
    # if file_id == 'DH_DEV_0094':
        # pdb.set_trace()
########## use pitDER to determine final results
    if pit_der > 0.4:
        tmp_after_list.write(result_rttm_path+'\n')
        tb.add_row([file_id, before_der, after_der,  pit_der, str(after_der)])
    else:
        tmp_after_list.write(before_rttm_path+'\n')
        tb.add_row([file_id, before_der, after_der,  pit_der, '*' + str(before_der)])
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














    


 