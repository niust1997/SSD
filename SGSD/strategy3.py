import os
import numpy as np
import pdb
import soundfile as sf
# import utils
import math
import prettytable as pt
import time
# method1 : xcorr + webrtcVAD
# method2 : xcorr + CLDNN_VAD
# method3 : xcorr + webrtcVAD  +  sep1 vs sep2 , to avoid speaker change detection problem

# method6 : direct use separated speech and VAD
# method7: use PIT-DER for fusion

rttm_dir = './BUT_fisher/'
oracle_rttm_dir = './oracle_rttm/'
files = os.listdir(rttm_dir)


rttm_output_dir = './pre_train_result'
# rttm_output_dir_fix = './pre_train_result_fix'
    
tb = pt.PrettyTable()
tb.field_names = ["ID", "before","after","PITder","final"]



tmp_oracle_list = open('tmp_oracle_list.scp','w')
tmp_before_list = open('tmp_before_list.scp','w')
tmp_after_list = open('tmp_after_list.scp','w')
tmp_detect_list = open('tmp_detect_list.scp', 'w')

for file in files:
    file_id = file.split('.')[0]
    
    oracle_rttm_path = os.path.join( oracle_rttm_dir ,  file_id+'.rttm')
    before_rttm_path = os.path.join( rttm_dir ,  file_id+'.rttm')
    result_rttm_path = os.path.join( rttm_output_dir ,  file_id+'.rttm')
    # result_rttm_path_fix = os.path.join(rttm_output_dir_fix, file_id+'.rttm')

    cmd = 'perl md-eval-v21.pl -afc -c 0  -r  {}  -s  {}  >./tmp.der'.format( oracle_rttm_path, before_rttm_path)
    os.system(cmd)
    stats = [line.strip() for line in open('tmp.der','r').readlines() if 'OVERALL' in line ]
    before_der = float( stats[0].split()[5] )
    
   
    cmd = 'perl md-eval-v21.pl -afc -c 0  -r  {}  -s  {}  >./tmp.der'.format( oracle_rttm_path, result_rttm_path)
    os.system(cmd)
    stats = [line.strip() for line in open('tmp.der','r').readlines() if 'OVERALL' in line ]
    after_der = float( stats[0].split()[5] )
    
    
    
    cmd = 'perl md-eval-v21.pl -afc -c 0  -r  {}  -s  {}  >./tmp.der'.format( before_rttm_path, result_rttm_path)
    os.system(cmd)
    stats = [line.strip() for line in open('tmp.der','r').readlines() if 'OVERALL' in line ]
    pit_der = float( stats[0].split()[5] )
    
    # pdb.set_trace()
    
    tmp_oracle_list.write( oracle_rttm_path +'\n')
    tmp_before_list.write( before_rttm_path +'\n')
########## use pitDER to determine final results
    if pit_der < 26:
        tmp_after_list.write(result_rttm_path+'\n')
        tb.add_row([file_id, before_der, after_der,  pit_der, after_der])
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














    


 