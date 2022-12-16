import os
import numpy as np
import pdb
import soundfile as sf
import utils
import math
import prettytable as pt
import time
# from ifly_vad import ifly_vad
import scipy.io.wavfile as wav_io
# method1 : xcorr + webrtcVAD
# method2 : xcorr + CLDNN_VAD
# method3 : xcorr + webrtcVAD  +  sep1 vs sep2 , to avoid speaker change detection problem

# method6 : direct use separated speech and VAD
# method7: use PIT-DER for fusion

def filter_segments_rules(segments):
    seg_num = segments.shape[0]
    filtered_segments = []
    for i in range( seg_num):
        start_time = segments[i][0]
        end_time =  segments[i][1]
        if end_time - start_time < 0.1:
            continue
        else:
            filtered_segments.append( "%.3f  %.3f " %(start_time, end_time) )
    return filtered_segments

def gest_total_duration(segments):
    seg_num = segments.shape[0]
    filtered_segments = []
    length = 0
    for i in range( seg_num):
        start_time = segments[i][0]
        end_time =  segments[i][1]
        duration = end_time - start_time
        length = length + duration
    return length
    
    
def xcorr(x, y, normed=True, detrend=False, maxlags=10):
    # Cross correlation of two signals of equal length
    # Returns the coefficients when normed=True
    # Returns inner products when normed=False
    #  lags, c = xcorr(x,y,maxlags=len(x)-1)

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')
    
    if detrend:
        import matplotlib.mlab as mlab
        x = mlab.detrend_mean(np.asarray(x)) # can set your preferences here
        y = mlab.detrend_mean(np.asarray(y))
    
    c = np.correlate(x, y, mode='full')

    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    return lags, c
    
 
def read_oracle_rttm( oracle_rttm):
    spk_dict = {}
    rttm_lines = [line.strip()  for line in open(oracle_rttm).readlines() ]
    voiced_flag = np.zeros( 8000 * 10 * 60)
    for line in rttm_lines:
        spk_id = line.split()[-3]
        start_time = int(  float( line.split()[3] ) *8000 )
        duration = int(  float( line.split()[4] ) * 8000 )
        end_time = start_time + duration
        voiced_flag [ start_time-1 :end_time -1] = voiced_flag [ start_time-1 :end_time -1]  +1
    return voiced_flag

def check_overlap(new_start, new_end, oracle_voiced_flag):
    new_start = int( new_start * 8000 )
    new_end = int( new_end * 8000)
    target_list = oracle_voiced_flag[new_start-1:new_end-1].tolist()
    if 2 in target_list:
        return "right"
    else:
        return "false"


def select_main_spk (wav_data, sep1_data, sep2_data):
    length = wav_data.shape[0]
    mid_point = math.floor( length/3)
 
    c1_max =0
    c2_max =0
 
    lags1,c1 = xcorr(wav_data[:mid_point], sep1_data[:mid_point])
    lags2,c2 = xcorr(wav_data[:mid_point], sep2_data[:mid_point])
    
    c1_max = c1_max+ max(abs(c1))
    c2_max = c2_max+ max(abs(c2))
    
    lags1,c1 = xcorr(wav_data[mid_point:mid_point*2], sep1_data[mid_point:mid_point*2])
    lags2,c2 = xcorr(wav_data[mid_point:mid_point*2], sep2_data[mid_point:mid_point*2])
    c1_max = c1_max+ max(abs(c1))
    c2_max = c2_max+ max(abs(c2))
            
    lags1,c1 = xcorr(wav_data[-mid_point:], sep1_data[-mid_point:])
    lags2,c2 = xcorr(wav_data[-mid_point:], sep2_data[-mid_point:])
    c1_max = c1_max+ max(abs(c1))
    c2_max = c2_max+ max(abs(c2))
    
    if c1_max > c2_max:
        main_spk_data = sep1_data
        infer_spk_data = sep2_data 
        infer_spk_name = '2'
    else:
        main_spk_data = sep2_data
        infer_spk_data = sep1_data
        infer_spk_name = '1'

    return     main_spk_data, infer_spk_data, infer_spk_name        
        
        
        
def get_segs( vad_numpy) :
   # pdb.set_trace()
    temp =  np.append( vad_numpy[1:], 0)    
    delta = vad_numpy - temp
    
    segs_start = np.argwhere( delta ==-1)
    segs_ends = np.argwhere( delta ==1)
    #segs_ends=segs_ends[1:]
    #pdb.set_trace()
    
    if segs_start.shape[0]!=segs_ends.shape[0]:
        segs_ends=segs_ends[1:]
    assert segs_start.shape[0] == segs_ends.shape[0]
    segs_pairs = []
    
    for i in range( segs_start.shape[0] ):
        start_point = segs_start[i]+2
        end_point = segs_ends[i]+2
        #seg = '{}  {}'.format( round(start_point[0]/8000, 3), round(end_point[0]/8000, 3) )
 
        seg = '{}  {}'.format( start_point[0], end_point[0] )
        segs_pairs.append(seg)
    return  segs_pairs
        
    

def refine_results_to_oracleVAD( spk1_vad_info,spk2_vad_info, vad_info, oracle_vad_info):
    for seg in vad_info:
        start = float( seg.split()[0] )
        end = float( seg.split()[1] )
        if start == 0:
            start = 0
        else:
            start = int( start * 100 -1 )    
        end = int( end * 100 -1 )
        
        tmp_spk1 = spk1_vad_info[ start : end] 
        tmp_spk2 = spk2_vad_info[ start : end] 
        temp_oracle = oracle_vad_info[ start : end] 

        total_spk = tmp_spk1 + tmp_spk2
      #  pdb.set_trace()
        if tmp_spk1.sum() >= tmp_spk2.sum():
            tmp_spk1 [temp_oracle>total_spk ] = 1
        else:
            tmp_spk2 [temp_oracle>total_spk ] = 1
            
        spk1_vad_info[ start : end] = tmp_spk1
        spk2_vad_info[ start : end] = tmp_spk2
        
    return spk1_vad_info, spk2_vad_info


def get_overlap_ratio( spk1_vad_info, spk2_vad_info, oracle_vad_info):
    total_points = oracle_vad_info.sum()
    total_presence = spk1_vad_info + spk2_vad_info
    overlap_2spk_points = sum( total_presence ==2)
    overlap_ratio = round( overlap_2spk_points/total_points , 3)
    return overlap_ratio

   
        
rttm_dir = './best_fisher/'
oracle_rttm_dir = './oracle_rttm/'
files = os.listdir(rttm_dir)


EPOCH = 'epoch0'

rttm_output_dir = './best_fisher_revised_directSep_finetune_allFT_tmp/{}/'.format(EPOCH)
os.makedirs(rttm_output_dir, exist_ok = True)

hop=30
mode=3

# debug_items = ['DH_DEV_0006', 'DH_DEV_0044','DH_DEV_0100','DH_DEV_0209','DH_DEV_0234']
# debug_items = [ 'DH_DEV_0100' ]
 

f = open('stats.list','w')
revise_rttm_file = open('revised_rttm_all_combine.list','w')
# files = ['DH_DEV_0006.rttm']
# pdb.set_trace()
for file in files:
    file_id = file.split('.')[0]
    
    oracle_rttm_path = oracle_rttm_dir + file
    # oracle_voiced_flag = read_oracle_rttm(oracle_rttm_path)

    rttm_path = os.path.join( rttm_dir,  file)
    rttm_lines = [line.strip() for line in open( rttm_path, 'r').readlines() ]
    
    rttm_out = open( '{}{}.rttm'.format(rttm_output_dir,file_id),'w')
       
    overlap_segs_num = 0
    overlap_segs_missed_num = 0
    
    oracle_vad_path = './sad/{}.lab'.format(file_id)
    vad_info = [line.strip() for line in open(oracle_vad_path).readlines() ]
 
    max_length = float( vad_info[-1].split()[-2] )
    oracle_vad_info = np.zeros( int( max_length * 100) )
    for seg in vad_info:
        start = float( seg.split()[0] )
        end = float( seg.split()[1] )
        if start == 0:
            start = 0
        else:
            start = int( start * 100 -1 )    
        end = int( end * 100 -1 )
        oracle_vad_info[ start : end] = 1
    
    
  #  sep_wav_path = './dihard3_dev_core_false_data_8k_testing/max_sep_clean_restoreFromMod34_mod92/{}.wav'.format(file_id)
    sep_wav_path = '/yrfs1/intern/stniu/separation_finetune_ft_dihard/finetune_withoutSNR_noNorm_testing_16.22_fuxian/{}/{}/{}.wav'.format(EPOCH, file_id, file_id)
    
    
    sep1_data, _ = sf.read(sep_wav_path.replace('.wav','_sep1.wav'), dtype='float32')    # (4799616,)
    sep2_data, _ = sf.read(sep_wav_path.replace('.wav','_sep2.wav'), dtype='float32')
    freq = 8000
    sep1_vad = utils.vad(sep1_data, freq, fs_vad = freq, hoplength = hop, vad_mode=mode) # (4799616,)
    sep2_vad = utils.vad(sep2_data, freq, fs_vad = freq, hoplength = hop, vad_mode=mode)
    # pdb.set_trace()
            
    sep1_segments = utils.get_segments(sep1_vad,freq)
    sep2_segments = utils.get_segments(sep2_vad,freq)
    
    spk1_filtered_segments = filter_segments_rules(sep1_segments)
    spk2_filtered_segments = filter_segments_rules(sep2_segments)
    
    # pdb.set_trace()
    spk1_vad_info = np.zeros_like(oracle_vad_info)
    for item in spk1_filtered_segments:
        new_start =  float( item.split()[0])
        new_end =  float( item.split()[1])
        
        if new_start == 0:
            new_start = 0
        else:
            new_start = int( new_start * 100 -1 )
            
        new_end = int( new_end * 100  -1)
        spk1_vad_info[ new_start : new_end] = 1        
    spk1_vad_info = spk1_vad_info * oracle_vad_info  ### valid VAD of oracle vad
        
        
    spk2_vad_info = np.zeros_like(oracle_vad_info)
    for item in spk2_filtered_segments:
        new_start =  float( item.split()[0])
        new_end =   float( item.split()[1])
        
        if new_start == 0:
            new_start = 0
        else:
            new_start = int( new_start * 100 -1 )
            
        new_end = int( new_end * 100 -1 )
        spk2_vad_info[ new_start : new_end] = 1
    spk2_vad_info = spk2_vad_info * oracle_vad_info 

    
    spk1_vad_info,spk2_vad_info = refine_results_to_oracleVAD( spk1_vad_info,spk2_vad_info, vad_info, oracle_vad_info)
    
    
    segs_pairs_1 = get_segs( spk1_vad_info)
    segs_pairs_2 = get_segs( spk2_vad_info)

    
    for seg in segs_pairs_1:
        start = round( float(seg.split()[0])/100, 3)
        duration =  round( (float( seg.split()[-1]) - float( seg.split()[0]))/100 , 3)
        line = 'SPEAKER {} 1 {} {} <NA> <NA> {} <NA> <NA>'.format(file_id, start, duration, '1' )
        rttm_out.write(line+'\n')
        
  
    for seg in segs_pairs_2:
        start = round( float(seg.split()[0])/100, 3)
        duration = round( (float( seg.split()[-1]) - float( seg.split()[0]))/100 , 3)
        line = 'SPEAKER {} 1 {} {} <NA> <NA> {} <NA> <NA>'.format(file_id, start, duration, '2' )
        rttm_out.write(line+'\n')
    rttm_out.close()
     
 
tb = pt.PrettyTable()
tb.field_names = ["ID", "before","after","PITder","final"]



tmp_oracle_list = open('tmp_oracle_list.scp','w')
tmp_before_list = open('tmp_before_list.scp','w')
tmp_after_list = open('tmp_after_list.scp','w')

for file in files:
    file_id = file.split('.')[0]
    
    oracle_rttm_path = os.path.join( oracle_rttm_dir ,  file_id+'.rttm')
    before_rttm_path = os.path.join( rttm_dir ,  file_id+'.rttm')
    result_rttm_path = os.path.join( rttm_output_dir ,  file_id+'.rttm')

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
    
    
    tmp_oracle_list.write( oracle_rttm_path +'\n')
    tmp_before_list.write( before_rttm_path +'\n')
########## use pitDER to determine final results
    if pit_der < 100:
        tmp_after_list.write(result_rttm_path+'\n')
        tb.add_row([file_id, before_der, after_der,  pit_der, after_der])
    else:
        tmp_after_list.write(before_rttm_path+'\n')
        tb.add_row([file_id, before_der, after_der,  pit_der, before_der])
    
    
tmp_oracle_list.close()
tmp_before_list.close()
tmp_after_list.close()

cmd ='perl md-eval-v21.pl -afc -c 0  -R  tmp_oracle_list.scp    -S  tmp_before_list.scp >./tmp_before.der'
os.system(cmd)
stats = [line.strip() for line in open('./tmp_before.der').readlines() if 'OVERALL' in line ]
before_der = float( stats[-1].split()[5] )



cmd ='perl md-eval-v21.pl -afc -c 0  -R  tmp_oracle_list.scp    -S  tmp_after_list.scp >./tmp_after.der'
os.system(cmd)
stats = [line.strip() for line in open('tmp_after.der','r').readlines() if 'OVERALL' in line ]
after_der = float( stats[-1].split()[5] )

tb.add_row(['overall', before_der, 'xxx','xxx', after_der])
    
tb.sortby='final'   
print(tb)  
    
    
    
    

# perl md-eval-v21.pl -afc -c 0  -r   ./oracle_rttm/DH_DEV_0028.rttm    -s  ./best_fisher_revised_directSep/DH_DEV_0028.rttm














    


 