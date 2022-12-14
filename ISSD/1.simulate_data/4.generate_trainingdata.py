import os
import numpy as np
import pandas as pd
# import pyloudnorm as pyln
# import soundfile as sf  
from tqdm import tqdm
import pdb
import random
import sys
import scipy.io.wavfile as wav_io


    
def mix_audio( input_wav1,  input_wav2,  snr, output_dir, location = 'T'):
        rate, input_data1 = wav_io.read(input_wav1) # as target
        rate, input_data2 = wav_io.read(input_wav2) # as interference
        input_data1 = input_data1.astype('float32')
        input_data2 = input_data2.astype('float32')
        
        mix_snr = snr

        len1 =  len(input_data1)
        len2 =  len(input_data2)        
       # pdb.set_trace()
        if len2<= len1:   
            repeat_data = np.tile( input_data2,  int(np.ceil( float(len1)/ float(len2) )) )
            input_data2 = repeat_data[0: len1]  
        else:
            noise_onset = np.random.randint(low =0, high = len2 - len1 , size=1)[0]
            noise_offset = noise_onset + len1
            input_data2 = input_data2[noise_onset: noise_offset]
            
      #  pdb.set_trace()
        assert len(input_data1)==len(input_data2), 'two sequence lengths are not equal!!!'
        scaler = get_scaler(input_data1, input_data2,  snr)

        input_data1 *= scaler
        location = random.choice('TF')
        (mixed_audio, speech_audio, noise_audio) = additive_mixing(input_data1, input_data2, location=location)

        return mixed_audio,speech_audio, noise_audio    
    




def mix_audio_withoutSNR( input_wav1,  input_wav2,  snr, output_dir, location = 'T'):
        # input_data1,_ = sf.read(input_wav1, dtype='float32') # as target
        # input_data2,_ = sf.read(input_wav2, dtype='float32') # as interference
        rate, input_data1 = wav_io.read(input_wav1) # as target
        rate, input_data2 = wav_io.read(input_wav2) # as interference
        input_data1 = input_data1.astype('float32')
        input_data2 = input_data2.astype('float32')

        mix_snr = snr

        len1 =  len(input_data1)
        len2 =  len(input_data2)        
       # pdb.set_trace()
        if len2<= len1:   
            repeat_data = np.tile( input_data2,  int(np.ceil( float(len1)/ float(len2) )) )
            input_data2 = repeat_data[0: len1]  
        else:
            noise_onset = np.random.randint(low =0, high = len2 - len1 , size=1)[0]
            noise_offset = noise_onset + len1
            input_data2 = input_data2[noise_onset: noise_offset]
            
      #  pdb.set_trace()
        assert len(input_data1)==len(input_data2), 'two sequence lengths are not equal!!!'
       # scaler = get_scaler(input_data1, input_data2,  snr)

       # input_data1 *= scaler
        location = random.choice('TF')
        speech_audio = input_data1
        noise_audio = input_data2
        mixed_audio = speech_audio+noise_audio
       # (mixed_audio, speech_audio, noise_audio) = additive_mixing(input_data1, input_data2, location=location)
        
        return mixed_audio,speech_audio, noise_audio  



    
def additive_mixing(speech_audio, noise_audio, location):
    mixed_audio = speech_audio + noise_audio
    alpha =  1. / np.max( np.abs(mixed_audio))
    speech_audio *= alpha
    noise_audio *= alpha
    mixed_audio *= alpha
    return mixed_audio, speech_audio, noise_audio


def get_scaler( speech_data, noise_data, snr ) :
    # first calculate RMS= root mean square
    speech_rms = rms(speech_data)
    noise_rms = rms(noise_data)
    original_rms_ratio = speech_rms/noise_rms
    target_rms_ratio = 10. ** (float(snr) / 20.)  # snr = 20 * lg(rms(s) / rms(n))
    scale_factor = target_rms_ratio / original_rms_ratio
    return scale_factor


def rms(x):
    # first calculate RMS= root mean square
    return np.sqrt( np.mean (np.abs(x) **2 , axis=0, keepdims = False)  )
    
    
def generate_ft_data(file_id, target_files, inter_files, root_dir ):
    spk1_num = len(target_files)
    spk2_num = len(inter_files)
    
    
    snrs=[0,-1,1,-3,3]
    
    s1_out_dir = os.path.join( root_dir, 's1')
    s2_out_dir = os.path.join( root_dir, 's2')
    mix_out_dir = os.path.join( root_dir, 'mix_both')
    
    
    cvs_path = './fine_tune_data_csv/{}.csv'.format(file_id)
    output_csv = open( cvs_path ,'w')
    output_csv.write('mixture_ID,mixture_path,source_1_path,source_2_path,noise_path,length\n')
    
    for snr in snrs:
        snr_dir = os.path.join( root_dir, 'snr{}'.format(snr))
        os.makedirs( snr_dir, exist_ok=True)
        for id in range(1000):
        
            rand1 = np.random.randint(0, spk1_num, size=1)[0]
            rand2 = np.random.randint(0, spk2_num, size=1)[0]
            inputfile1 = target_files[rand1] 
            inputfile2 = inter_files[rand2] 

            (mixed_audio,speech_audio, noise_audio) = mix_audio_withoutSNR(inputfile1, inputfile2, snr, output_dir )
            
            
            
            
            
            s1_out_path = os.path.join( snr_dir, 'snr{}_id{}_spk1.wav'.format(snr,id) )
            s2_out_path = os.path.join( snr_dir, 'snr{}_id{}_spk2.wav'.format(snr,id) )
            mix_out_path = os.path.join( snr_dir, 'snr{}_id{}_mix.wav'.format(snr,id) )
            
            mixture_ID = 'snr{}_id{}'.format(snr,id)
            length = mixed_audio.shape[0]
            csv_line = '{},{},{},{},{}\n'.format( mixture_ID, os.path.abspath(mix_out_path), os.path.abspath(s1_out_path), os.path.abspath(s2_out_path), length)
            output_csv.write(csv_line)
            
            wav_io.write(s1_out_path, 8000, np.int16(speech_audio))
            wav_io.write(s2_out_path, 8000, np.int16(noise_audio))
            wav_io.write(mix_out_path, 8000, np.int16(mixed_audio))
        
    output_csv.close()
 #   exit()
    #mixture_ID,mixture_path,source_1_path,source_2_path,noise_path,length
    
    


from tqdm import tqdm
#file_id = 'DH_DEV_0003'
file_ids = [  file.split('.')[0]   for file in os.listdir( './epoch0_1417_split/')] 
file_ids.sort()
v2 = sys.argv[1]
k = int(v2)
# pdb.set_trace()
debug_items = file_ids[(k - 1) * 16: k * 16] # ['DH_DEV_0006', 'DH_DEV_0033', 'DH_DEV_0044','DH_DEV_0100','DH_DEV_0209','DH_DEV_0234']
print(debug_items)
# pdb.set_trace()
for file_id in tqdm(debug_items):
    if file_id not in debug_items:
        continue
    # pdb.set_trace()
    # print(file_id)
    wav_seg_dir = './spkdiar_result_wav_fix/{}'.format(file_id)
    output_dir = './spkdiar_result_wav_fix/{}_finetune_data_debug_withoutSNR_noNorm/'.format(file_id)
    os.makedirs(output_dir, exist_ok=True)
    
    spk1_files = [ os.path.join(wav_seg_dir, file) for file in os.listdir( wav_seg_dir) if file.endswith('.wav') and 'spk1' in file ] 
    spk2_files = [ os.path.join(wav_seg_dir, file) for file in os.listdir( wav_seg_dir) if file.endswith('.wav') and 'spk2' in file ] 
    
    generate_ft_data( file_id, spk1_files, spk2_files, output_dir )
    
    