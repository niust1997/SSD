# -*- coding:utf-8 -*-
import os
import numpy as np
import argparse
import csv
import time
import random
from scipy import signal
import h5py
from sklearn import preprocessing
import pdb
import scipy.io.wavfile as wav
import random


def combine_list(out_list, list1, list2, list3):
	assert os.path.exists(list1), 'list1 is not valid'
	assert os.path.exists(list2), 'list2 is not valid'
	assert os.path.exists(list3), 'list3 is not valid'
	in1 = open(list1)
	list1 = in1.readlines()
	in1.close()
	
	in2 = open(list2)
	list2 = in2.readlines()
	in2.close()
	
	in3 = open(list3)
	list3 = in3.readlines()
	in3.close() 
	
	out = open(out_list,'w')
	out.writelines(list1)
	out.write('\n')
	out.writelines(list2)
	out.write('\n')
	out.writelines(list3)
	out.close()


def mix_audio_normal(args):
    input_list_speech = args.target_spk  # target spk list
    other_spk1_list = args.other_spk1 # spk2 list
    other_spk2_list = args.other_spk2 # spk2 list
    other_spk3_list = args.other_spk3 # spk2 list
    workdir  = args.workplace
    start_id = args.start_id 
    end_id = args.end_id
    snr = args.snr
	
    if not os.path.exists(workdir):
		os.makedirs(workdir)
	
    scp_dir = os.path.join(workdir,'scp')
    if not os.path.exists(scp_dir): 
		os.makedirs(scp_dir)
	
    assert os.path.exists(input_list_speech), 'input_list_speech is not valid'
    target_list = os.path.join(scp_dir)+'/target_spk.list'  # target speaker source
    os.system('cp %s %s' %(input_list_speech, target_list)) 
	
    other_spk_all_list = scp_dir + '/other_spk_all.list'  # other speakers source
    if not os.path.exists(other_spk_all_list):
		combine_list(other_spk_all_list, other_spk1_list, other_spk2_list,other_spk3_list )
	
	
    output_dir = os.path.join(workdir,'mixed_speech')+'/snr'+ str(snr)
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    f_speech = open(target_list)
    speech_list = f_speech.readlines()
    speech_num = len(speech_list)

    f_inter = open(other_spk_all_list)
    inter_list = f_inter.readlines()
    inter_num = len(inter_list)
	
    fea_mix = open( scp_dir + '/SNR'+ str(snr)+'_'+str(start_id)+'_'+str(end_id)+ '_fea_mix.list' , 'w' ) # other speakers source
    fea_target = open(scp_dir + '/SNR'+ str(snr)+'_'+str(start_id)+'_'+str(end_id)+ '_fea_target.list' , 'w' )
    fea_other = open(scp_dir + '/SNR'+ str(snr)+'_'+str(start_id)+'_'+str(end_id)+ '_fea_other.list' , 'w' )
	
    for i in range(start_id, end_id):
        rand1 = np.random.randint(0, speech_num, size=1)[0]
        rand2 = np.random.randint(0, inter_num, size=1)[0]
        inputfile1 = speech_list[rand1].strip('\n')
        inputfile2 = inter_list[rand2].strip('\n')
		#need save csv file ???

        (mixed_audio,speech_audio, noise_audio) = mix_audio_NoSingleSpk(inputfile1, inputfile2, snr, output_dir )

        mix_wav = os.path.join(output_dir,str(i))+'_mix.wav'
        wav.write(mix_wav,16000,  np.int16(mixed_audio*32767))
        mix_raw = os.path.join(output_dir,str(i))+'_mix.raw'
        os.system('./bin/WAV2RAW %s %s'%(mix_wav,mix_raw))
        mix_fea = os.path.join(output_dir,str(i))+'_mix.fea'
        os.system('./bin/Wav2LogSpec_be -F RAW -fs 16 %s %s' %(mix_raw,mix_fea) )
        fea_mix.write(os.path.abspath(mix_fea)+'\n')
		
        s_wav = os.path.join(output_dir,str(i))+'_s.wav'
        wav.write(s_wav,16000,   np.int16(speech_audio*32767))
        s_raw = os.path.join(output_dir,str(i))+'_s.raw'
        os.system('./bin/WAV2RAW %s %s'%(s_wav,s_raw) )
        s_fea = os.path.join(output_dir,str(i))+'_s.fea'
        os.system('./bin/Wav2LogSpec_be -F RAW -fs 16 %s %s' %(s_raw,s_fea) )
        fea_target.write(os.path.abspath(s_fea)+'\n')
		
        n_wav = os.path.join(output_dir,str(i))+'_n.wav'
        wav.write(n_wav,16000,   np.int16(noise_audio*32767))
        n_raw = os.path.join(output_dir,str(i))+'_n.raw'
        os.system('./bin/WAV2RAW %s %s'%(n_wav,n_raw) )
        n_fea = os.path.join(output_dir,str(i))+'_n.fea'
        os.system('./bin/Wav2LogSpec_be -F RAW -fs 16 %s %s' %(n_raw,n_fea) )
        fea_other.write(os.path.abspath(n_fea)+'\n')
	
    fea_mix.close()
    fea_target.close()
    fea_other.close()
	

def mix_audio_NoSingleSpk( input_wav1,  input_wav2,  snr, output_dir, location = 'T'):
        rate, input_data1 = wav.read(input_wav1) # as target
        rate, input_data2 = wav.read(input_wav2) # as interference

        input_data1 = input_data1.astype('float32') / 32768 # int16 to float
        input_data2 = input_data2.astype('float32') / 32768 
        mix_snr = snr

        len1 =  len(input_data1)
        len2 =  len(input_data2)		
       # pdb.set_trace()
        if len2<= len1:  #噪声长度小于speech时，要重复； 不然直接截取
            repeat_data = np.tile( input_data2,  int(np.ceil( float(len1)/ float(len2) )) )
            input_data2 = repeat_data[0: len1] #截取跟speech一样长
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




    
if __name__== '__main__' : #“Make a script both importable and executable”
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    mix_speech = subparsers.add_parser('mix_speech_normal')
    mix_speech.add_argument('--target_spk', type=str, required = True)
    mix_speech.add_argument('--other_spk1', type =str, required=True)
    mix_speech.add_argument('--other_spk2', type =str, required=True)
    mix_speech.add_argument('--other_spk3', type =str, required=True)
    mix_speech.add_argument('--workplace', type=str, required = True)
    mix_speech.add_argument('--start_id', type=int, required = True )
    mix_speech.add_argument('--end_id', type=int, required=True)
    mix_speech.add_argument('--snr', type=int, required=True)
    args = parser.parse_args()


    
    if args.mode == 'mix_speech_normal':
        mix_audio_normal(args)
        print ('all Done!!!!!!!! \n')

# python prepare_data.py mix_speech 
# --input_list_speech ./input1.txt --input_list_interfence ./input2.txt 
# --output_dir /Users/sunlei/codes/mix_speech/1_1000 --start_id 1 --end_id 10








