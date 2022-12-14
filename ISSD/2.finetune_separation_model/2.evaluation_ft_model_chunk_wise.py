import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
from pprint import pprint
import numpy as np
import pdb
from asteroid.torch_utils import load_state_dict_in
from asteroid.utils import tensors_to_device
from model import make_model_and_optimizer
import scipy.io.wavfile as wavfile

def find_latest_model(dir):
    file_lists = [file  for file in os.listdir(dir) if '_ckpt_epoch' in file ]
    if len(file_lists) >=1:
        file_lists.sort( key=lambda fn: os.path.getmtime(dir+"/"+fn) if not os.path.isdir(dir+"/"+fn) else 0 )
        print("the latest file is " + file_lists[-1] )
        model_path = os.path.join( dir, file_lists[-1] )
        return model_path
    else:
        return None
        
        

GPU_ID = 2

N_SRC = 2

COPY_ORI = 1  # for compare

IF_RAW = 0


evaluation_wav_dir = '/work/sre/leisun8/tools/dihard2020/dev_wav_8k/'



#ft_list = [line.strip() for line in open('./scp/ft_list_fisher_dev.scp','r').readlines()]

ft_list = [line  for line in os.listdir('./training_data_withoutSNR_noNorm_new_data')  ]

#debug_items = ['DH_DEV_0006' , 'DH_DEV_0044','DH_DEV_0100','DH_DEV_0209','DH_DEV_0234']
#debug_items = [ 'DH_DEV_0100' ]
# ft_list = [ 'DH_DEV_0075' ]
#pdb.set_trace()
epoch = 0
for id in ft_list:
    print(id)
    #evaluation_model_dir  = './exp/train_ft_{}_withoutSNR_noNorm_sep_clean/'.format(id)
    #evaluation_model_path = find_latest_model(evaluation_model_dir)
    evaluation_model_path = './exp_withoutSNR_noNorm_16.22_fuxian/train_ft_{}_sep_clean/_ckpt_epoch_{}.ckpt'.format(id, epoch)
    eval_save_dir = './finetune_withoutSNR_noNorm_testing_16.22_fuxian_chunkwise/epoch{}/{}/'.format(epoch, id)
    #eval_save_dir = './epoch10_debug/{}/'.format(id)
    os.makedirs( eval_save_dir, exist_ok=True)
    if os.path.getsize(eval_save_dir) > 3:
        continue

    evaluation_model_dir = os.path.dirname(evaluation_model_path)
    model_conf = os.path.join(evaluation_model_dir,'conf.yml')


    with open(model_conf) as f:
        train_conf = yaml.safe_load(f)
       
        model, _ = make_model_and_optimizer(train_conf) 
        
        checkpoint = torch.load(evaluation_model_path, map_location='cpu')
        state = checkpoint['state_dict']
        state_copy = state.copy()
        # Remove unwanted keys
        for keys, values in state.items():
            if keys.startswith('loss'):
                del state_copy[keys]
                print(keys)
        model = load_state_dict_in(state_copy, model)
        
        if not torch.cuda.is_available():
            raise Exception ("cuda is not available")
        else:
            torch.cuda.set_device(GPU_ID)
            model.cuda()
            
        model_device = next(model.parameters()).device
        
        if not os.path.exists(eval_save_dir):
            os.makedirs(eval_save_dir)

        torch.no_grad().__enter__()
        
        wav_path = os.path.join(evaluation_wav_dir, id+'.wav')
        wav_out_path = os.path.join( eval_save_dir, id+'.wav')
            
        if IF_RAW :
            cmd = 'sox -t raw -c 1 -e signed-integer -b 16 -r 8000 {} tmp.wav'.format(wav_path)
            os.system(cmd)
            _, s = wavfile.read('tmp.wav')
        else:
            _, s = wavfile.read(wav_path)
            #_, s = wavfile.read(wav_path)
        if s.dtype == np.int16:
            s = np.float32(s) / 32768 # transform it
        elif s.dtype == np.float32:
            s = s
        else:
            raise Exception("unknown wave format")
            
        
        # mix = torch.from_numpy(s)
        # mix = tensors_to_device( mix , device=model_device)

           # pdb.set_trace()
        # est_sources = model(mix.unsqueeze(0)) #torch.Size([1, 60320])

        # est_sources_np = est_sources.squeeze().cpu().data.numpy()

        #  pdb.set_trace()
          
        fs = 8000
        Nh = 30 # 9600
        Nc = 10 # 6400
        Nf = 20 # 3200
        total_frame = s.shape[0]
        # s = s[0: 200000]
        cur_frame = 0
        window_size = int((Nh + Nc + Nf) * fs)
        hop_size = int(Nc * fs)
        feature_list = []
        while(cur_frame < total_frame):
            if cur_frame + window_size <= total_frame:
                feature_list.append((cur_frame, cur_frame + window_size))
                cur_frame += hop_size
            else:
                cur_frame = max(0, total_frame - window_size)
                feature_list.append((cur_frame, total_frame))
                break
        # print(cur_frame)
        # pdb.set_trace()
        full_audio = s[:]
        est_sources_list = []

        # first_est
        s = full_audio[: int((Nh + Nc) * fs)]
        s = torch.from_numpy(s)
        mixture = tensors_to_device( s , device=model_device)
        # pdb.set_trace()
        est_sources_time = model(mixture.unsqueeze(0)).detach().squeeze().cpu()
        # pdb.set_trace()
        est_sources_time = est_sources_time[:, : int(Nh * fs)]
        est_sources_list.append(est_sources_time)

        for start_end in feature_list[:-1]:
            print(start_end)
            start = start_end[0]
            end = start_end[1]
            s = full_audio[start: end]
            s = torch.from_numpy(s)
            # pdb.set_trace()
            mixture = tensors_to_device( s , device=model_device)
            # pha = tensors_to_device(pha, device=model_device)
            est_sources_time = model(mixture.unsqueeze(0)).detach().squeeze().cpu()
            est_sources_time = est_sources_time[:, int(Nh * fs) : int((Nh + Nc) * fs)]
            est_sources_list.append(est_sources_time)
            

        # last_est
        start_end = feature_list[-1]
        start = start_end[0]
        end = start_end[1]
        s = full_audio[start: end]
        s = torch.from_numpy(s)
        mixture = tensors_to_device( s , device=model_device)
        # pha = tensors_to_device(pha, device=model_device)
        est_sources_time = model(mixture.unsqueeze(0)).detach().squeeze().cpu()
        last_start = int(feature_list[-2][0] + (Nh + Nc) * fs - start)

        est_sources_time = est_sources_time[:, last_start:]
        est_sources_list.append(est_sources_time)
        est_sources = torch.cat(est_sources_list, dim=1)
        est_sources_np = est_sources.numpy()
        # pdb.set_trace()
        MAX_S = np.max(np.abs(full_audio)) 
        
        
        
        if N_SRC == 2:
                
            ratio = MAX_S / np.max(np.abs(est_sources_np[0,:]))        
            sf.write( wav_out_path.replace('.wav','_sep1.wav'), est_sources_np[0,:]*ratio, 8000)
                
            ratio = MAX_S / np.max(np.abs(est_sources_np[1,:]))    
            sf.write( wav_out_path.replace('.wav','_sep2.wav'), est_sources_np[1,:]*ratio, 8000)
                
            if COPY_ORI:
                os.system('cp {} {}'.format(wav_path, eval_save_dir))
                
        # pdb.set_trace()
        
        
        
        
        
    
    