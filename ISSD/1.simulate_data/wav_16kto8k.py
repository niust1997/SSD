#-----------------niuniu----------------------
# built vox train data for TS-VAD
# 16kHz: 16000 sample per second
# overlap_length (second): overlap_frame/16000
# wav_param_list: nchannels, sampwidth, framerate, nframes,
#---------------------------------------------

import os
import numpy as np
import pdb


def m4a2wav(input_path, output_path):
    sys =  'ffmpeg -i ' + input_path + ' -ar 8000 ' + output_path
    os.system(sys)

def main():
    # m4a_list = [line.strip() for line in open('epoch49_eval_se23_mixer6_8k.list','r').readlines() ]
    # fisher_list = [line.strip() for line in open('fisher_full.list','r').readlines() ]
    # pdb.set_trace()
    path_16k = './spkdiar_result_wav_fix_16k'
    i = 0
    id_list = os.listdir(path_16k)
    id_list.sort()
    for id in id_list: 
        cur_id_path = os.path.join(path_16k, id)
        for wav_file in os.listdir(cur_id_path):
            wav_file_path = os.path.join(cur_id_path, wav_file)
            # pdb.set_trace()
            # for line in m4a_list:
                # i += 1
                # print(i)
                # pdb.set_trace()
            input_path = wav_file_path
            output_path = input_path.replace('16k', '')
            output_dir = output_path.replace(os.path.basename(output_path), '')
            # pdb.set_trace()
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print('input_path', input_path)
            print('output_path', output_path)
            print('output_dir', output_dir)
            m4a2wav(input_path, output_path)
            print('done')

if __name__ == '__main__':
    main()