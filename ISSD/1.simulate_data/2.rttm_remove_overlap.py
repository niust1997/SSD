import os
import soundfile as sf
import pdb
import numpy as np
from tqdm import tqdm

file_ids = [  file.split('.')[0] for file in os.listdir( './epoch0_1417_split/') if file.endswith('.rttm') ]

for file_id in tqdm(file_ids):
    # print(file_id)
    result_rttm = './epoch0_1417_split/{}.rttm'.format(file_id)
    wav_file = '/work/sre/leisun8/tools/dihard2020/dev_wav_8k/{}.wav'.format(file_id)
    wav_data, fs = sf.read(wav_file, dtype = 'float32')
    nframe = wav_data.shape[0] # frame shift 1/8000 second
    output_dir = './epoch0_1417_split_fix/'
    os.makedirs(output_dir, exist_ok = True)
    
    # rttm = [line.strip() for line in open(result_rttm,'r').readlines()]
    
    # get the rttm label
    rttm = {}
    current_frames = nframe
    for line in open(result_rttm):
        line = line.split(" ")
        line=[item for item in line if item!='']
        # pdb.set_trace()
        session = line[1]
        spk = line[7]
        # pdb.set_trace()
        if not session in rttm.keys():
            rttm[session] = {}
        if not spk in rttm[session].keys():
            rttm[session][spk] = np.zeros([current_frames], dtype=np.uint8)
        
        start = np.int(np.float(line[3]) * fs)
        end = np.int(np.float(line[4]) * fs) + start
        if start > current_frames:
            continue
        if start < current_frames and end > current_frames:
            end = current_frames
        rttm[session][spk][start: end] = 1
    
    for session in rttm.keys():
        num_speaker = 0
        for spk in rttm[session].keys():
            num_speaker += rttm[session][spk]
        for spk in rttm[session].keys():
            rttm[session][spk][num_speaker > 1] = 100
    
    output_path = output_dir + '{}.rttm'.format(file_id)
    session_label = rttm
    # pdb.set_trace()
    with open(output_path, "w") as OUT:
        for session in session_label.keys():
            num_speaker = len(session_label[session].keys())
            spk_list = []
            for per_spk in session_label[session].keys():
                spk_list.append(per_spk)
            spk_list.sort()
            for k in range(num_speaker):
                i = 0
                spk = spk_list[k]
                # pdb.set_trace()
                while i < len(session_label[session][spk]):
                    if session_label[session][spk][i] == 1 and session_label[session][spk][i] != 100:
                        start = i
                        durance = 1
                        i += 1
                        while i < len(session_label[session][spk]) and session_label[session][spk][i] == 1 and session_label[session][spk][i] != 100:
                            i += 1
                            durance += 1
                        #if min_segments > durance:
                        #    continue
                        end = start + durance
                        #print("{} {}".format(start, end))
                        OUT.write("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(session, start / fs, durance / fs, str(k + 1)))
                    i += 1
        
        
        
