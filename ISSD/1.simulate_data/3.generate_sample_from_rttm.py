import os
import soundfile as sf
import pdb
from tqdm import tqdm

  
file_ids = [  file.split('.')[0] for file in os.listdir( './epoch0_1417_split_fix/') if file.endswith('.rttm') ] 
# pdb.set_trace()
for file_id in tqdm(file_ids):
    result_rttm = './epoch0_1417_split_fix/{}.rttm'.format(file_id)
    wav_file = '/work/sre/leisun8/tools/dihard2020/dev_wav_8k/{}.wav'.format(file_id)
    wav_data, fs = sf.read(wav_file, dtype='float32' )

    output_dir = './spkdiar_result_wav_fix/{}'.format(file_id)
    os.makedirs(output_dir, exist_ok=True)

    rttm = [line.strip() for line in open(result_rttm,'r').readlines() ]
    # pdb.set_trace()
    id = 1
    for line in rttm:
        spk_name = line.split()[-3]
        start_time = float( line.split()[3] )
        duration = float( line.split()[4] )
        end_time = start_time + duration 
        # pdb.set_trace()
        
        tmp_data = []
        # slice the segment which > 5s
        if duration >= 5:
            slice_num = int(duration // 3) + 1
            current_start_time = start_time
            for k in range(slice_num):
                current_end_time = start_time + (k + 1) * 3
                if current_end_time < end_time:
                    tmp_data.append(wav_data[int(current_start_time * fs) : int(current_end_time * fs)])
                    current_start_time = current_end_time
                else:
                    tmp_data.append(wav_data[int(current_start_time * fs) : int(end_time * fs)])

        elif duration > 0.5:
            tmp_data.append(wav_data[int(start_time * fs) : int(end_time * fs)])
        
        else:
            continue
        
        for j in range(len(tmp_data)):
            current_temp_data = tmp_data[j]
            output_path = os.path.join(output_dir, 'spk{}_id{}.wav'.format(spk_name, id))
            id = id + 1
            sf.write(output_path, current_temp_data, fs)
        
        
        