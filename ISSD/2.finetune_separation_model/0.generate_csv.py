import os
import pdb
import random
#input_dir = '/work/sre/leisun8/tools/dihard2020/fine_tune/fine_tune_data_csv/'
input_dir = '../1separation_finetune/fine_tune_data_csv/'

file_ids = [ file.split('.')[0] for file in os.listdir( input_dir) ]


for file in file_ids:
    # if '0100' not in file:
        # continue
    print(file)
    output_dir = os.path.join('./training_data_withoutSNR_noNorm_webrtcvad/', file)
    output_dir_train = os.path.join(output_dir, 'train')
    output_dir_dev = os.path.join(output_dir, 'dev')
    os.makedirs( output_dir_train, exist_ok = True)
    os.makedirs( output_dir_dev, exist_ok = True)
    
    
    input_csv = os.path.join( input_dir, '{}.csv'.format(file) )
    context =[line for line in open( input_csv, 'r').readlines() ]
    context = context[1:]
    random.shuffle(context)
    
    
    #pdb.set_trace()
    with open( os.path.join(output_dir_train, 'mixture_train_mix_clean.csv'),'w' ) as f:
        f.write( 'mixture_ID,mixture_path,source_1_path,source_2_path,length\n')
        for id in range(0, 4500):
            f.write(context[id])

    with open( os.path.join(output_dir_dev, 'mixture_dev_mix_clean.csv'),'w' ) as f:
        f.write( 'mixture_ID,mixture_path,source_1_path,source_2_path,length\n')
        for id in range(4500, 5000):
            f.write(context[id])
    