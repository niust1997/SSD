import os
import argparse
import json
import pdb
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from asteroid.data.librimix_dataset import LibriMix
# from my_librimix_reader import myLibriMix
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.torch_utils import load_state_dict_in
from model import make_model_and_optimizer

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
# This can be changed: `python train.py --gpus 0,1` will only train on 2 GPUs.
parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, help='list of GPUs', default='0')
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Full path to save best validation model')
                    
parser.add_argument('--pretrained_model', default=None,
                    help='Full path of the pretrained model')

def main(conf):
    print( conf['data']['train_dir'] )
    train_set = LibriMix(csv_dir=conf['data']['train_dir'],
                         task=conf['data']['task'],
                         sample_rate=conf['data']['sample_rate'],
                         n_src=conf['data']['n_src'],
                         segment=conf['data']['segment'])

    val_set = LibriMix(csv_dir=conf['data']['valid_dir'],
                       task=conf['data']['task'],
                       sample_rate=conf['data']['sample_rate'],
                       n_src=conf['data']['n_src'],
                       segment=conf['data']['segment'])

    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=conf['training']['batch_size'],
                              num_workers=conf['training']['num_workers'],
                              drop_last=True)

    val_loader = DataLoader(val_set, shuffle=True,
                            batch_size=conf['training']['batch_size'],
                            num_workers=conf['training']['num_workers'],
                            drop_last=True)
    conf['masknet'].update({'n_src': conf['data']['n_src']})

    # Define model and optimizer in a local function (defined in the recipe).
    # Two advantages to this : re-instantiating the model and optimizer
    # for retraining and evaluating is straight-forward.
    model, optimizer = make_model_and_optimizer(conf)
    print(model)
    
    pretrained_model =  conf['main_args']['pretrained_model']
    checkpoint =   torch.load(pretrained_model)
    state = checkpoint['state_dict']
    state_copy = state.copy()
    model = load_state_dict_in(state_copy, model)
    print("\nloading model state from pretrained model path: {}\n".format( pretrained_model) )
    
    print("start fine-tune \n\n")
    ## md['state_dict']['model.masker.mask_net.1.weight']
    
    
    # Define scheduler
    scheduler = None
    if conf['training']['half_lr']:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5,
                                      patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf['main_args']['exp_dir']
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, 'conf.yml')
    with open(conf_path, 'w') as outfile:
        yaml.safe_dump(conf, outfile)
 
    # Define Loss function.
    #loss_func = PITLossWrapper(pairwise_neg_sisdr, mode='pairwise')
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
    system = System(model=model, loss_func=loss_func, optimizer=optimizer,
                    train_loader=train_loader, val_loader=val_loader,
                    scheduler=scheduler, config=conf)

    # Define callbacks
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints/')
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss',
                                 mode='min', save_top_k=100, verbose=1)
    early_stopping = False
    if conf['training']['early_stop']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                       verbose=1)

    # Don't ask GPU if they are not available.
    if not torch.cuda.is_available():
        print('No available GPU were found, set gpus to None')
        conf['main_args']['gpus'] = None
    trainer = pl.Trainer(max_epochs=conf['training']['epochs'],
                         checkpoint_callback=checkpoint,
                         early_stop_callback=early_stopping,
                         default_root_dir=exp_dir,
                         gpus=conf['main_args']['gpus'],
                         distributed_backend='dp',
                         train_percent_check=1.0,  # Useful for fast experiment
                         gradient_clip_val=5.)
    trainer.fit(system)

    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(checkpoint.best_k_models, f, indent=0)


if __name__ == '__main__':
    import yaml
    from pprint import pprint as print   
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open('local/conf.yml') as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    print(arg_dic)
    main(arg_dic)
