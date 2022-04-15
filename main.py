import argparse
import os

from datasets import CustomCollator, load_dataset
from engine import create_model

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader

def str2bool(v):
    """
    src: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arg_parser():

    parser = argparse.ArgumentParser(description='Traning and evaluation script for hateful meme classification')

    # dataset parameters
    parser.add_argument('--dataset', default='original', choices=['original', 'masked', 'inpainted'])
    parser.add_argument('--labels', default='original', choices=['original', 'fine_grained', 'fine_grained_gold'])
    parser.add_argument('--image_size', type=int, default=224)

    # model parameters
    parser.add_argument('--clip_pretrained_model', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--use_pretrained_map', default=False, type=str2bool)
    parser.add_argument('--num_mapping_layers', default=1, type=int)
    parser.add_argument('--map_dim', default=768, type=int)
    parser.add_argument('--head', default='clip', choices=['clip', 'concat', 'cross'])
    parser.add_argument('--num_pre_output_layers', default=1, type=int)
    parser.add_argument('--drop_probs', type=float, nargs=3, default=[0.1, 0.4, 0.2], help="Set drop probailities for map, head, pre_output")
    parser.add_argument('--image_encoder', type=str, default='clip', choices=['clip'])
    parser.add_argument('--text_encoder', type=str, default='clip', choices=['clip'])
    parser.add_argument('--freeze_image_encoder', type=str2bool, default=True)
    parser.add_argument('--freeze_text_encoder', type=str2bool, default=True)

    # training parameters
    parser.add_argument('--remove_matches', type=str2bool, default=False)
    parser.add_argument('--gpus', default='0', help='GPU ids concatenated with space')
    parser.add_argument('--strategy', default=None)
    parser.add_argument('--limit_train_batches', default=1.0)
    parser.add_argument('--limit_val_batches', default=1.0)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--max_epochs', type=int, default=-1)
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--val_check_interval', default=1.0)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_fine_grained_loss', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_clip_val', type=float, default=0.1)

    # other parameters
    parser.add_argument('--eval_split', default='test_seen', choices=['test_seen', 'val_seen'])

    return parser

def main(args):
    
    # load dataset
    dataset_train = load_dataset(args=args, split='train')
    dataset_val = load_dataset(args=args, split=args.eval_split)
    print("Number of training examples:", len(dataset_train))
    print("Number of validation examples:", len(dataset_val))
    print("Sample item:", dataset_train[0])
    print("Image size:", dataset_train[0]['image'].size)

    # load dataloader
    num_cpus = min(args.batch_size, 16) #(multiprocessing.cpu_count() // len(args.gpus))-1
    collator = CustomCollator(args, dataset_train.fine_grained_labels)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_cpus, collate_fn=collator)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator)
    
    # create model
    seed_everything(42, workers=True)
    model = create_model(args, dataset_train.fine_grained_labels)

    # sanity check
    # batch = next(iter(dataloader_train))
    # output = model(batch)
    # print(output)

    wandb_logger = WandbLogger(project="meme", config=args)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', filename=wandb_logger.experiment.name+'-{epoch:02d}',  monitor="train/accuracy", mode='max', verbose=True, save_weights_only=True)
    trainer = Trainer(gpus=args.gpus, max_epochs=args.max_epochs, max_steps=args.max_steps, gradient_clip_val=args.gradient_clip_val, 
        logger=wandb_logger, log_every_n_steps=args.log_every_n_steps, val_check_interval=args.val_check_interval,
        strategy=args.strategy, callbacks=[checkpoint_callback],
        limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
        deterministic=True)
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)


if __name__ == '__main__':

    parser = get_arg_parser()
    args = parser.parse_args()
    args.gpus = [int(id_) for id_ in args.gpus.split()]
    if args.strategy == 'ddp':
        args.strategy = DDPPlugin(find_unused_parameters=False)
    elif args.strategy == 'none':
        args.strategy = None

    main(args)