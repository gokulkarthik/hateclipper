import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer, CLIPProcessor


class HatefulMemesDataset(Dataset):
    def __init__(self, root_folder, image_folder, split='train', image_size=224):
        super(HatefulMemesDataset, self).__init__()
        self.root_folder = root_folder
        self.image_folder = image_folder
        self.image_size = image_size
        self.info_file = os.path.join(root_folder, 'info.csv')
        self.df = pd.read_csv(self.info_file)
        self.df = self.df[self.df['split']==split].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {}
        image_fn = row['img'].split('/')[1]
        item['image'] = Image.open(f"{self.image_folder}/{image_fn}").convert('RGB').resize((self.image_size, self.image_size))
        item['text'] = row['text']
        item['label'] = row['label']
        item['idx_meme'] = row['id']
        item['idx_image'] = row['pseudo_img_idx']
        item['idx_text'] = row['pseudo_text_idx']

        return item

class CustomCollator(object):

    def __init__(self, args):
        self.args = args
        self.image_processor = CLIPProcessor.from_pretrained(args.clip_pretrained_model)
        self.text_processor = CLIPTokenizer.from_pretrained(args.clip_pretrained_model)

    def __call__(self, batch):
        pixel_values = self.image_processor(images=[item['image'] for item in batch], return_tensors="pt")['pixel_values']
        text_output = self.text_processor([item['text'] for item in batch], padding=True, return_tensors="pt")
        labels = torch.LongTensor([item['label'] for item in batch])
        idx_memes = torch.LongTensor([item['idx_meme'] for item in batch])
        idx_images = torch.LongTensor([item['idx_image'] for item in batch])
        idx_texts = torch.LongTensor([item['idx_text'] for item in batch])

        if text_output['input_ids'].shape[1] > 77: # 77 is the max seq length of this model
            text_output['input_ids'] = text_output['input_ids'][:, :77]
            text_output['attention_mask'] = text_output['attention_mask'][:, :77]

        batch = {}
        batch['pixel_values'] = pixel_values,
        batch['input_ids'] = text_output['input_ids']
        batch['attention_mask'] = text_output['attention_mask']
        batch['labels'] = labels
        batch['idx_memes'] = idx_memes
        batch['idx_images'] = idx_images
        batch['idx_texts'] = idx_texts

        return batch



def load_dataset(args, split):

    if args.dataset == 'original':
        image_folder = 'data/hateful_memes/img'
    elif args.dataset == 'masked':
        image_folder = 'data/hateful_memes_masked/'
    elif args.dataset == 'inpainted':
        image_folder = 'data/hateful_memes_inpainted/'
    
    dataset = HatefulMemesDataset(root_folder='data/hateful_memes', image_folder=image_folder, split=split, image_size=args.image_size)

    return dataset
