import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer, CLIPProcessor


class HatefulMemesDataset(Dataset):
    def __init__(self, root_folder, image_folder, split='train', labels='original', image_size=224):
        super(HatefulMemesDataset, self).__init__()
        self.root_folder = root_folder
        self.image_folder = image_folder
        self.split = split
        self.labels = labels
        self.image_size = image_size
        self.info_file = os.path.join(root_folder, 'info_fine_grained.csv')
        self.df = pd.read_csv(self.info_file)
        self.df = self.df[self.df['split']==self.split].reset_index(drop=True)
        float_cols = self.df.select_dtypes(float).columns
        self.df[float_cols] = self.df[float_cols].fillna(-1).astype('Int64')

        if self.labels == 'fine_grained':
            self.pc_columns = [col for col in self.df.columns if col.endswith('_pc') and not col.endswith('_gold_pc')]
            self.pc_columns.remove('gold_pc')
            self.attack_columns = [col for col in self.df.columns if col.endswith('_attack') and not col.endswith('_gold_attack')]
            self.attack_columns.remove('gold_attack')
            self.fine_grained_labels = self.pc_columns + self.attack_columns
        elif self.labels == 'fine_grained_gold':
            self.pc_columns = [col for col in self.df.columns if col.endswith('_gold_pc')]
            self.attack_columns = [col for col in self.df.columns if col.endswith('_gold_attack')]
            self.fine_grained_labels = self.pc_columns + self.attack_columns
        else:
            self.fine_grained_labels = []
        
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

        if self.labels.startswith('fine_grained'):
            for label in self.fine_grained_labels:
                item[label] = row[label]

        return item

class CustomCollator(object):

    def __init__(self, args, fine_grained_labels):
        self.args = args
        self.fine_grained_labels = fine_grained_labels
        self.image_processor = CLIPProcessor.from_pretrained(args.clip_pretrained_model)
        self.text_processor = CLIPTokenizer.from_pretrained(args.clip_pretrained_model)

    def __call__(self, batch):
        pixel_values = self.image_processor(images=[item['image'] for item in batch], return_tensors="pt")['pixel_values']
        text_output = self.text_processor([item['text'] for item in batch], padding=True, return_tensors="pt", truncation=True)
        labels = torch.LongTensor([item['label'] for item in batch])
        idx_memes = torch.LongTensor([item['idx_meme'] for item in batch])
        idx_images = torch.LongTensor([item['idx_image'] for item in batch])
        idx_texts = torch.LongTensor([item['idx_text'] for item in batch])

        batch_new = {}
        batch_new['pixel_values'] = pixel_values,
        batch_new['input_ids'] = text_output['input_ids']
        batch_new['attention_mask'] = text_output['attention_mask']
        batch_new['labels'] = labels
        batch_new['idx_memes'] = idx_memes
        batch_new['idx_images'] = idx_images
        batch_new['idx_texts'] = idx_texts

        if self.args.labels.startswith('fine_grained'):
            for label in self.fine_grained_labels:
                batch_new[label] = torch.LongTensor([item[label] for item in batch])

        return batch_new



def load_dataset(args, split):

    if args.dataset == 'original':
        image_folder = 'data/hateful_memes/img'
    elif args.dataset == 'masked':
        image_folder = 'data/hateful_memes_masked/'
    elif args.dataset == 'inpainted':
        image_folder = 'data/hateful_memes_inpainted/'
    
    dataset = HatefulMemesDataset(root_folder='data/hateful_memes', image_folder=image_folder, split=split, 
        labels=args.labels, image_size=args.image_size)

    return dataset
