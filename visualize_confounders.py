import copy
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from dotmap import DotMap
from PIL import Image
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from engine import CLIPClassifier
from datasets import CustomCollator, load_dataset


class DerivedModelImage(nn.Module):
    def __init__(self, pre_output, output):
        super(DerivedModelImage, self).__init__()
        self.pre_output = pre_output
        self.output = output
        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, image_features):
        text_features = torch.ones_like(image_features)
        features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [batch_size, d, d]
        features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
        features_pre_output = self.pre_output(features)
        logits = self.output(features_pre_output).squeeze(dim=1)

        preds_proxy = torch.sigmoid(logits)
        preds = (preds_proxy >= 0.5).long()

        loss = self.cross_entropy_loss(logits, torch.Tensor([1.0]*len(logits)))

        return preds_proxy.unsqueeze(dim=1)


class DerivedModelText(nn.Module):
    def __init__(self, pre_output, output):
        super(DerivedModelText, self).__init__()
        self.pre_output = pre_output
        self.output = output
        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, text_features):
        image_features = torch.ones_like(text_features)
        features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [batch_size, d, d]
        features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
        features_pre_output = self.pre_output(features)
        logits = self.output(features_pre_output).squeeze(dim=1)

        preds_proxy = torch.sigmoid(logits)
        preds = (preds_proxy >= 0.5).long()

        loss = self.cross_entropy_loss(logits, torch.Tensor([1.0]*len(logits)))

        return preds_proxy.unsqueeze(dim=1)

class DerivedModel(nn.Module):
    def __init__(self, pre_output, output):
        super(DerivedModel, self).__init__()
        self.pre_output = pre_output
        self.output = output
        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, features):
        features_pre_output = self.pre_output(features)
        logits = self.output(features_pre_output).squeeze(dim=1)

        preds_proxy = torch.sigmoid(logits)
        preds = (preds_proxy >= 0.5).long()

        loss = self.cross_entropy_loss(logits, torch.Tensor([1.0]*len(logits)))

        return preds_proxy.unsqueeze(dim=1)


target_dir= 'explaining/confounders'
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.mkdir(target_dir)

### Setup model and deeplift
run_name = 'glowing-aardvark-22-epoch=08'

#  dargs
args = SimpleNamespace()
args.dataset = 'original'
args.labels = 'original'
args.image_size = 224
args.caption_mode = "none"
split = 'train'

# cargs
args.clip_pretrained_model = "openai/clip-vit-large-patch14"

# args
checkpoint_path = f'checkpoints/{run_name}.ckpt'
args.use_pretrained_map = False
args.num_mapping_layers = 1
args.map_dim = 32
args.fusion = 'cross'
args.num_pre_output_layers = 1
args.lr = 0.0001
args.weight_decay = 0.0001
args.weight_fine_grained_loss = 0
args.weight_image_loss = 0
args.weight_text_loss = 0
args.weight_fine_grained_loss = 0
args.weight_super_loss = 0
args.local_pretrained_weights = 'none'
args.compute_fine_grained_metrics = False
args.text_encoder = 'clip'
args.image_encoder = 'clip'
args.freeze_image_encoder = True
args.freeze_text_encoder = True
args.drop_probs = [0.2, 0.4, 0.1]
args.clip_pretrained_model = "openai/clip-vit-large-patch14"
args.caption_mode = "none"
fine_grained_labels = [] #['disability_pc', 'nationality_pc', 'pc_empty_pc', 'race_pc', 'religion_pc', 'sex_pc', 'attack_empty_attack', 'contempt_attack', 'dehumanizing_attack', 'exclusion_attack', 'inciting_violence_attack', 'inferiority_attack', 'mocking_attack', 'slurs_attack']

dataset = load_dataset(args=args, split=split)
print("Number of examples:", len(dataset))
print("Sample item:", dataset[0])

collator = CustomCollator(args, dataset.fine_grained_labels)
batch_size_bg = 100
dataloader_bg = DataLoader(dataset, batch_size=batch_size_bg, shuffle=False, num_workers=1, collate_fn=collator)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collator)

model = CLIPClassifier.load_from_checkpoint(checkpoint_path, args=args, fine_grained_labels=fine_grained_labels, compute_fine_grained_metrics=False)
model.automatic_optimization = False
model.eval()
print("Mode:", model.training)

pre_output = copy.deepcopy(model.pre_output)
output = copy.deepcopy(model.output)
dmodel_image = DerivedModelImage(pre_output, output)
dmodel_image.eval()
dmodel_text = DerivedModelText(pre_output, output)
dmodel_text.eval()
dmodel = DerivedModel(pre_output, output)
dmodel.eval()

# Init DeepLift
batch_bg = next(iter(dataloader_bg))
image_features, text_features = model.common_step(batch_bg, batch_idx=0, calling_function='visualisation-v1')  # [batch_size, d], [batch_size, d]
image_features, text_features = image_features.detach(), text_features.detach()
features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [batch_size, d, d]
features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
e_image = shap.DeepExplainer(dmodel_image, image_features)
e_text = shap.DeepExplainer(dmodel_text, text_features)
e = shap.DeepExplainer(dmodel, features)
print("DONE: Init DeepLift")

### visualise confounders
df = pd.read_csv("data/hateful_memes/hateful_memes_expanded.csv")
print(len(df))
df = df[df['split']=='train']
print(len(df))
df = df.sort_index(ascending=False)
print(df['pseudo_text_idx'].nunique())
print(df['pseudo_img_idx'].nunique())

## image confounders
cnt = 0
for group_id, df_group in tqdm(df.groupby('pseudo_text_idx'), total=df['pseudo_text_idx'].nunique()):
    for row_id, row in df_group.iterrows():
        item = {}
        item['id'] = row['id']
        item['image'] = Image.open(f"data/hateful_memes/{row['img']}").convert('RGB').resize((args.image_size, args.image_size))
        item['text'] = row['text']
        item['label'] = row['label']

        pixel_values = collator.image_processor(images=[item['image']], return_tensors="pt")['pixel_values']
        text_output = collator.text_processor([item['text']], padding=True, return_tensors="pt", truncation=True)

        batch = {}
        batch['pixel_values'] = pixel_values,
        batch['input_ids'] = text_output['input_ids']
        batch['attention_mask'] = text_output['attention_mask']
        batch['labels'] = torch.LongTensor([item['label']])

        image_features, text_features = model.common_step(batch, batch_idx=cnt, calling_function='visualisation-v1') # [1, d], [1, d]
        image_features, text_features = image_features.detach(), text_features.detach()
        features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [1, d, d]
        features = features.reshape(features.shape[0], -1)  # [1, d*d]

        shap_scores_image = e_image.shap_values(image_features) # [1, d]
        shap_scores_text = e_image.shap_values(text_features) # [1, d]
        shap_scores = e.shap_values(features) # [1, d*d]
        shap_scores = shap_scores.flatten().reshape(args.map_dim, args.map_dim) # [d, d]

        sns.heatmap(shap_scores, center=0)
        plt.savefig(f'{target_dir}/base_t{group_id}_l{item["label"]}_{row["img"].rsplit("/", 1)[-1]}')
        plt.show()
        plt.close()

        shap_scores_image_text =  np.vstack([shap_scores_image, shap_scores_text]) 
        sns.heatmap(shap_scores_image_text, center=0)
        plt.savefig(f'{target_dir}/t{group_id}_l{item["label"]}_{row["img"].rsplit("/", 1)[-1]}')
        plt.show()
        plt.close()

        # shap_scores = np.vstack([shap_scores_text, np.zeros_like(shap_scores_text), shap_scores]) # [d+2, d]
        # shap_scores_image = np.vstack([np.zeros((2, 1)), shap_scores_image.T]) # [d+2 , 1]
        # shap_scores = np.hstack([shap_scores_image, np.zeros_like(shap_scores_image), shap_scores]) # [d+2, d+2]

        # sns.heatmap(shap_scores, center=0)
        # plt.savefig(f'{target_dir}/t{group_id}_l{item["label"]}_{row["img"].rsplit("/", 1)[-1]}')
        # plt.show()
        # plt.close()

        cnt += 1
        
    if cnt > 150:
        print("DONE: Image confounders")
        break


## text confounders
cnt = 0
for group_id, df_group in tqdm(df.groupby('pseudo_img_idx'), total=df['pseudo_img_idx'].nunique()):
    for row_id, row in df_group.iterrows():
        item = {}
        item['id'] = row['id']
        item['image'] = Image.open(f"data/hateful_memes/{row['img']}").convert('RGB').resize((args.image_size, args.image_size))
        item['text'] = row['text']
        item['label'] = row['label']

        pixel_values = collator.image_processor(images=[item['image']], return_tensors="pt")['pixel_values']
        text_output = collator.text_processor([item['text']], padding=True, return_tensors="pt", truncation=True)

        batch = {}
        batch['pixel_values'] = pixel_values,
        batch['input_ids'] = text_output['input_ids']
        batch['attention_mask'] = text_output['attention_mask']
        batch['labels'] = torch.LongTensor([item['label']])

        image_features, text_features = model.common_step(batch, batch_idx=cnt, calling_function='visualisation-v1') # [1, d], [1, d]
        image_features, text_features = image_features.detach(), text_features.detach()
        features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [1, d, d]
        features = features.reshape(features.shape[0], -1)  # [1, d*d]

        shap_scores_image = e_image.shap_values(image_features) # [1, d]
        shap_scores_text = e_image.shap_values(text_features) # [1, d]
        shap_scores = e.shap_values(features) # [1, d*d]
        shap_scores = shap_scores.flatten().reshape(args.map_dim, args.map_dim) # [d, d]

        sns.heatmap(shap_scores, center=0)
        plt.savefig(f'{target_dir}/base_i{group_id}_l{item["label"]}_{row["img"].rsplit("/", 1)[-1]}')
        plt.show()
        plt.close()

        shap_scores_image_text =  np.vstack([shap_scores_image, shap_scores_text]) 
        sns.heatmap(shap_scores_image_text, center=0)
        plt.savefig(f'{target_dir}/t{group_id}_l{item["label"]}_{row["img"].rsplit("/", 1)[-1]}')
        plt.show()
        plt.close()

        # shap_scores = np.vstack([shap_scores_text, np.zeros_like(shap_scores_text), shap_scores]) # [d+2, d]
        # shap_scores_image = np.vstack([np.zeros((2, 1)), shap_scores_image.T]) # [d+2 , 1]
        # shap_scores = np.hstack([shap_scores_image, np.zeros_like(shap_scores_image), shap_scores]) # [d+2, d+2]

        # sns.heatmap(shap_scores, center=0)
        # plt.savefig(f'{target_dir}/t{group_id}_l{item["label"]}_{row["img"].rsplit("/", 1)[-1]}')
        # plt.show()
        # plt.close()

        cnt += 1
        
    if cnt > 150:
        print("DONE: Text confounders")
        break
        
