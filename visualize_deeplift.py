import copy
import os
import shutil
from types import SimpleNamespace

import numpy as np
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


target_dir= 'explaining/deep_lift'
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.mkdir(target_dir)

num_clusters = 30
run_name = 'glowing-aardvark-22-epoch=08'#'treasured-surf-21-epoch=17'

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
dmodel = DerivedModel(pre_output, output)
dmodel.eval()

# Init DeepLift
batch_bg = next(iter(dataloader_bg))
features_bg = model.common_step(batch_bg, batch_idx=0, calling_function='visualisation-v2').detach()  # [batch_size, d*d]
e = shap.DeepExplainer(dmodel, features_bg)
print("DONE: Init DeepLift")

# run DeepLift
e_vectors = []
ids = []
cnt = 0
for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

    assert len(batch['pixel_values']) == 1

    if batch_idx < batch_size_bg: # skip examples used to init deeplift
        continue

    if cnt == 150: # 
        print("DONE: Deeplift run")
        break

    if batch['labels'][0].item() == 1: # for only hateful_memes
        cnt += 1
        features = model.common_step(batch, batch_idx=batch_idx, calling_function='visualisation-v2').detach()  # [batch_size, d*d]
        features = e.shap_values(features) # [batch_size, d*d]
        features = features.flatten() 

        e_vectors.append(features)
        ids.append(batch['idx_memes'][0].item())

        del features

kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(e_vectors)
clusters = np.array(kmeans.labels_)
ids = np.array(ids)
print(clusters)


for i in range(num_clusters):
    ids_cluster = ids[clusters == i]
    ids_cluster = [f"data/hateful_memes/img/{id_:05d}.png" for id_ in ids_cluster]
    print(f'Cluster {i}')
    for id_ in ids_cluster:
        plt.imshow(Image.open(id_))
        plt.savefig(f'{target_dir}/c{i}_{id_.rsplit("/", 1)[-1]}')
        plt.show()
        plt.close()

