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

        return loss


target_dir= 'explaining/dir_vectors'
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.mkdir(target_dir)

num_clusters = 15
run_name = 'treasured-surf-21-epoch=17'

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
args.map_dim = 128
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
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=collator)

model = CLIPClassifier.load_from_checkpoint(checkpoint_path, args=args, fine_grained_labels=fine_grained_labels, compute_fine_grained_metrics=False)
model.automatic_optimization = False
model.eval()
print("Mode:", model.training)

pre_output = copy.deepcopy(model.pre_output)
output = copy.deepcopy(model.output)
dmodel = DerivedModel(pre_output, output)
dmodel.eval()

# get top positives and negatives
features = torch.zeros((1, args.map_dim**2), requires_grad=True)
loss = dmodel.forward(features)
dmodel.zero_grad()
loss.backward()

features_grad = -features.grad.data
features_grad = features_grad.squeeze().reshape(args.map_dim, args.map_dim)
print(features_grad.shape)

q = torch.quantile(features_grad, 0.8)
top_pos_positions = (features_grad >= q).nonzero().tolist()
top_pos_positions = [tuple(l) for l in top_pos_positions]

q = torch.quantile(features_grad, 0.2)
top_neg_positions = (features_grad <= q).nonzero().tolist()
top_neg_positions = [tuple(l) for l in top_neg_positions]


pos_vectors = []
neg_vectors = []
dir_vectors = []
ids = []
for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

    assert len(batch['pixel_values']) == 1

    if batch_idx == 200:
        break

    if batch['labels'][0].item() == 1:
        features = model.common_step(batch, batch_idx, calling_function='visualisation-v2').detach()
        features = features.squeeze().reshape(args.map_dim, args.map_dim)
        mul = features #* features_grad
        label = {0: 'non-hateful', 1: 'hateful'}[batch['labels'][0].item()]

        q = torch.quantile(mul, 0.9)
        top_pos_positions_local = (mul >= q).nonzero().tolist()
        top_pos_positions_local = [tuple(l) for l in top_pos_positions_local]

        q = torch.quantile(mul, 0.1)
        top_neg_positions_local = (mul <= q).nonzero().tolist()
        top_neg_positions_local = [tuple(l) for l in top_neg_positions_local]

        matching_pos = np.array(list(set(top_pos_positions).intersection(set(top_pos_positions_local))))
        matching_neg = np.array(list(set(top_neg_positions).intersection(set(top_neg_positions_local))))

        
        pos_vector, neg_vector = np.zeros((args.map_dim, args.map_dim)), np.zeros((args.map_dim, args.map_dim))
        dir_vector = np.zeros((args.map_dim, args.map_dim))
        if len(matching_pos):
            pos_vector[matching_pos[:, 0], matching_pos[:, 1]] = 1
            dir_vector[matching_pos[:, 0], matching_pos[:, 1]] = 1
        if len(matching_neg):
            neg_vector[matching_neg[:, 0], matching_neg[:, 1]] = 1
            dir_vector[matching_neg[:, 0], matching_neg[:, 1]] = -1
        pos_vector = pos_vector.flatten()
        neg_vector = neg_vector.flatten()
        dir_vector = dir_vector.flatten()

        #pos_vectors.append(pos_vector)
        #neg_vectors.append(neg_vector)
        dir_vectors.append(dir_vector)
        ids.append(batch['idx_memes'][0].item())

        del features
        del mul
        del top_pos_positions_local
        del top_neg_positions_local
        del matching_pos
        del matching_neg
        del pos_vector
        del neg_vector
        del dir_vector


kmeans_dir = KMeans(n_clusters=num_clusters, random_state=0).fit(dir_vectors)
# kmeans_pos = KMeans(n_clusters=num_clusters, random_state=0).fit(pos_vectors)
# kmeans_neg = KMeans(n_clusters=num_clusters, random_state=0).fit(neg_vectors)

clusters_dir = np.array(kmeans_dir.labels_)
# clusters_pos = np.array(kmeans_pos.labels_)
# clusters_neg = np.array(kmeans_neg.labels_)
ids = np.array(ids)

print(clusters_dir)


for i in range(num_clusters):
    ids_cluster = ids[clusters_dir == i]
    ids_cluster = [f"data/hateful_memes/img/{id_:05d}.png" for id_ in ids_cluster]
    print(f'Cluster {i}')
    for id_ in ids_cluster:
        plt.imshow(Image.open(id_))
        plt.savefig(f'{target_dir}/c{i}_{id_.rsplit("/", 1)[-1]}')
        plt.show()
        plt.close()
