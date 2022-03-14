from pyexpat import features
import copy
import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from transformers import CLIPTextModel, CLIPVisionModel, CLIPModel

class CLIPClassifier(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.map_size = args.map_size    
        self.num_mapping_layers = args.num_mapping_layers 
        self.clip = CLIPModel.from_pretrained(args.clip_pretrained_model)   
        self.image_encoder = copy.deepcopy(self.clip.vision_model)
        self.text_encoder = copy.deepcopy(self.clip.text_model)
        # self.image_encoder = CLIPVisionModel.from_pretrained(args.clip_pretrained_model)
        # self.text_encoder = CLIPTextModel.from_pretrained(args.clip_pretrained_model)
            
        self.use_pretrained_map = args.use_pretrained_map
        if self.use_pretrained_map:
            final_map_dim = self.map_size#self.clip.projection_dim
            
            self.image_map = nn.Sequential(
                copy.deepcopy(self.clip.visual_projection),
                nn.ReLU(),
                nn.Linear(self.clip.projection_dim, self.map_size)
                )
            self.text_map = nn.Sequential(
                copy.deepcopy(self.clip.text_projection),
                nn.ReLU(),
                nn.Linear(self.clip.projection_dim, self.map_size)
                )
        else:
            image_map_layers = [nn.Linear(self.image_encoder.config.hidden_size, self.map_size), nn.Dropout(p=args.drop_probs[0])]#nn.BatchNorm1d(self.map_size), nn.Dropout(p=args.drop_probs[1])]
            text_map_layers = [nn.Linear(self.text_encoder.config.hidden_size, self.map_size), nn.Dropout(p=args.drop_probs[0])]#nn.BatchNorm1d(self.map_size), nn.Dropout(p=args.drop_probs[1])] 
            for i in range(1, self.num_mapping_layers):
                image_map_layers.append(nn.ReLU())
                image_map_layers.append(nn.Linear(self.map_size//(2**(i-1)), self.map_size//(2**i)))
                # image_map_layers.append(nn.BatchNorm1d(self.map_size//(2**i)))
                image_map_layers.append(nn.Dropout(p=args.drop_probs[0]))
                text_map_layers.append(nn.ReLU())
                text_map_layers.append(nn.Linear(self.map_size//(2**(i-1)), self.map_size//(2**i)))
                # text_map_layers.append(nn.BatchNorm1d(self.map_size//(2**i)))
                text_map_layers.append(nn.Dropout(p=args.drop_probs[0]))
            final_map_dim = self.map_size//(2**(self.num_mapping_layers-1))

            self.image_map = nn.Sequential(*image_map_layers)
            self.text_map = nn.Sequential(*text_map_layers)

        self.head = args.head
        if args.head == 'clip':
            self.logit_scale = nn.Parameter(torch.ones([], device=self.image_encoder.device) * np.log(1 / 0.07))
            self.remove_matches = args.remove_matches
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        elif args.head == 'concat':
            input_dim_for_out = 2 * final_map_dim
            hidden_dim_for_out = int(math.sqrt(final_map_dim))
            self.out = nn.Sequential(
                nn.Dropout(p=args.drop_probs[1]),
                nn.Linear(input_dim_for_out, hidden_dim_for_out),
                nn.ReLU(),
                nn.Dropout(p=args.drop_probs[2]),
                # nn.Linear(hidden_dim_for_out, hidden_dim_for_out),
                # nn.ReLU(),
                nn.Linear(hidden_dim_for_out, 1),
            )
            self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        elif args.head == 'cross':
            input_dim_for_out = final_map_dim ** 2
            hidden_dim_for_out = int(math.sqrt(final_map_dim))
            self.out = nn.Sequential(
                nn.Dropout(p=args.drop_probs[1]),
                nn.Linear(input_dim_for_out, final_map_dim),
                nn.ReLU(),
                nn.Dropout(p=args.drop_probs[2]),
                # nn.Linear(final_map_dim, hidden_dim_for_out),
                # nn.ReLU(),
                nn.Linear(final_map_dim, 1),
            )
            self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        else:
            raise ValueError()

        if args.freeze_image_encoder:
            for n, p in self.image_encoder.named_parameters():
                p.requires_grad_(False)

        if args.freeze_text_encoder:
            for n, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)

        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.acc = torchmetrics.Accuracy()
        self.auroc = torchmetrics.AUROC()

        del self.clip

    def forward(self, batch):
        image_features = self.image_encoder(pixel_values=batch['pixel_values'][0]).pooler_output
        image_features = self.image_map(image_features)
        text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        if self.head == 'clip':
            logit_scale = self.logit_scale.exp()
            logits = torch.mm(image_features, text_features.t()) * logit_scale
            preds = (torch.diagonal(logits) < 0).long()
        elif self.head == 'concat':
            features = torch.cat([image_features, text_features], dim=1)
            logits = self.out(features)
            preds = (torch.sigmoid(logits) > 0.5).long()
        elif self.head == 'cross':
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [16, d, d]
            features = features.reshape(features.shape[0], -1)  # [16, d*d]
            logits = self.out(features)
            preds = (torch.sigmoid(logits) > 0.5).long()

        return preds

    def common_step(self, batch, batch_idx):
        image_features = self.image_encoder(pixel_values=batch['pixel_values'][0]).pooler_output
        image_features = self.image_map(image_features)
        text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        if self.head == 'clip':
            logit_scale = self.logit_scale.exp()
            logits = torch.mm(image_features, text_features.t()) * logit_scale
            logits_diagonal = torch.diagonal(logits)
            preds_proxy = -logits_diagonal
            preds = (preds_proxy >= 0).long()
            accuracy = self.acc(preds, batch['labels'])
            auroc = self.auroc(preds_proxy, batch['labels'])

            # another way of classification by flipping text features
            logits_flipped = torch.mm(image_features, -text_features.t()) * logit_scale
            logits_diagonal_flipped = torch.diagonal(logits_flipped)
            preds_proxy_flipped = logits_diagonal_flipped - logits_diagonal
            preds_flipped = (preds_proxy_flipped >= 0).long()
            accuracy_flipped = self.acc(preds_flipped, batch['labels'])
            auroc_flipped = self.auroc(preds_proxy_flipped, batch['labels'])

            if self.remove_matches:
                batch_size = len(preds)

                # [N, N]: 1->no matching image; 0-> matching image
                idx_images_a = batch['idx_images'].unsqueeze(0).repeat(batch_size, 1)
                idx_images_b = batch['idx_images'].unsqueeze(1).repeat(1, batch_size)
                idx_images_non_match = 1 - (idx_images_a == idx_images_b).long().fill_diagonal_(0)
                image_match_score = 1 - (idx_images_non_match.sum()/(batch_size**2))

                # [N, N]: 1->no matching text; 0-> matching text
                idx_texts_a = batch['idx_texts'].unsqueeze(0).repeat(batch_size, 1)
                idx_texts_b = batch['idx_texts'].unsqueeze(1).repeat(1, batch_size)
                idx_texts_non_match = 1 - (idx_texts_a == idx_texts_b).long().fill_diagonal_(0)
                text_match_score = 1 - (idx_texts_non_match.sum()/(batch_size**2))

                self.log('extra/image_match_score', image_match_score)
                self.log('extra/text_match_score', text_match_score)

                logits_text = logits * idx_texts_non_match
                logits_image = logits.t() * idx_images_non_match
            else:
                logits_text = logits
                logits_image = logits.t()

            # to flip loss values if the item is hateful
            loss_flipper = torch.ones(len(batch['labels']), device=self.image_encoder.device)
            loss_flipper[batch['labels']==1] = -1
            labels = torch.arange(logits.shape[0], device=logits.device)

            loss_text = (self.cross_entropy_loss(logits_text, labels) * loss_flipper).mean()
            #loss_text = torch.clamp(loss_text, min=-5, max=-5)
            loss_image = (self.cross_entropy_loss(logits_image, labels) * loss_flipper).mean()
            #loss_image = torch.clamp(loss_image, min=-5, max=-5)
            loss = (loss_text + loss_image) / 2

        elif self.head == 'concat':
            features = torch.cat([image_features, text_features], dim=1)
            logits = self.out(features).squeeze(dim=1)
            preds_proxy = torch.sigmoid(logits)
            preds = (preds_proxy > 0.5).long()
            accuracy = self.acc(preds, batch['labels'])
            auroc = self.auroc(preds_proxy, batch['labels'])

            loss = self.cross_entropy_loss(logits, batch['labels'].float())
            accuracy_flipped, auroc_flipped = 0, 0

        elif self.head == 'cross':
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [16, d, d]
            features = features.reshape(features.shape[0], -1)  # [16, d*d]
            logits = self.out(features).squeeze(dim=1)
            preds_proxy = torch.sigmoid(logits)
            preds = (preds_proxy > 0.5).long()
            accuracy = self.acc(preds, batch['labels'])
            auroc = self.auroc(preds_proxy, batch['labels'])

            loss = self.cross_entropy_loss(logits, batch['labels'].float())
            accuracy_flipped, auroc_flipped = 0, 0

        return loss, accuracy, auroc, accuracy_flipped, auroc_flipped
        
    def training_step(self, batch, batch_idx):
        loss, accuracy, auroc, accuracy_flipped, auroc_flipped = self.common_step(batch, batch_idx)
        self.log('train/loss', loss)
        self.log('train/accuracy', accuracy)
        self.log('train/auroc', auroc)
        if self.head == 'clip':
            self.log('train/accuracy_flipped', accuracy_flipped)
            self.log('train/auroc_flipped', auroc_flipped)

        return loss

    def validation_step(self, batch, batch_idx):
        loss , accuracy, auroc, accuracy_flipped, auroc_flipped = self.common_step(batch, batch_idx)
        self.log('val/loss', loss)
        self.log('val/accuracy', accuracy)
        self.log('val/auroc', auroc)
        if self.head == 'clip':
            self.log('val/accuracy_flipped', accuracy_flipped)
            self.log('val/auroc_flipped', auroc_flipped)
         
        return loss

    def training_epoch_end(self, validation_step_outputs):
        self.acc.reset()
        self.auroc.reset()

    def validation_epoch_end(self, validation_step_outputs):
        self.acc.reset()
        self.auroc.reset()

    def configure_optimizers(self):
        param_dicts = [{"params": [p for n, p in self.named_parameters() if p.requires_grad]},]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer


def create_model(args):
    model = CLIPClassifier(args=args)

    return model