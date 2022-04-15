from pyexpat import features
import copy
import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from transformers import CLIPModel

class CLIPClassifier(pl.LightningModule):

    def __init__(self, args, fine_grained_labels, compute_fine_grained_metrics):
        super().__init__()

        self.use_pretrained_map = args.use_pretrained_map
        self.num_mapping_layers = args.num_mapping_layers 
        self.map_dim = args.map_dim  
        self.head = args.head
        self.num_pre_output_layers = args.num_pre_output_layers 
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.weight_fine_grained_loss = args.weight_fine_grained_loss
        self.fine_grained_labels = fine_grained_labels
        self.compute_fine_grained_metrics = compute_fine_grained_metrics

        self.acc = torchmetrics.Accuracy()
        self.auroc = torchmetrics.AUROC()
        self.precision_score = torchmetrics.Precision()
        self.recall = torchmetrics.Recall()
        self.f1 = torchmetrics.F1Score()
        
        self.clip = CLIPModel.from_pretrained(args.clip_pretrained_model)   
        self.image_encoder = copy.deepcopy(self.clip.vision_model)
        self.text_encoder = copy.deepcopy(self.clip.text_model)
        
        if self.use_pretrained_map:
            self.image_map = nn.Sequential(
                copy.deepcopy(self.clip.visual_projection),
                nn.ReLU(),
                nn.Linear(self.clip.projection_dim, self.map_dim)
                )
            self.text_map = nn.Sequential(
                copy.deepcopy(self.clip.text_projection),
                nn.ReLU(),
                nn.Linear(self.clip.projection_dim, self.map_dim)
                )

        else:
            image_map_layers = [nn.Linear(self.image_encoder.config.hidden_size, self.map_dim), nn.Dropout(p=args.drop_probs[0])]
            text_map_layers = [nn.Linear(self.text_encoder.config.hidden_size, self.map_dim), nn.Dropout(p=args.drop_probs[0])]
            for _ in range(1, self.num_mapping_layers):
                image_map_layers.extend([nn.ReLU(), nn.Linear(self.map_dim, self.map_dim), nn.Dropout(p=args.drop_probs[0])])
                text_map_layers.extend([nn.ReLU(), nn.Linear(self.map_dim, self.map_dim), nn.Dropout(p=args.drop_probs[0])])

            self.image_map = nn.Sequential(*image_map_layers)
            self.text_map = nn.Sequential(*text_map_layers)
        
        if args.head == 'concat':
            pre_output_input_dim = self.map_dim*2
        elif args.head == 'cross':
            pre_output_input_dim = self.map_dim**2

        pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
        output_input_dim = pre_output_input_dim
        if self.num_pre_output_layers >= 1:
            pre_output_layers.extend([nn.Linear(pre_output_input_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
            output_input_dim = self.map_dim
        for _ in range(1, self.num_pre_output_layers):
            pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.output = nn.Linear(output_input_dim, 1)

        if self.fine_grained_labels:
            self.output_pc1 = nn.Linear(output_input_dim, 1)
            self.output_pc2 = nn.Linear(output_input_dim, 1)
            self.output_pc3 = nn.Linear(output_input_dim, 1)
            self.output_pc4 = nn.Linear(output_input_dim, 1)
            self.output_pc5 = nn.Linear(output_input_dim, 1)
            self.output_pc6 = nn.Linear(output_input_dim, 1)
            self.output_attack1 = nn.Linear(output_input_dim, 1)
            self.output_attack2 = nn.Linear(output_input_dim, 1)
            self.output_attack3 = nn.Linear(output_input_dim, 1)
            self.output_attack4 = nn.Linear(output_input_dim, 1)
            self.output_attack5 = nn.Linear(output_input_dim, 1)
            self.output_attack6 = nn.Linear(output_input_dim, 1)
            self.output_attack7 = nn.Linear(output_input_dim, 1)
            self.output_attack8 = nn.Linear(output_input_dim, 1)
            self.outputs_fine_grained = [self.output_pc1, self.output_pc2, self.output_pc3, self.output_pc4, self.output_pc5, self.output_pc6,
                self.output_attack1, self.output_attack2, self.output_attack3, self.output_attack4, self.output_attack5, self.output_attack6, self.output_attack7, self.output_attack8]

        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        if args.freeze_image_encoder:
            for _, p in self.image_encoder.named_parameters():
                p.requires_grad_(False)

        if args.freeze_text_encoder:
            for _, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)

        del self.clip

    def forward(self, batch):
        image_features = self.image_encoder(pixel_values=batch['pixel_values'][0]).pooler_output
        image_features = self.image_map(image_features)
        text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        if self.head == 'concat':
            features = torch.cat([image_features, text_features], dim=1)
        elif self.head == 'cross':
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [16, d, d]
            features = features.reshape(features.shape[0], -1)  # [16, d*d]

        features = self.pre_output(features)
        logits = self.output(features)
        preds = (torch.sigmoid(logits) >= 0.5).long()
    
        features = self.pre_output(features)
        logits = self.output(features)
        preds = (torch.sigmoid(logits) >= 0.5).long()

        return preds

    def common_step(self, batch, batch_idx, calling_function):
        image_features = self.image_encoder(pixel_values=batch['pixel_values'][0]).pooler_output
        image_features = self.image_map(image_features)
        text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        output = {}

        if self.head == 'concat':
            features = torch.cat([image_features, text_features], dim=1)
        elif self.head == 'cross':
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [16, d, d]
            features = features.reshape(features.shape[0], -1)  # [16, d*d]
        self.features = features

        features = self.pre_output(features)
        logits = self.output(features).squeeze(dim=1)
        preds_proxy = torch.sigmoid(logits)
        preds = (preds_proxy >= 0.5).long()

        output['loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
        output['accuracy'] = self.acc(preds, batch['labels'])
        output['auroc'] = self.auroc(preds_proxy, batch['labels'])

        if calling_function == 'training':
            for fine_grained_label, output_fine_grained in zip(self.fine_grained_labels, self.outputs_fine_grained):
                logits = output_fine_grained(features).squeeze(dim=1)
                preds_proxy = torch.sigmoid(logits)
                preds = (preds_proxy >= 0.5).long()
                output[f'{fine_grained_label}_loss'] = self.cross_entropy_loss(logits, batch[fine_grained_label].float())

        elif calling_function == 'validation' and self.compute_fine_grained_metrics:
            for fine_grained_label, output_fine_grained in zip(self.fine_grained_labels, self.outputs_fine_grained):
                logits = output_fine_grained(features).squeeze(dim=1)
                preds_proxy = torch.sigmoid(logits)
                preds = (preds_proxy >= 0.5).long()
                output[f'{fine_grained_label}_loss'] = self.cross_entropy_loss(logits, batch[fine_grained_label].float())
                output[f'{fine_grained_label}_accuracy'] = self.acc(preds, batch[fine_grained_label])
                output[f'{fine_grained_label}_auroc'] = self.auroc(preds_proxy, batch[fine_grained_label])
                output[f'{fine_grained_label}_precision'] = self.precision_score(preds, batch[fine_grained_label])
                output[f'{fine_grained_label}_recall'] = self.recall(preds, batch[fine_grained_label])
                output[f'{fine_grained_label}_f1'] = self.f1(preds, batch[fine_grained_label])

        return output
        
    def training_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx, calling_function='training')

        if self.fine_grained_labels:
            fine_grained_loss = 0
            for fine_grained_label in self.fine_grained_labels:
                fine_grained_loss += output[f'{fine_grained_label}_loss']
            fine_grained_loss /= len(self.fine_grained_labels)
        else:
            fine_grained_loss = 0.0
        total_loss = output['loss'] + self.weight_fine_grained_loss * fine_grained_loss

        self.log('train/loss', output['loss'])
        self.log('train/accuracy', output['accuracy'])
        self.log('train/auroc', output['auroc'])
        self.log('train/fine_grained_loss', fine_grained_loss)
        self.log('train/total_loss', total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx, calling_function='validation')
        
        if self.compute_fine_grained_metrics:
            fine_grained_loss = torch.mean([output[f'{fine_grained_label}_loss'] for fine_grained_label in self.fine_grained_labels])
        else:
            fine_grained_loss = 0.0
        total_loss = output['loss'] + self.weight_fine_grained_loss * fine_grained_loss

        self.log('val/loss', output['loss'])
        self.log('val/accuracy', output['accuracy'])
        self.log('val/auroc', output['auroc'])

        if self.compute_fine_grained_metrics:
            self.log('val/fine_grained_loss', fine_grained_loss)
            self.log('val/total_loss', total_loss)

            for fine_grained_label in self.fine_grained_labels:
                self.log(f'val-fine-grained/{fine_grained_label}_precision', output[f'{fine_grained_label}_precision'])
                self.log(f'val-fine-grained/{fine_grained_label}_recall', output[f'{fine_grained_label}_recall'])
                self.log(f'val-fine-grained/{fine_grained_label}_f1', output[f'{fine_grained_label}_f1'])
         
        return total_loss

    def training_epoch_end(self, validation_step_outputs):
        self.acc.reset()
        self.auroc.reset()
        self.precision_score.reset()
        self.recall.reset()
        self.f1.reset()

    def validation_epoch_end(self, validation_step_outputs):
        self.acc.reset()
        self.auroc.reset()
        self.precision_score.reset()
        self.recall.reset()
        self.f1.reset()

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]}
            ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer


def create_model(args, fine_grained_labels):
    if args.eval_split == 'test_seen':
        compute_fine_grained_metrics = False
    elif args.eval_split == 'val_seen':
        compute_fine_grained_metrics = True

    model = CLIPClassifier(args=args, fine_grained_labels=fine_grained_labels, compute_fine_grained_metrics = compute_fine_grained_metrics)

    return model