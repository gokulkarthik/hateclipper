from pyexpat import features
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from transformers import CLIPTextModel, CLIPVisionModel

class CLIPClassifier(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.map_size = args.map_size    
        self.num_mapping_layers = args.num_mapping_layers            
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        image_map_layers = [nn.Linear(self.image_encoder.config.hidden_size, self.map_size)]
        text_map_layers = [nn.Linear(self.text_encoder.config.hidden_size, self.map_size)] 
        for i in range(1, self.num_mapping_layers):
            image_map_layers.append(nn.Linear(self.map_size//(2**(i-1)), self.map_size//(2**i)))
            text_map_layers.append(nn.Linear(self.map_size//(2**(i-1)), self.map_size//(2**i)))

        self.image_map = nn.Sequential(*image_map_layers)
        self.text_map = nn.Sequential(*text_map_layers)
        self.head = args.head
        if args.head == 'clip':
            self.logit_scale = nn.Parameter(torch.ones([], device=self.image_encoder.device) * np.log(1 / 0.07))
            self.remove_matches = args.remove_matches
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        elif args.head == 'concat':
            self.out = nn.Linear(2*(self.map_size//(2**(self.num_mapping_layers-1))), 1)
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

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.train_auroc = torchmetrics.AUROC()
        self.val_auroc = torchmetrics.AUROC()

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
            preds = (torch.diagonal(logits) < 0).long()
            preds_proxy = -torch.diagonal(logits)
            accuracy = self.train_acc(preds, batch['labels'])
            auroc = self.train_auroc(preds_proxy, batch['labels'])

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
            loss_text = torch.clamp(loss_text, min=-5, max=-5)
            loss_image = (self.cross_entropy_loss(logits_image, labels) * loss_flipper).mean()
            loss_image = torch.clamp(loss_image, min=-5, max=-5)
            loss = (loss_text + loss_image) / 2

        elif self.head == 'concat':
            features = torch.cat([image_features, text_features], dim=1)
            logits = self.out(features).squeeze(dim=1)
            preds = (torch.sigmoid(logits) > 0.5).long()
            preds_proxy = torch.sigmoid(logits)
            accuracy = self.train_acc(preds, batch['labels'])
            auroc = self.train_auroc(preds_proxy, batch['labels'])

            loss = self.cross_entropy_loss(logits, batch['labels'].float())

        return loss, accuracy, auroc
        
    def training_step(self, batch, batch_idx):
        loss, accuracy, auroc = self.common_step(batch, batch_idx)
        self.log('train/loss', loss)
        self.log('train/accuracy', accuracy)
        self.log('train/auroc', auroc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss , accuracy, auroc = self.common_step(batch, batch_idx)
        self.log('val/loss', loss)
        self.log('val/accuracy', accuracy)
        self.log('val/auroc', auroc)
         
        return loss

    def training_epoch_end(self, validation_step_outputs):
        self.train_acc.reset()
        self.train_auroc.reset()

    def validation_epoch_end(self, validation_step_outputs):
        self.val_acc.reset()
        self.val_auroc.reset()

    def configure_optimizers(self):
        param_dicts = [{"params": [p for n, p in self.named_parameters() if p.requires_grad]},]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer


def create_model(args):
    model = CLIPClassifier(args=args)

    return model