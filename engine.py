from pyexpat import features
import copy
import math
from sys import prefix
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from transformers import CLIPModel, AutoConfig, AutoModel

class CLIPClassifier(pl.LightningModule):

    def __init__(self, args, fine_grained_labels, compute_fine_grained_metrics):
        super().__init__()

        self.caption_mode = args.caption_mode
        self.use_pretrained_map = args.use_pretrained_map
        self.num_mapping_layers = args.num_mapping_layers 
        self.map_dim = args.map_dim  
        self.fusion = args.fusion
        self.num_pre_output_layers = args.num_pre_output_layers 
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.weight_image_loss = args.weight_image_loss
        self.weight_text_loss = args.weight_text_loss
        self.weight_fine_grained_loss = args.weight_fine_grained_loss
        self.weight_super_loss = args.weight_super_loss
        self.fine_grained_labels = fine_grained_labels
        self.compute_fine_grained_metrics = compute_fine_grained_metrics

        # for tamil dataset
        self.text_encoder_name = args.text_encoder
        self.dataset = args.dataset

        self.acc = torchmetrics.Accuracy()
        if self.dataset == 'prop':
            self.auroc = torchmetrics.AUROC(num_classes=22)
            self.precision_score = torchmetrics.Precision(mdmc_average='global')
            self.recall = torchmetrics.Recall(mdmc_average='global')
            self.f1 = torchmetrics.F1Score(mdmc_average='global')
        else:
            self.auroc = torchmetrics.AUROC()
            self.precision_score = torchmetrics.Precision()
            self.recall = torchmetrics.Recall()
            self.f1 = torchmetrics.F1Score()
        
        self.clip = CLIPModel.from_pretrained(args.clip_pretrained_model)
        if args.local_pretrained_weights != 'none':
            state_dict = torch.load(args.local_pretrained_weights)['state_dict']
            state_dict = {k[5:]:v for k,v in state_dict.items() if k.startswith('clip')}
            self.clip.load_state_dict(state_dict)
        if args.image_encoder == 'clip': 
            self.image_encoder = copy.deepcopy(self.clip.vision_model)
        else:
            raise ValueError()
        if args.text_encoder == 'clip':
            self.text_encoder = copy.deepcopy(self.clip.text_model)
        elif args.text_encoder:
            config = AutoConfig.from_pretrained(args.text_encoder, output_hidden_states=True)
            self.text_encoder = AutoModel.from_pretrained(args.text_encoder, config=config)
        else:
            raise ValueError()
        
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
        
        if args.fusion in ['align', 'align_shuffle']:
            pre_output_input_dim = self.map_dim
        elif args.fusion == 'concat':
            pre_output_input_dim = self.map_dim*2
        elif args.fusion.startswith('cross'):
            pre_output_input_dim = self.map_dim**2
        elif args.fusion == 'align_concat':
            pre_output_input_dim = self.map_dim*3
        elif args.fusion == 'attention_m':
            self.gen_query = nn.Linear(self.map_dim, self.map_dim//4) 
            self.gen_key = nn.Linear(self.map_dim, self.map_dim//4) 
            self.soft = nn.Softmax(dim=1)
            pre_output_input_dim = self.map_dim*2

        pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
        output_input_dim = pre_output_input_dim
        if self.num_pre_output_layers >= 1: # first pre-output layer
            pre_output_layers.extend([nn.Linear(pre_output_input_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
            output_input_dim = self.map_dim
        for _ in range(1, self.num_pre_output_layers): # next pre-output layers
            pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])

        self.pre_output = nn.Sequential(*pre_output_layers)
        if self.dataset in ['original', 'masked', 'inpainted', 'tamil']:
            self.output = nn.Linear(output_input_dim, 1)
        elif self.dataset == 'prop':
            self.output = nn.Linear(output_input_dim, 22)

        if self.weight_image_loss > 0:
            pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
            for _ in range(self.num_pre_output_layers): # next pre-output layers
                pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
            self.pre_output_image = nn.Sequential(*pre_output_layers)
            if self.dataset in ['original', 'masked', 'inpainted', 'tamil']:
                self.output_image = nn.Linear(output_input_dim, 1)
            elif self.dataset == 'prop':
                self.output_image = nn.Linear(output_input_dim, 22)
        if self.weight_text_loss > 0:
            pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
            for _ in range(self.num_pre_output_layers): # next pre-output layers
                pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
            self.pre_output_text = nn.Sequential(*pre_output_layers)
            if self.dataset in ['original', 'masked', 'inpainted', 'tamil']:
                self.output_text = nn.Linear(output_input_dim, 1)
            elif self.dataset == 'prop':
                self.output_text = nn.Linear(output_input_dim, 22)

        if self.fine_grained_labels:
            if self.dataset in ['original', 'masked', 'inpainted']:
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
                self.output_super = nn.Linear(15, 1)

        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        if args.freeze_image_encoder:
            for _, p in self.image_encoder.named_parameters():
                p.requires_grad_(False)

        if args.freeze_text_encoder:
            for _, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)

        del self.clip
        if self.caption_mode == 'replace_image':
            del self.image_encoder, self.image_map

    def forward(self, batch):
        if self.caption_mode != "replace_image":
            image_features = self.image_encoder(pixel_values=batch['pixel_values'][0]).pooler_output
            image_features = self.image_map(image_features)
        elif self.text_encoder_name == 'clip':
            image_features = self.text_encoder(input_ids=batch['input_ids_caption'], attention_mask=batch['attention_mask_caption']).pooler_output
            image_features = self.text_map(image_features)
        else:
            text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])["hidden_states"][-1][:, 0, :]
            image_features = self.text_map(image_features)

        if self.text_encoder_name == 'clip':
            text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output
        else:
            text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])["hidden_states"][-1][:, 0, :]
        text_features = self.text_map(text_features)

        if self.caption_mode.startswith('parallel'):
            caption_features = self.text_encoder(input_ids=batch['input_ids_caption'], attention_mask=batch['attention_mask_caption']).pooler_output
            caption_features = self.text_map(caption_features)

        image_features = F.normalize(image_features, p=2, dim=1) # [batch_size, d]
        text_features = F.normalize(text_features, p=2, dim=1) # [batch_size, d]
        if self.caption_mode.startswith('parallel'):
            caption_features = F.normalize(caption_features, p=2, dim=1)

        if self.fusion in ['align', 'align_shuffle']:
            features = torch.mul(image_features, text_features)  # [batch_size, d]
        elif self.fusion == 'concat':
            features = torch.cat([image_features, text_features], dim=1)  # [batch_size, 2*d]
        elif self.fusion.startswith('cross'):
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [batch_size, d, d]
            if self.fusion == 'cross_nd':
                mask = torch.eye(self.map_dim).repeat(features.shape[0], 1, 1).bool()
                features[mask] = torch.zeros(features.shape[0]*self.map_dim, device=features.device)
                del mask
            features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
        elif self.fusion == 'align_concat':
            features = torch.cat([torch.mul(image_features, text_features), image_features, text_features], dim=1)  # [batch_size, 3*d]
        elif self.fusion == 'attention_m':
            q1 = F.relu(self.gen_query(image_features))
            k1 = F.relu(self.gen_key(image_features))
            q2 = F.relu(self.gen_query(text_features))
            k2 = F.relu(self.gen_key(text_features))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
            prob_1 = wt_i1_i2[:,0]
            prob_2 = wt_i1_i2[:,1]
            wtd_i1 = image_features * prob_1[:, None]
            wtd_i2 = text_features * prob_2[:, None]
            features = torch.cat((wtd_i1,wtd_i2), 1) # [batch_size, 2*d]
        else:
                raise ValueError()

        if self.caption_mode.startswith('parallel'):
            if self.fusion in ['align', 'align_shuffle']:
                features_parallel = torch.mul(caption_features, text_features)  # [batch_size, d]
            elif self.fusion == 'concat':
                features_parallel = torch.cat([caption_features, text_features], dim=1)  # [batch_size, 2*d]
            elif self.fusion.startswith('cross'):
                features_parallel = torch.bmm(caption_features.unsqueeze(2), text_features.unsqueeze(1)) # [batch_size, d, d]
                if self.fusion == 'cross_nd':
                    mask = torch.eye(self.map_dim).repeat(features.shape[0], 1, 1).bool()
                    features[mask] = torch.zeros(features.shape[0]*self.map_dim, device=features.device)
                    del mask
                features_parallel = features_parallel.reshape(features_parallel.shape[0], -1)  # [batch_size, d*d]
            elif self.fusion == 'align_concat':
                features = torch.cat([torch.mul(image_features, text_features), image_features, text_features], dim=1)  # [batch_size, 3*d]
            elif self.fusion == 'attention_m':
                q1 = F.relu(self.gen_query(image_features))
                k1 = F.relu(self.gen_key(image_features))
                q2 = F.relu(self.gen_query(text_features))
                k2 = F.relu(self.gen_key(text_features))
                score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
                score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
                wt_score1_score2_mat = torch.cat((score1, score2), 1)
                wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
                prob_1 = wt_i1_i2[:,0]
                prob_2 = wt_i1_i2[:,1]
                wtd_i1 = image_features * prob_1[:, None]
                wtd_i2 = text_features * prob_2[:, None]
                features = torch.cat((wtd_i1,wtd_i2), 1) # [batch_size, 2*d]
            else:
                raise ValueError()

            if self.caption_mode == 'parallel_max':
                features = torch.maximum(features, features_parallel)
            elif self.caption_mode == 'parallel_mean':
                features = (features + features_parallel) / 2.0
            elif self.caption_mode == 'parallel_align':
                features = torch.mul(features, features_parallel)
            else:
                raise ValueError()

        features = self.pre_output(features)
        logits = self.output(features)
        preds = (torch.sigmoid(logits) >= 0.5).long()

        return preds

    def common_step(self, batch, batch_idx, calling_function='validation'):
        if self.caption_mode != "replace_image":
            image_features = self.image_encoder(pixel_values=batch['pixel_values'][0]).pooler_output
            image_features = self.image_map(image_features)
        else:
            image_features = self.text_encoder(input_ids=batch['input_ids_caption'], attention_mask=batch['attention_mask_caption']).pooler_output
            image_features = self.text_map(image_features)
        text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)
        if self.caption_mode.startswith('parallel'):
            caption_features = self.text_encoder(input_ids=batch['input_ids_caption'], attention_mask=batch['attention_mask_caption']).pooler_output
            caption_features = self.text_map(caption_features)


        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        if self.caption_mode.startswith('parallel'):
            caption_features = F.normalize(caption_features, p=2, dim=1)
        output = {}

        if self.weight_image_loss > 0:
            features_pre_output = self.pre_output_image(image_features)
            logits = self.output_image(features_pre_output).squeeze(dim=1) # [batch_size, 1]
            preds_proxy = torch.sigmoid(logits)
            preds = (preds_proxy >= 0.5).long()

            output['image_loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
            output['image_accuracy'] = self.acc(preds, batch['labels'])
            output['image_auroc'] = self.auroc(preds_proxy, batch['labels'])

        if self.weight_text_loss > 0:
            features_pre_output = self.pre_output_text(text_features)
            logits = self.output_text(features_pre_output).squeeze(dim=1) # [batch_size, 1]
            preds_proxy = torch.sigmoid(logits)
            preds = (preds_proxy >= 0.5).long()

            output['text_loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
            output['text_accuracy'] = self.acc(preds, batch['labels'])
            output['text_auroc'] = self.auroc(preds_proxy, batch['labels'])

        if self.fusion in ['align', 'align_shuffle']:
            features = torch.mul(image_features, text_features)
        elif self.fusion == 'concat':
            features = torch.cat([image_features, text_features], dim=1)
        elif self.fusion.startswith('cross'):
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [16, d, d]
            if self.fusion == 'cross_nd':
                mask = torch.eye(self.map_dim).repeat(features.shape[0], 1, 1).bool()
                features[mask] = torch.zeros(features.shape[0]*self.map_dim, device=features.device)
                del mask
            features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
        elif self.fusion == 'align_concat':
                features = torch.cat([torch.mul(image_features, text_features), image_features, text_features], dim=1)  # [batch_size, 3*d]
        elif self.fusion == 'attention_m':
            q1 = F.relu(self.gen_query(image_features))
            k1 = F.relu(self.gen_key(image_features))
            q2 = F.relu(self.gen_query(text_features))
            k2 = F.relu(self.gen_key(text_features))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
            prob_1 = wt_i1_i2[:,0]
            prob_2 = wt_i1_i2[:,1]
            wtd_i1 = image_features * prob_1[:, None]
            wtd_i2 = text_features * prob_2[:, None]
            features = torch.cat((wtd_i1,wtd_i2), 1) # [batch_size, 2*d]
        else:
            raise ValueError()

        if self.caption_mode.startswith('parallel'):
            if self.fusion in ['align', 'align_shuffle']:
                features_parallel = torch.mul(caption_features, text_features)  # [batch_size, d]
            elif self.fusion == 'concat':
                features_parallel = torch.cat([caption_features, text_features], dim=1)  # [batch_size, 2*d]
            elif self.fusion.startswith('cross'):
                features_parallel = torch.bmm(caption_features.unsqueeze(2), text_features.unsqueeze(1)) # [batch_size, d, d]
                if self.fusion == 'cross_nd':
                    mask = torch.eye(self.map_dim).repeat(features.shape[0], 1, 1).bool()
                    features[mask] = torch.zeros(features.shape[0]*self.map_dim, device=features.device)
                    del mask
                features_parallel = features_parallel.reshape(features_parallel.shape[0], -1)  # [batch_size, d*d]
            elif self.fusion == 'align_concat':
                features = torch.cat([torch.mul(image_features, text_features), image_features, text_features], dim=1)  # [batch_size, 3*d]
            elif self.fusion == 'attention_m':
                q1 = F.relu(self.gen_query(image_features))
                k1 = F.relu(self.gen_key(image_features))
                q2 = F.relu(self.gen_query(text_features))
                k2 = F.relu(self.gen_key(text_features))
                score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
                score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
                wt_score1_score2_mat = torch.cat((score1, score2), 1)
                wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
                prob_1 = wt_i1_i2[:,0]
                prob_2 = wt_i1_i2[:,1]
                wtd_i1 = image_features * prob_1[:, None]
                wtd_i2 = text_features * prob_2[:, None]
                features = torch.cat((wtd_i1,wtd_i2), 1) # [batch_size, 2*d]
            else:
                raise ValueError()

            if self.caption_mode == 'parallel_max':
                features = torch.maximum(features, features_parallel)
            elif self.caption_mode == 'parallel_mean':
                features = (features + features_parallel) / 2.0
            elif self.caption_mode == 'parallel_align':
                features = torch.mul(features, features_parallel)
            else:
                raise ValueError()

        features_pre_output = self.pre_output(features)
        logits = self.output(features_pre_output).squeeze(dim=1) # [batch_size, 1(or)n]
        if self.fine_grained_labels and self.dataset in ['original', 'masked', 'inpainted']:
            logits_for_super = [torch.relu(logits)]
        preds_proxy = torch.sigmoid(logits)
        preds = (preds_proxy >= 0.5).long()

        output['loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
        output['accuracy'] = self.acc(preds, batch['labels'])
        output['auroc'] = self.auroc(preds_proxy, batch['labels'])
        if self.dataset in ['tamil', 'prop']:
            output['precision'] = self.precision_score(preds, batch['labels'])
            output['recall'] = self.recall(preds, batch['labels'])
            output['f1'] = self.f1(preds, batch['labels'])

        if calling_function == 'training' and self.fine_grained_labels and self.dataset in ['original', 'masked', 'inpainted']:
            for fine_grained_label, output_fine_grained in zip(self.fine_grained_labels, self.outputs_fine_grained):
                logits = output_fine_grained(features_pre_output).squeeze(dim=1)
                logits_for_super.append(torch.relu(logits))
                preds_proxy = torch.sigmoid(logits)
                preds = (preds_proxy >= 0.5).long()
                output[f'{fine_grained_label}_loss'] = self.cross_entropy_loss(logits, batch[fine_grained_label].float()) 
            logits_for_super = torch.stack(logits_for_super, dim=1) # [batch_size, 15]       
            logits = self.output_super(logits_for_super).squeeze(dim=1)
            preds_proxy = torch.sigmoid(logits)
            preds = (preds_proxy >= 0.5).long()
            output['super_loss'] = self.cross_entropy_loss(logits, batch['labels'].float()) 
            output['super_accuracy'] = self.acc(preds, batch['labels'])
            output['super_auroc'] = self.auroc(preds_proxy, batch['labels'])

        elif calling_function == 'validation' and self.fine_grained_labels and self.dataset in ['original', 'masked', 'inpainted']:
            for fine_grained_label, output_fine_grained in zip(self.fine_grained_labels, self.outputs_fine_grained):
                logits = output_fine_grained(features_pre_output).squeeze(dim=1)
                logits_for_super.append(torch.relu(logits))
                preds_proxy = torch.sigmoid(logits)
                preds = (preds_proxy >= 0.5).long()
                output[f'{fine_grained_label}_loss'] = self.cross_entropy_loss(logits, batch[fine_grained_label].float())
                output[f'{fine_grained_label}_accuracy'] = self.acc(preds, batch[fine_grained_label])
                output[f'{fine_grained_label}_auroc'] = self.auroc(preds_proxy, batch[fine_grained_label])
                output[f'{fine_grained_label}_precision'] = self.precision_score(preds, batch[fine_grained_label])
                output[f'{fine_grained_label}_recall'] = self.recall(preds, batch[fine_grained_label])
                output[f'{fine_grained_label}_f1'] = self.f1(preds, batch[fine_grained_label])
            logits_for_super = torch.stack(logits_for_super, dim=1) # [batch_size, 15]       
            logits = self.output_super(logits_for_super).squeeze(dim=1)
            preds_proxy = torch.sigmoid(logits)
            preds = (preds_proxy >= 0.5).long()
            output[f'super_loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
            output[f'super_accuracy'] = self.acc(preds, batch['labels'])
            output[f'super_auroc'] = self.auroc(preds_proxy, batch['labels'])
            output[f'super_precision'] = self.precision_score(preds, batch['labels'])
            output[f'super_recall'] = self.recall(preds, batch['labels'])
            output[f'super_f1'] = self.f1(preds, batch['labels'])

        elif calling_function == 'visualisation-v1':
            return image_features, text_features

        elif calling_function == 'visualisation-v2':
            return features

        return output
        
    def training_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx, calling_function='training')

        if self.weight_image_loss > 0:
            image_loss = output['image_loss']
        else:
            image_loss = 0
        
        if self.weight_text_loss > 0:
            text_loss = output['text_loss']
        else:
            text_loss = 0

        if self.fine_grained_labels  and self.dataset in ['original', 'masked', 'inpainted']:
            fine_grained_loss = 0
            for fine_grained_label in self.fine_grained_labels:
                fine_grained_loss += output[f'{fine_grained_label}_loss']
            fine_grained_loss /= len(self.fine_grained_labels)
            super_loss = output['super_loss']
        else:
            fine_grained_loss = 0.0
            super_loss = 0.0

        total_loss = output['loss'] + self.weight_image_loss * image_loss + self.weight_text_loss * text_loss + self.weight_fine_grained_loss * fine_grained_loss + self.weight_super_loss * super_loss

        self.log('train/total_loss', total_loss)
        self.log('train/loss', output['loss'])
        self.log('train/accuracy', output['accuracy'])
        self.log('train/auroc', output['auroc'])
        if self.dataset in ['tamil', 'prop']:
            self.log('train/precision', output['precision'])
            self.log('train/recall', output['recall'])
            self.log('train/f1', output['f1'])

        if self.weight_image_loss > 0:
            self.log('train/image_loss', image_loss)
        if self.weight_text_loss > 0:
            self.log('train/text_loss', text_loss)
        if self.fine_grained_labels and self.dataset in ['original', 'masked', 'inpainted']:
            self.log('train/fine_grained_loss', fine_grained_loss)
            self.log('train/super_loss', super_loss)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx, calling_function='validation')

        if self.weight_image_loss > 0:
            image_loss = output['image_loss']
        else:
            image_loss = 0
        
        if self.weight_text_loss > 0:
            text_loss = output['text_loss']
        else:
            text_loss = 0
        
        if self.fine_grained_labels and self.compute_fine_grained_metrics and self.dataset in ['original', 'masked', 'inpainted']:
            fine_grained_loss = torch.mean(torch.Tensor([output[f'{fine_grained_label}_loss'] for fine_grained_label in self.fine_grained_labels]))
            super_loss = output['super_loss']
        else:
            fine_grained_loss = 0.0
            super_loss = 0.0

        total_loss = output['loss'] + self.weight_image_loss * image_loss + self.weight_text_loss * text_loss + self.weight_fine_grained_loss * fine_grained_loss + self.weight_super_loss * super_loss
        
        self.log(f'val/total_loss', total_loss)
        self.log(f'val/loss', output['loss'])
        self.log(f'val/accuracy', output['accuracy'])
        self.log(f'val/auroc', output['auroc'])
        if self.dataset in ['tamil', 'prop']:
            self.log('val/precision', output['precision'])
            self.log('val/recall', output['recall'])
            self.log('val/f1', output['f1'])

        if self.weight_image_loss > 0:
            self.log(f'val/image_loss', image_loss)
        if self.weight_text_loss > 0:
            self.log(f'val/text_loss', text_loss)

        if self.fine_grained_labels and self.compute_fine_grained_metrics and self.dataset in ['original', 'masked', 'inpainted']:
            self.log(f'val/fine_grained_loss', fine_grained_loss)
            self.log(f'val/super_loss', super_loss)

            for fine_grained_label in self.fine_grained_labels:
                self.log(f'val-fine-grained/{fine_grained_label}_accuracy', output[f'{fine_grained_label}_accuracy'])
                self.log(f'val-fine-grained/{fine_grained_label}_auroc', output[f'{fine_grained_label}_auroc'])
                self.log(f'val-fine-grained/{fine_grained_label}_precision', output[f'{fine_grained_label}_precision'])
                self.log(f'val-fine-grained/{fine_grained_label}_recall', output[f'{fine_grained_label}_recall'])
                self.log(f'val-fine-grained/{fine_grained_label}_f1', output[f'{fine_grained_label}_f1'])
            
            self.log(f'val/super_loss', output['super_loss'])
            self.log(f'val/super_accuracy', output['super_accuracy'])
            self.log(f'val/super_auroc', output['super_auroc'])
        
        return total_loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        prefix_map = {
            0: 'dev_seen',
            1: 'test_seen',
            2: 'dev_unseen',
            3: 'test_unseen'
        }
        prefix = prefix_map[dataloader_idx]
        if dataloader_idx == 0:
            calling_function = 'validation'
        elif dataloader_idx == 1:
            calling_function = 'training'
            
        output = self.common_step(batch, batch_idx, calling_function=calling_function)
        
        self.log(f'{prefix}/accuracy', output['accuracy'])
        self.log(f'{prefix}/auroc', output['auroc'])
        if self.dataset in ['tamil', 'prop']:
            self.log(f'{prefix}/precision', output['precision'])
            self.log(f'{prefix}/recall', output['recall'])
            self.log(f'{prefix}/f1', output['f1'])

        if self.fine_grained_labels and self.dataset in ['original', 'masked', 'inpainted']:
            self.log(f'{prefix}/super_accuracy', output['super_accuracy'])
            self.log(f'{prefix}/super_auroc', output['super_auroc'])

            if dataloader_idx  == 0:
                for fine_grained_label in self.fine_grained_labels:
                    self.log(f'{prefix}-fine-grained/{fine_grained_label}_accuracy', output[f'{fine_grained_label}_accuracy'])
                    self.log(f'{prefix}-fine-grained/{fine_grained_label}_auroc', output[f'{fine_grained_label}_auroc'])
                    self.log(f'{prefix}-fine-grained/{fine_grained_label}_precision', output[f'{fine_grained_label}_precision'])
                    self.log(f'{prefix}-fine-grained/{fine_grained_label}_recall', output[f'{fine_grained_label}_recall'])
                    self.log(f'{prefix}-fine-grained/{fine_grained_label}_f1', output[f'{fine_grained_label}_f1'])
                
        
        return output

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

    def test_epoch_end(self, validation_step_outputs):
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
    compute_fine_grained_metrics = True
    model = CLIPClassifier(args=args, fine_grained_labels=fine_grained_labels, compute_fine_grained_metrics = compute_fine_grained_metrics)

    return model