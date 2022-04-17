import pytorch_lightning as pl
import torch
from transformers import CLIPModel



class HatefulMemesCLIP(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.clip = CLIPModel.from_pretrained(args.clip_pretrained_model)

    def forward(self, batch):
        output = self.clip(pixel_values=batch['pixel_values'][0], input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], return_loss=True, return_dict=True)

        return output
        
    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = output.loss
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = output.loss
        self.log('val/loss', loss)

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]}
            ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer



def create_model(args):
    model = HatefulMemesCLIP(args=args)

    return model