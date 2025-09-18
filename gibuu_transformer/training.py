"""
Training utilities for GiBUU Transformer.
"""

import torch
import lightning.pytorch as pl
from .model import GiBUUTransformer
from .loss import particle_gpt_loss


class GiBUUSeqOfSetsModel(pl.LightningModule):
    """
    Lightning module for GiBUU particle sequence generation using GPT-style autoregressive training.
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Model
        self.net = GiBUUTransformer(cfg['model'])
        
        # Loss weights
        self.type_loss_weight = cfg.get('type_loss_weight', 1.0)
        self.feat_loss_weight = cfg.get('feat_loss_weight', 1.0)
        
        # Training config
        self.lr = cfg.get('lr', 1e-4)
        self.weight_decay = cfg.get('weight_decay', 1e-5)
        
        self.save_hyperparameters()

    def forward(self, encoded_ids, particle_feats, padding_mask=None, causal_mask=None):
        """
        Forward pass through the model.
        
        Parameters:
        -----------
        encoded_ids: tensor (batch, seq_len)
            Encoded particle IDs
        particle_feats: tensor (batch, seq_len, 7)
            Particle features [x, y, z, E, Px, Py, Pz]
        padding_mask: tensor (batch, seq_len), optional
            Boolean padding mask
        causal_mask: tensor (batch, seq_len, seq_len), optional
            Boolean causal mask
            
        Returns:
        --------
        output: dict
            Model output with 'particle_types' and 'particle_feats'
        """
        return self.net(encoded_ids, particle_feats, padding_mask, causal_mask)

    def training_step(self, batch, batch_idx):
        """
        Training step.
        """
        # Forward pass
        causal_mask = batch.get('causal_mask', None)
        output = self(
            batch['encoded_ids'], 
            batch['particle_feats'], 
            batch['padding_mask'],
            causal_mask
        )
        
        # Compute loss
        loss_dict = particle_gpt_loss(output, batch)
        
        # Log losses
        self.log('train_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('train_type_loss', loss_dict['type_loss'])
        self.log('train_feat_loss', loss_dict['feat_loss'])
        
        return loss_dict['total_loss']

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        """
        # Forward pass
        causal_mask = batch.get('causal_mask', None)
        output = self(
            batch['encoded_ids'], 
            batch['particle_feats'], 
            batch['padding_mask'],
            causal_mask
        )
        
        # Compute loss
        loss_dict = particle_gpt_loss(output, batch)
        
        # Log losses
        self.log('val_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('val_type_loss', loss_dict['type_loss'])
        self.log('val_feat_loss', loss_dict['feat_loss'])
        
        return loss_dict['total_loss']

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=100,  # Adjust based on your training epochs
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


def train_model(cfg, seqdata, checkpoint_path=None, resume_training=False, 
                log_dir="lightning_logs", exp_name="test"):
    """
    Train the model with checkpointing and resume capabilities.
    
    Parameters:
    -----------
    cfg: dict
        Configuration dictionary
    seqdata: dict
        Processed sequence data
    checkpoint_path: str, optional
        Path to checkpoint for resuming training
    resume_training: bool
        Whether to resume training from checkpoint
    log_dir: str
        Directory for logging
    exp_name: str
        Experiment name
        
    Returns:
    --------
    model: GiBUUSeqOfSetsModel
        Trained model
    trainer: pl.Trainer
        PyTorch Lightning trainer
    """
    from .data_processing import create_dataloaders
    
    # Create model
    if resume_training and checkpoint_path:
        print(f"Resuming training from: {checkpoint_path}")
        model = GiBUUSeqOfSetsModel.load_from_checkpoint(checkpoint_path)
    else:
        print("Starting new training")
        model = GiBUUSeqOfSetsModel(cfg)
    
    # Create dataloaders
    train_cfg = cfg["trainer"]
    train_loader, val_loader = create_dataloaders(
        seqdata,
        batch_size=train_cfg["batch_size"],
        max_seq_len=train_cfg["max_seq_len"]
    )

    # Create trainer with checkpointing
    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator='auto',
        devices='auto',
        log_every_n_steps=train_cfg["log_every_n_steps"],
        
        # Checkpointing
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=f'{log_dir}/{exp_name}/checkpoints',
                filename='model-{epoch:02d}-{val_loss:.4f}',
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                every_n_epochs=train_cfg.get('every_n_epochs', 2),
                save_last=True
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.0,
                patience=10,
                mode='min',
                stopping_threshold=None,
                verbose=True
            )
        ],
        
        # Logging
        logger=pl.loggers.CSVLogger(
            save_dir=log_dir,
            name=exp_name,
            version=None
        )
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    
    return model, trainer
