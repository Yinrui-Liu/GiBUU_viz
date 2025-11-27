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


class GiBUUPropagationLightning(pl.LightningModule):
    """
    Lightning module for GiBUU propagation model with zero-inflated architecture.
    
    This module handles training for predicting particle feature changes during propagation,
    including step-level interaction prediction and optionally zero-inflated E/p predictions.
    """
    def __init__(self, cfg):
        super().__init__()
        
        from .model import GiBUUPropagationModel
        from .loss import gibuu_propagation_loss
        
        model_cfg = cfg.get('model', {})
        self.net = GiBUUPropagationModel(
            num_particle_types=model_cfg.get('num_particle_types', 4096),
            feature_dim=model_cfg.get('feature_dim', 7),
            hidden_dims=model_cfg.get('hidden_dims', [256, 128, 64]),
            dropout=model_cfg.get('dropout', 0.1),
            aggregation_method=model_cfg.get('aggregation_method', 'mean'),
            use_zero_inflation=model_cfg.get('use_zero_inflation', True)
        )
        
        # Loss weights
        self.feature_loss_weight = cfg.get('feature_loss_weight', 1.0)
        self.position_loss_weight = cfg.get('position_loss_weight', 1.0)
        self.em_zero_loss_weight = cfg.get('em_zero_loss_weight', 1.0)
        self.em_value_loss_weight = cfg.get('em_value_loss_weight', 1.0)
        self.interaction_loss_weight = cfg.get('interaction_loss_weight', 1.0)
        self.apply_feature_loss_only_when_no_interaction = cfg.get('apply_feature_loss_only_when_no_interaction', True)
        self.pos_weight = cfg.get('pos_weight', None)
        self.use_huber_loss = cfg.get('use_huber_loss', True)
        self.huber_delta = cfg.get('huber_delta', 1.0)
        self.use_zero_inflation = model_cfg.get('use_zero_inflation', True)
        
        # Training config
        self.lr = cfg.get('lr', 1e-4)
        self.weight_decay = cfg.get('weight_decay', 1e-5)
        
        self.save_hyperparameters()
    
    def forward(self, particle_type, features, padding_mask=None):
        """Forward pass through the propagation model."""
        return self.net(particle_type, features, padding_mask)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        from .loss import gibuu_propagation_loss
        
        output = self(batch['particle_types'], batch['input_features'], batch['padding_mask'])
        
        loss_dict = gibuu_propagation_loss(
            output,
            batch['input_features'],
            batch['target_features'],
            batch['target_interaction'],
            batch['padding_mask'],
            feature_loss_weight=self.feature_loss_weight,
            position_loss_weight=self.position_loss_weight,
            em_zero_loss_weight=self.em_zero_loss_weight,
            em_value_loss_weight=self.em_value_loss_weight,
            interaction_loss_weight=self.interaction_loss_weight,
            apply_feature_loss_only_when_no_interaction=self.apply_feature_loss_only_when_no_interaction,
            pos_weight=self.pos_weight,
            use_huber_loss=self.use_huber_loss,
            huber_delta=self.huber_delta,
            use_zero_inflation=self.use_zero_inflation
        )
        
        # Log all losses
        self.log('train_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('train_feature_loss', loss_dict['feature_loss'])
        self.log('train_position_loss', loss_dict['position_loss'])
        self.log('train_em_zero_loss', loss_dict['em_zero_loss'])
        self.log('train_em_value_loss', loss_dict['em_value_loss'])
        self.log('train_interaction_loss', loss_dict['interaction_loss'])
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        from .loss import gibuu_propagation_loss
        
        output = self(batch['particle_types'], batch['input_features'], batch['padding_mask'])
        
        loss_dict = gibuu_propagation_loss(
            output,
            batch['input_features'],
            batch['target_features'],
            batch['target_interaction'],
            batch['padding_mask'],
            feature_loss_weight=self.feature_loss_weight,
            position_loss_weight=self.position_loss_weight,
            em_zero_loss_weight=self.em_zero_loss_weight,
            em_value_loss_weight=self.em_value_loss_weight,
            interaction_loss_weight=self.interaction_loss_weight,
            apply_feature_loss_only_when_no_interaction=self.apply_feature_loss_only_when_no_interaction,
            pos_weight=self.pos_weight,
            use_huber_loss=self.use_huber_loss,
            huber_delta=self.huber_delta,
            use_zero_inflation=self.use_zero_inflation
        )
        
        # Log all losses
        self.log('val_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('val_feature_loss', loss_dict['feature_loss'])
        self.log('val_position_loss', loss_dict['position_loss'])
        self.log('val_em_zero_loss', loss_dict['em_zero_loss'])
        self.log('val_em_value_loss', loss_dict['em_value_loss'])
        self.log('val_interaction_loss', loss_dict['interaction_loss'])
        
        return loss_dict['total_loss']
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1000,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
