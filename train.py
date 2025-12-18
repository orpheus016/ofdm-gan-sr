# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Training Script
# =============================================================================
"""
TRAINING SCRIPT FOR CWGAN-GP
=============================

This script implements the complete training loop for the Conditional
Wasserstein GAN with Gradient Penalty for OFDM signal reconstruction.

Training Algorithm:
-------------------
For each epoch:
    For each batch:
        1. Sample real pairs: (clean_signal, noisy_signal)
        2. Generate fake signals: fake = G(noisy_signal)
        
        # Train Discriminator (n_critic times)
        3. Compute D(real, condition) and D(fake, condition)
        4. Compute gradient penalty on interpolated samples
        5. Update D to minimize: L_D = E[D(fake)] - E[D(real)] + λ·GP
        
        # Train Generator (1 time)
        6. Generate fake signals
        7. Compute D(fake, condition)
        8. Compute reconstruction loss L1(fake, real)
        9. Update G to minimize: L_G = -E[D(fake)] + λ_rec·L1

Usage:
------
    python train.py --config config/config.yaml
    python train.py --epochs 100 --batch_size 32 --lr 1e-4
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.generator import MiniGenerator
from models.discriminator import MiniDiscriminator, compute_gradient_penalty
from utils.dataset import SyntheticOFDMDataset, OFDMDataset, create_dataloader
from utils.quantization import export_weights_fpga, QuantizationConfig


class CWGANGPTrainer:
    """
    Trainer class for CWGAN-GP OFDM Signal Reconstruction.
    
    Implements the complete training loop with:
    - Wasserstein loss with gradient penalty
    - Conditional training (noisy → clean)
    - L1 reconstruction loss
    - Learning rate scheduling
    - Checkpointing
    - TensorBoard logging
    """
    
    def __init__(
        self,
        generator: MiniGenerator,
        discriminator: MiniDiscriminator,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: torch.device = None,
        config: Dict = None
    ):
        """
        Initialize trainer.
        
        Args:
            generator: Mini U-Net generator model
            discriminator: Mini Critic model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Computation device
            config: Training configuration dictionary
        """
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Configuration with defaults
        self.config = config or {}
        self._setup_config()
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.lr_g,
            betas=self.betas
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d,
            betas=self.betas
        )
        
        # Schedulers
        self.scheduler_G = optim.lr_scheduler.StepLR(
            self.optimizer_G,
            step_size=self.scheduler_step,
            gamma=self.scheduler_gamma
        )
        self.scheduler_D = optim.lr_scheduler.StepLR(
            self.optimizer_D,
            step_size=self.scheduler_step,
            gamma=self.scheduler_gamma
        )
        
        # Logging
        self.writer = None
        self.log_dir = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _setup_config(self):
        """Setup configuration with defaults."""
        training_config = self.config.get('training', {})
        
        # Basic settings
        self.epochs = training_config.get('epochs', 200)
        self.batch_size = training_config.get('batch_size', 32)
        
        # WGAN-GP specific
        self.n_critic = training_config.get('n_critic', 5)
        self.gp_weight = training_config.get('gp_weight', 10.0)
        
        # Loss weights
        loss_config = training_config.get('loss', {})
        self.adv_weight = loss_config.get('adversarial_weight', 1.0)
        self.rec_weight = loss_config.get('reconstruction_weight', 100.0)
        
        # Optimizer settings
        opt_config = training_config.get('optimizer', {})
        self.lr_g = float(opt_config.get('lr_generator', 1e-4))
        self.lr_d = float(opt_config.get('lr_discriminator', 1e-4))
        self.betas = tuple(opt_config.get('betas', [0.0, 0.9]))
        
        # Scheduler settings
        sched_config = training_config.get('scheduler', {})
        self.scheduler_step = sched_config.get('step_size', 50)
        self.scheduler_gamma = sched_config.get('gamma', 0.5)
        
        # Checkpointing
        self.checkpoint_interval = training_config.get('checkpoint_interval', 10)
        self.save_best = training_config.get('save_best', True)
        
        # Logging
        log_config = self.config.get('logging', {})
        self.log_interval = log_config.get('log_interval', 100)
        
        # Paths
        paths_config = self.config.get('paths', {})
        self.checkpoint_dir = paths_config.get('checkpoint_dir', './checkpoints')
        self.log_dir_base = paths_config.get('log_dir', './logs')
        
    def setup_logging(self, experiment_name: str = None):
        """Setup TensorBoard logging."""
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        self.log_dir = Path(self.log_dir_base) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        
        # Log config
        if self.config:
            self.writer.add_text('config', str(self.config))
            
    def train_discriminator(
        self,
        real_signal: torch.Tensor,
        noisy_signal: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train discriminator for one step.
        
        WGAN-GP Critic Training:
            L_D = E[D(G(c), c)] - E[D(x, c)] + λ·GP
            
        Args:
            real_signal: Clean signal [B, 2, L]
            noisy_signal: Noisy signal (condition) [B, 2, L]
            
        Returns:
            Dictionary of loss values
        """
        self.discriminator.train()
        self.optimizer_D.zero_grad()
        
        batch_size = real_signal.size(0)
        
        # Generate fake samples
        with torch.no_grad():
            fake_signal = self.generator(noisy_signal)
            
        # Discriminator scores
        # D(real, condition) - should be high
        d_real = self.discriminator(real_signal, noisy_signal)
        
        # D(fake, condition) - should be low
        d_fake = self.discriminator(fake_signal, noisy_signal)
        
        # Wasserstein loss: E[D(fake)] - E[D(real)]
        # We want to maximize E[D(real)] - E[D(fake)], so minimize E[D(fake)] - E[D(real)]
        wasserstein_loss = d_fake.mean() - d_real.mean()
        
        # Gradient penalty
        gp = compute_gradient_penalty(
            self.discriminator,
            real_signal,
            fake_signal,
            noisy_signal,
            self.device
        )
        
        # Total discriminator loss
        d_loss = wasserstein_loss + self.gp_weight * gp
        
        # Backward and update
        d_loss.backward()
        self.optimizer_D.step()
        
        return {
            'd_loss': d_loss.item(),
            'wasserstein_distance': -wasserstein_loss.item(),
            'gradient_penalty': gp.item(),
            'd_real_mean': d_real.mean().item(),
            'd_fake_mean': d_fake.mean().item()
        }
        
    def train_generator(
        self,
        real_signal: torch.Tensor,
        noisy_signal: torch.Tensor
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        Train generator for one step.
        
        Generator Loss:
            L_G = -E[D(G(c), c)] + λ_rec·||G(c) - x||_1
            
        Args:
            real_signal: Clean signal [B, 2, L]
            noisy_signal: Noisy signal (condition) [B, 2, L]
            
        Returns:
            Tuple of (loss_dict, generated_signal)
        """
        self.generator.train()
        self.optimizer_G.zero_grad()
        
        # Generate fake samples
        fake_signal = self.generator(noisy_signal)
        
        # Adversarial loss: maximize D(fake) → minimize -D(fake)
        d_fake = self.discriminator(fake_signal, noisy_signal)
        adv_loss = -d_fake.mean()
        
        # Reconstruction loss (L1)
        rec_loss = nn.functional.l1_loss(fake_signal, real_signal)
        
        # Total generator loss
        g_loss = self.adv_weight * adv_loss + self.rec_weight * rec_loss
        
        # Backward and update
        g_loss.backward()
        self.optimizer_G.step()
        
        return {
            'g_loss': g_loss.item(),
            'adv_loss': adv_loss.item(),
            'rec_loss': rec_loss.item()
        }, fake_signal.detach()
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average losses
        """
        epoch_losses = {
            'd_loss': 0,
            'g_loss': 0,
            'wasserstein_distance': 0,
            'rec_loss': 0
        }
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            noisy = batch['noisy'].to(self.device)
            clean = batch['clean'].to(self.device)
            
            # Train discriminator (n_critic times per generator step)
            d_losses = {}
            for _ in range(self.n_critic):
                d_losses = self.train_discriminator(clean, noisy)
                
            # Train generator
            g_losses, fake = self.train_generator(clean, noisy)
            
            # Accumulate losses
            epoch_losses['d_loss'] += d_losses['d_loss']
            epoch_losses['g_loss'] += g_losses['g_loss']
            epoch_losses['wasserstein_distance'] += d_losses['wasserstein_distance']
            epoch_losses['rec_loss'] += g_losses['rec_loss']
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'D': f"{d_losses['d_loss']:.4f}",
                'G': f"{g_losses['g_loss']:.4f}",
                'W': f"{d_losses['wasserstein_distance']:.4f}",
                'L1': f"{g_losses['rec_loss']:.4f}"
            })
            
            # Log to tensorboard
            if self.writer and self.global_step % self.log_interval == 0:
                self.writer.add_scalar('train/d_loss', d_losses['d_loss'], self.global_step)
                self.writer.add_scalar('train/g_loss', g_losses['g_loss'], self.global_step)
                self.writer.add_scalar('train/wasserstein', d_losses['wasserstein_distance'], self.global_step)
                self.writer.add_scalar('train/rec_loss', g_losses['rec_loss'], self.global_step)
                self.writer.add_scalar('train/gp', d_losses['gradient_penalty'], self.global_step)
                
            self.global_step += 1
            
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
            
        return epoch_losses
        
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
            
        self.generator.eval()
        self.discriminator.eval()
        
        val_losses = {
            'rec_loss': 0,
            'mse': 0
        }
        n_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                noisy = batch['noisy'].to(self.device)
                clean = batch['clean'].to(self.device)
                
                fake = self.generator(noisy)
                
                # L1 loss
                rec_loss = nn.functional.l1_loss(fake, clean)
                
                # MSE
                mse = nn.functional.mse_loss(fake, clean)
                
                val_losses['rec_loss'] += rec_loss.item()
                val_losses['mse'] += mse.item()
                n_batches += 1
                
        for key in val_losses:
            val_losses[key] /= n_batches
            
        return val_losses
        
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = Path(path).parent / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
    def train(self, epochs: int = None, experiment_name: str = None):
        """
        Full training loop.
        
        Args:
            epochs: Number of epochs (overrides config)
            experiment_name: Name for logging/checkpoints
        """
        if epochs is None:
            epochs = self.epochs
            
        # Setup
        self.setup_logging(experiment_name)
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"N critic: {self.n_critic}")
        print(f"GP weight: {self.gp_weight}")
        print(f"Reconstruction weight: {self.rec_weight}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate()
            
            # Log epoch results
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train - D: {train_losses['d_loss']:.4f}, "
                  f"G: {train_losses['g_loss']:.4f}, "
                  f"W: {train_losses['wasserstein_distance']:.4f}, "
                  f"L1: {train_losses['rec_loss']:.4f}")
            
            if val_losses:
                print(f"  Val - L1: {val_losses['rec_loss']:.4f}, "
                      f"MSE: {val_losses['mse']:.4f}")
                      
                if self.writer:
                    self.writer.add_scalar('val/rec_loss', val_losses['rec_loss'], epoch)
                    self.writer.add_scalar('val/mse', val_losses['mse'], epoch)
                    
            # Update schedulers
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Save checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0:
                ckpt_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
                self.save_checkpoint(str(ckpt_path))
                print(f"  Saved checkpoint: {ckpt_path}")
                
            # Save best model
            if self.save_best and val_losses:
                if val_losses['rec_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['rec_loss']
                    self.save_checkpoint(
                        str(checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'),
                        is_best=True
                    )
                    print(f"  New best model! Val L1: {self.best_val_loss:.4f}")
                    
        # Training complete
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/3600:.2f} hours")
        
        # Save final model
        final_path = checkpoint_dir / 'final_model.pt'
        self.save_checkpoint(str(final_path))
        
        # Export weights for FPGA (can be disabled via config)
        if self.config.get('export_after_training', True):
            export_dir = Path(self.config.get('paths', {}).get('export_dir', './export'))
            export_dir.mkdir(parents=True, exist_ok=True)
            
            print("\nExporting weights for FPGA...")
            quant_config = QuantizationConfig()
            export_weights_fpga(self.generator, str(export_dir / 'generator'), quant_config)
        
        if self.writer:
            self.writer.close()
            
        return self.generator, self.discriminator


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train CWGAN-GP for OFDM')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (for testing)')
    parser.add_argument('--skip_export', action='store_true',
                        help='Skip FPGA export after training')
    parser.add_argument('--export_only', action='store_true',
                        help='Only export weights (no training)')
    parser.add_argument('--export_checkpoint', type=str, default=None,
                        help='Checkpoint path to load for export-only mode')
    parser.add_argument('--export_dir', type=str, default=None,
                        help='Directory to write exported weights (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override with command line args
    if args.epochs:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.lr:
        config.setdefault('training', {}).setdefault('optimizer', {})['lr_generator'] = args.lr
        config.setdefault('training', {}).setdefault('optimizer', {})['lr_discriminator'] = args.lr
    if args.skip_export:
        config['export_after_training'] = False
    if args.export_dir:
        config.setdefault('paths', {})['export_dir'] = args.export_dir

    # Export-only mode: load checkpoint and export without training
    if args.export_only:
        ckpt_path = args.export_checkpoint
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError("--export_only requires --export_checkpoint pointing to a valid file")
        
        # Create model and load checkpoint
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = MiniGenerator()
        checkpoint = torch.load(ckpt_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        
        # Export directory
        export_dir = Path(config.get('paths', {}).get('export_dir', './export'))
        export_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Exporting weights from checkpoint: {ckpt_path}")
        print(f"Writing to: {export_dir}")
        
        # Run export (device-agnostic; utils handles cpu conversion)
        quant_config = QuantizationConfig()
        export_weights_fpga(generator, str(export_dir / 'generator'), quant_config)
        print("Export complete.")
        return
        
    # Get training params
    batch_size = config.get('training', {}).get('batch_size', 64)
    frame_length = config.get('ofdm', {}).get('frame_length', 16)  # Mini architecture: 16 samples
    
    # Create dataset
    if args.synthetic:
        print("Using synthetic OFDM dataset")
        train_dataset = SyntheticOFDMDataset(
            n_samples=10000,
            frame_length=frame_length,
            snr_range=tuple(config.get('channel', {}).get('snr_range', [5, 20]))
        )
        val_dataset = SyntheticOFDMDataset(
            n_samples=1000,
            frame_length=frame_length,
            snr_range=tuple(config.get('channel', {}).get('snr_range', [5, 20]))
        )
    else:
        # Try to load from data directory
        data_dir = config.get('paths', {}).get('train_dir', './data/train')
        if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
            print(f"Loading data from {data_dir}")
            train_dataset = OFDMDataset(
                image_dir=data_dir,
                frame_length=frame_length,
                modulation=config.get('ofdm', {}).get('modulation', 'QPSK'),
                snr_range=tuple(config.get('channel', {}).get('snr_range', [5, 20]))
            )
            val_dataset = None  # TODO: Add validation split
        else:
            print("No data found, using synthetic dataset")
            train_dataset = SyntheticOFDMDataset(n_samples=10000, frame_length=frame_length)
            val_dataset = SyntheticOFDMDataset(n_samples=1000, frame_length=frame_length)
            
    # Create dataloaders
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, num_workers=0)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if val_dataset else None
    
    # Create models
    generator = MiniGenerator()
    discriminator = MiniDiscriminator()
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Create trainer
    trainer = CWGANGPTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
        
    # Train
    trainer.train(experiment_name=args.experiment)


if __name__ == "__main__":
    main()
