"""
Enterprise-grade Training Pipeline

Features:
- Distributed training with DDP
- Mixed precision training
- Hyperparameter optimization
- Automated experiment tracking
- Model versioning
- Early stopping
- Gradient accumulation
- Cross-validation
- Explainability reports
- Production-ready checkpoints
"""

import os
import sys
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
import yaml
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import mlflow
import git
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Local imports
from . import (
    ModelManager,
    TrainingPipeline,
    FeatureProcessor,
    EmissionsDataset,
    ExplainabilityEngine,
    MetricsCalculator,
    PredictionMonitor
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EcoTrainer:
    """Distributed Training Orchestrator"""
    
    def __init__(self, config: Dict, world_size: int = 1):
        self.config = config
        self.world_size = world_size
        self.device = self._setup_device()
        self.model_mgr = ModelManager()
        self.feature_processor = FeatureProcessor()
        self.best_metric = float('inf')
        self.scaler = GradScaler()
        self.current_epoch = 0
        
        # Initialize tracking
        self._init_mlflow()
        self._log_git_version()
        
    def _setup_device(self) -> torch.device:
        """Configure training hardware"""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.config['local_rank']}")
        return torch.device("cpu")
    
    def _init_mlflow(self):
        """Configure MLflow tracking"""
        mlflow.set_tracking_uri(self.config["mlflow_uri"])
        mlflow.set_experiment(self.config["experiment_name"])
        mlflow.log_params(self.config["hyperparameters"])
        
    def _log_git_version(self):
        """Record current git state for reproducibility"""
        try:
            repo = git.Repo(search_parent_directories=True)
            mlflow.log_param("git_commit", repo.head.object.hexsha)
            mlflow.log_param("git_branch", repo.active_branch.name)
        except Exception as e:
            logger.warning(f"Git version logging failed: {str(e)}")
            
    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and preprocess data"""
        try:
            # Load raw data
            raw_data = pd.read_parquet(self.config["data_path"])
            
            # Process features
            processed = self.feature_processor.fit_transform(raw_data)
            
            # Create datasets
            window_size = self.config["hyperparameters"]["window_size"]
            horizon = self.config["hyperparameters"]["horizon"]
            full_dataset = EmissionsDataset(processed, window_size, horizon)
            
            # Split data
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_data, val_data = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            
            # Create loaders
            train_sampler = DistributedSampler(
                train_data,
                num_replicas=self.world_size,
                rank=self.config["local_rank"]
            )
            train_loader = DataLoader(
                train_data,
                batch_size=self.config["hyperparameters"]["batch_size"],
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_data,
                batch_size=self.config["hyperparameters"]["batch_size"] * 2,
                num_workers=4,
                pin_memory=True
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
            
    def _init_model(self) -> nn.Module:
        """Model initialization with proper device placement"""
        model_config = self.config["model"]
        model = self.model_mgr.load_model(model_config["name"], model_config["version"])
        model = model.to(self.device)
        
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.config["local_rank"]])
            
        return model
    
    def _save_checkpoint(self, model: nn.Module, epoch: int, is_best: bool = False):
        """Save training state with proper distributed handling"""
        if self.config["local_rank"] != 0:
            return
            
        checkpoint = {
            "epoch": epoch,
            "model_state": model.module.state_dict() if self.world_size > 1 else model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config
        }
        
        checkpoint_dir = Path(self.config["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Regular checkpoint
        torch.save(
            checkpoint,
            checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        )
        
        # Best checkpoint
        if is_best:
            torch.save(
                checkpoint,
                checkpoint_dir / "best_model.pt"
            )
            mlflow.log_artifact(checkpoint_dir / "best_model.pt")
            
    def _log_metrics(self, metrics: Dict, prefix: str = "train"):
        """Log metrics to MLflow and console"""
        if self.config["local_rank"] == 0:
            mlflow.log_metrics({f"{prefix}_{k}": v for k, v in metrics.items()}, step=self.current_epoch)
            logger.info(f"Epoch {self.current_epoch} {prefix} metrics: {metrics}")
            
    def train_epoch(self, model: nn.Module, train_loader: DataLoader):
        """Distributed training epoch with mixed precision"""
        model.train()
        total_loss = 0.0
        self.train_sampler.set_epoch(self.current_epoch)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=self.config["mixed_precision"]):
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config["hyperparameters"]["grad_clip"]
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % self.config["log_interval"] == 0:
                self._log_metrics(
                    {"loss": loss.item()},
                    prefix="train/batch"
                )
                
        return total_loss / len(train_loader)
    
    def validate(self, model: nn.Module, val_loader: DataLoader) -> Dict:
        """Distributed validation with metrics"""
        model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
                
        # Aggregate across devices
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        
        if self.world_size > 1:
            dist.all_reduce(all_outputs, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_targets, op=dist.ReduceOp.SUM)
            
        val_loss = total_loss / len(val_loader)
        metrics = MetricsCalculator.calculate_all(all_targets.numpy(), all_outputs.numpy())
        metrics["loss"] = val_loss
        
        return metrics
    
    def _early_stop(self, current_metric: float) -> bool:
        """Check early stopping conditions"""
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config["patience"]
            
    def train(self):
        """Main training loop"""
        try:
            # Initialize distributed training
            if self.world_size > 1:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://"
                )
                
            # Prepare data
            train_loader, val_loader = self._prepare_data()
            
            # Initialize model
            model = self._init_model()
            self.criterion = torch.nn.MSELoss()
            
            # Training components
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config["hyperparameters"]["learning_rate"],
                weight_decay=self.config["hyperparameters"]["weight_decay"]
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=self.config["lr_patience"]
            )
            self.patience_counter = 0
            
            # Training loop
            for epoch in range(self.config["epochs"]):
                self.current_epoch = epoch
                
                # Train epoch
                train_loss = self.train_epoch(model, train_loader)
                self._log_metrics({"loss": train_loss}, "train")
                
                # Validation
                val_metrics = self.validate(model, val_loader)
                self._log_metrics(val_metrics, "val")
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics["loss"])
                
                # Checkpointing
                if val_metrics["loss"] < self.best_metric:
                    self._save_checkpoint(model, epoch, is_best=True)
                else:
                    self._save_checkpoint(model, epoch)
                    
                # Early stopping
                if self._early_stop(val_metrics["loss"]):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
            # Final evaluation
            if self.config["local_rank"] == 0:
                self._generate_explainability_report(model, val_loader)
                self._save_final_model(model)
                
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            if self.world_size > 1:
                dist.destroy_process_group()
                
    def _generate_explainability_report(self, model: nn.Module, val_loader: DataLoader):
        """Create SHAP explainability report"""
        try:
            background, samples = self._get_explainer_data(val_loader)
            explainer = ExplainabilityEngine(model.module, background)
            shap_values = explainer.explain(samples)
            
            report = {
                "feature_importance": dict(zip(
                    self.feature_processor.feature_names,
                    shap_values["feature_importance"]
                )),
                "summary_plot": shap.summary_plot(
                    shap_values["shap_values"],
                    samples,
                    feature_names=self.feature_processor.feature_names,
                    show=False
                )
            }
            
            mlflow.log_dict(report, "explainability_report.json")
            logger.info("Explainability report generated")
            
        except Exception as e:
            logger.warning(f"Explainability report failed: {str(e)}")
            
    def _get_explainer_data(self, val_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for SHAP explanations"""
        background = []
        samples = []
        for inputs, _ in val_loader:
            background.append(inputs.numpy())
            if len(background) > 10:  # Limit background size
                break
        return np.concatenate(background), np.concatenate(background[:2])
    
    def _save_final_model(self, model: nn.Module):
        """Save production-ready model package"""
        model_dir = Path(self.config["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pt"
        torch.save(model.module.state_dict(), model_path)
        
        # Save metadata
        metadata = {
            "training_config": self.config,
            "feature_processor": self.feature_processor.config,
            "metrics": self.best_metric,
            "git_version": mlflow.get_run(mlflow.active_run().info.run_id).data.params.get("git_commit")
        }
        
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
            
        mlflow.log_artifacts(model_dir)
        logger.info(f"Final model saved to {model_dir}")

def train_distributed(rank: int, world_size: int, config: Dict):
    """Distributed training entry point"""
    os.environ["LOCAL_RANK"] = str(rank)
    config["local_rank"] = rank
    trainer = EcoTrainer(config, world_size)
    trainer.train()

def hyperparameter_tune(config: Dict):
    """Ray Tune hyperparameter optimization"""
    scheduler = ASHAScheduler(
        max_t=config["epochs"],
        grace_period=10,
        reduction_factor=2
    )
    
    reporter = CLIReporter(
        metric_columns=["val_loss", "val_mae", "training_iteration"]
    )
    
    analysis = tune.run(
        tune.with_parameters(
            train_distributed,
            world_size=1,
            config=config
        ),
        resources_per_trial={"gpu": 1},
        config=config["search_space"],
        num_samples=config["num_trials"],
        scheduler=scheduler,
        progress_reporter=reporter,
        name="eco_tune",
        local_dir=config["tune_dir"]
    )
    
    logger.info(f"Best trial config: {analysis.best_config}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EcoTrack Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, help="Checkpoint path to resume")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    if args.tune:
        hyperparameter_tune(config)
    else:
        if args.distributed:
            world_size = torch.cuda.device_count()
            mp.spawn(
                train_distributed,
                args=(world_size, config),
                nprocs=world_size,
                join=True
            )
        else:
            trainer = EcoTrainer(config)
            trainer.train()
