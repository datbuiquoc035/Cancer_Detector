import os
import time
import json
import random
import uuid as _uuid
import threading
from collections import deque
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from queue import Queue

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from features import extract_features_from_df, FEATURE_COUNT, REQUIRED_COLUMNS
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve, auc
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class ClinVarNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None, dropout: float = 0.3):
        super(ClinVarNet, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Trainer:
    def __init__(self, log_queue: Queue, output_dir: str = ''):
        self.log_queue = log_queue
        self.output_dir = output_dir
        self.log_buffer = deque(maxlen=200)
        self._log_buffer_lock = threading.Lock()
        self._is_training = False
        self._stop_requested = False
        self._training_event = threading.Event()
        self._stop_event = threading.Event()
        self.model: Optional[ClinVarNet] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.criterion: Optional[nn.BCELoss] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.device: torch.device = torch.device('cpu')
        self.input_dim: Optional[int] = None
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.train_loss: float = 0.0
        self.train_acc: float = 0.0
        self.val_loss: float = 0.0
        self.val_acc: float = 0.0
        self.best_val_acc: float = 0.0
        self.eta_str: str = '-'

        self.epoch_history: List[int] = []
        self.loss_history: List[float] = []
        self.acc_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.val_acc_history: List[float] = []

        self.training_thread: Optional[threading.Thread] = None
        self.chart_dir: Optional[str] = None
        self.chart_files: Dict[str, str] = {}

    @property
    def is_training(self) -> bool:
        return self._is_training

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @stop_requested.setter
    def stop_requested(self, value: bool):
        self._stop_requested = value
        if value:
            self._stop_event.set()
        else:
            self._stop_event.clear()

    def log(self, message: str):
        msg = {'type': 'log', 'message': message}
        self.log_queue.put(msg)
        with self._log_buffer_lock:
            self.log_buffer.append(msg)

    def emit_metrics(self, epoch: int, total_epochs: int, train_loss: float, train_acc: float,
                     val_loss: float, val_acc: float, best_val_acc: float, eta_str: str):
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        self.train_loss = round(train_loss, 4)
        self.train_acc = round(train_acc, 4)
        self.val_loss = round(val_loss, 4)
        self.val_acc = round(val_acc, 4)
        self.best_val_acc = round(best_val_acc, 4)
        self.eta_str = eta_str
        msg = {
            'type': 'metrics',
            'epoch': epoch,
            'total_epochs': total_epochs,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4),
            'best_val_acc': round(best_val_acc, 4),
            'eta': eta_str,
            'progress': (epoch + 1) / total_epochs * 100
        }
        self.log_queue.put(msg)
        self.log_buffer.append(msg)

    def emit_complete(self, metrics: dict):
        msg = {'type': 'complete', 'metrics': metrics}
        self.log_queue.put(msg)
        self.log_buffer.append(msg)

    def emit_error(self, message: str):
        msg = {'type': 'error', 'message': message}
        self.log_queue.put(msg)
        self.log_buffer.append(msg)

    def get_state(self) -> dict:
        with self._log_buffer_lock:
            log_buffer_copy = list(self.log_buffer)
        return {
            'is_training': self._is_training,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'val_loss': self.val_loss,
            'val_acc': self.val_acc,
            'best_val_acc': self.best_val_acc,
            'eta': self.eta_str,
            'progress': ((self.current_epoch + 1) / self.total_epochs * 100) if self.total_epochs > 0 else 0,
            'log_buffer': log_buffer_copy,
        }

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def prepare_data(self, data_path: str, validation_split: float = 0.2, random_seed: int = 42,
                     sample_size: int = 50000):
        read_rows = int(sample_size * 1.5)
        self.log(f"Reading {read_rows:,} rows × {len(REQUIRED_COLUMNS)} columns from CSV...")
        try:
            df = pd.read_csv(
                data_path,
                usecols=REQUIRED_COLUMNS,
                dtype={'CHR_GRCh38': str},
                nrows=read_rows,
                low_memory=False,
            )
        except ValueError as e:
            raise RuntimeError(
                f"CSV columns don't match REQUIRED_COLUMNS. "
                f"Expected: {REQUIRED_COLUMNS}. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")
        df['CHR_GRCh38'] = pd.to_numeric(df['CHR_GRCh38'], errors='coerce').fillna(0)

        if len(df) == 0:
            raise ValueError("Dataset is empty")

        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_seed)
            self.log(f"Sampled {sample_size:,} rows (read {read_rows:,} total)")
        else:
            self.log(f"Loaded {len(df):,} variants")

        self.log("Creating classification labels...")
        sig_series = df['Clinical_Significance'].astype(str).str.lower()
        df['is_pathogenic'] = sig_series.str.contains(
            'pathogenic|cancer', regex=True, na=False
        ).astype(int)

        self.log("Preparing features via features.py...")
        self.label_encoders = {}
        X, feature_names = extract_features_from_df(df, label_encoders=self.label_encoders, fit_encoders=True)
        y = df['is_pathogenic'].values

        self.log(f"Features ({len(feature_names)}): {feature_names}")
        self.log(f"Feature shape: {X.shape}")
        self.log(f"Positive class: {int(y.sum())} ({100*y.mean():.1f}%)")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=random_seed, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.log(f"Training samples: {len(X_train_scaled):,}, Validation samples: {len(X_val_scaled):,}")

        X_train_t = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_t = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)

        self.input_dim = X_train_t.shape[1]
        return X_train_t, y_train_t, X_val_t, y_val_t

    def train(self, data_path: str, epochs: int = 5, batch_size: int = 64,
              learning_rate: float = 0.001, dropout: float = 0.3,
              validation_split: float = 0.2, hidden_dims: Optional[List[int]] = None,
              scheduler_type: str = "ReduceLROnPlateau", gradient_clip: float = 1.0,
              use_class_weights: bool = True, random_seed: int = 42,
              sample_size: int = 50000):
        self._is_training = True
        self._stop_requested = False
        self._training_event.set()
        self._stop_event.clear()
        self.epoch_history = []
        self.loss_history = []
        self.acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.chart_files = {}

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.log(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.log("Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device('cpu')
            self.log("Using CPU")

        self.log("Initializing training...")
        self.set_seed(random_seed)

        try:
            X_train, y_train, X_val, y_val = self.prepare_data(
                data_path, validation_split, random_seed, sample_size
            )

            self.model = ClinVarNet(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            ).to(self.device)

            pos_count = int(y_train.sum().item())
            neg_count = len(y_train) - pos_count
            if use_class_weights and pos_count > 0 and neg_count > 0:
                pos_weight = neg_count / pos_count
                self.criterion = nn.BCELoss(weight=torch.tensor([pos_weight], device=self.device))
                self.log(f"Class weights applied: positive weight = {pos_weight:.2f}")
            else:
                self.criterion = nn.BCELoss()

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

            if scheduler_type == "ReduceLROnPlateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='max', factor=0.5, patience=5
                )
            elif scheduler_type == "CosineAnnealing":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=epochs
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=10, gamma=0.5
                )

            self.log("=" * 60)
            self.log(f"Configuration:")
            self.log(f"  Epochs: {epochs}, Batch size: {batch_size}")
            self.log(f"  Learning rate: {learning_rate}, Dropout: {dropout}")
            self.log(f"  Hidden dims: {hidden_dims}")
            self.log(f"  Scheduler: {scheduler_type}, Grad clip: {gradient_clip}")
            self.log(f"  Class weights: {use_class_weights}, Seed: {random_seed}")
            self.gradient_clip = gradient_clip
            self.log("=" * 60)

            total_params = sum(p.numel() for p in self.model.parameters())
            self.log(f"Model parameters: {total_params:,}")
            self.start_time = time.time()

            best_val_acc = 0
            patience_counter = 0
            max_patience = 15
            epoch_times = []

            for epoch in range(epochs):
                if self._stop_requested:
                    self.log("Training stopped by user")
                    break

                epoch_start = time.time()

                train_loss, train_acc, val_loss, val_acc, auc_score, f1, precision, recall = \
                    self._train_epoch(X_train, y_train, X_val, y_val, batch_size)

                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)

                self.epoch_history.append(epoch + 1)
                self.loss_history.append(train_loss)
                self.acc_history.append(train_acc)
                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)

                eta_str = "-"
                if len(epoch_times) > 0:
                    avg_epoch_time = sum(epoch_times) / len(epoch_times)
                    remaining = epochs - (epoch + 1)
                    eta_seconds = avg_epoch_time * remaining
                    if eta_seconds > 3600:
                        eta_str = f"{eta_seconds/3600:.1f}h {eta_seconds%3600/60:.0f}m"
                    elif eta_seconds > 60:
                        eta_str = f"{eta_seconds/60:.1f}m {eta_seconds%60:.0f}s"
                    else:
                        eta_str = f"{eta_seconds:.0f}s"

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                self.emit_metrics(epoch, epochs, train_loss, train_acc, val_loss, val_acc,
                                  best_val_acc, eta_str)

                if scheduler_type == "ReduceLROnPlateau":
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

                if patience_counter >= max_patience:
                    self.log(f"Early stopping at epoch {epoch+1} (no improvement for {max_patience} epochs)")
                    break

            self.log(f"Final learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            self.model.eval()
            with torch.no_grad():
                y_pred_proba = self.model(X_val).cpu().numpy().flatten()
                y_pred = (y_pred_proba >= 0.5).astype(int)
                y_true = y_val.cpu().numpy().flatten()

                final_acc = accuracy_score(y_true, y_pred)
                final_auc = roc_auc_score(y_true, y_pred_proba)
                final_f1 = f1_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred)
                precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
                pr_auc = auc(recall_vals, precision_vals)

            self.final_accuracy = final_acc
            self.final_auc = final_auc
            self.final_f1 = final_f1
            self.final_pr_auc = pr_auc

            self.log("=" * 60)
            self.log("TRAINING COMPLETE")
            self.log(f"Final Accuracy: {final_acc:.4f}")
            self.log(f"Final AUC: {final_auc:.4f}")
            self.log(f"Final F1-Score: {final_f1:.4f}")
            self.log(f"PR-AUC: {pr_auc:.4f}")
            self.log(f"Best Validation Accuracy: {best_val_acc:.4f}")
            self.log(f"Confusion Matrix:\n{cm}")

            self._y_true = y_true
            self._y_pred = y_pred
            self._y_pred_proba = y_pred_proba
            self._learning_rates = []
            for param_group in self.optimizer.param_groups:
                self._learning_rates.append(param_group['lr'])

            model_id = _uuid.uuid4().hex[:12]
            chart_dir = self.output_dir if self.output_dir else os.path.join(os.path.dirname(data_path), 'output_clinvar')
            chart_files = self.save_charts(chart_dir, model_id)

            final_metrics = {
                'model_id': model_id,
                'accuracy': round(final_acc, 4),
                'auc': round(final_auc, 4),
                'f1': round(final_f1, 4),
                'pr_auc': round(pr_auc, 4),
                'best_val_acc': round(best_val_acc, 4),
                'confusion_matrix': cm.tolist(),
                'chart_files': chart_files,
            }

            save_config = {
                'data_path': data_path,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'dropout': dropout,
                'validation_split': validation_split,
                'hidden_dims': hidden_dims,
                'scheduler_type': scheduler_type,
                'gradient_clip': gradient_clip,
                'use_class_weights': use_class_weights,
                'random_seed': random_seed,
                'sample_size': sample_size,
            }
            save_dir = self.output_dir if self.output_dir else os.path.join(os.path.dirname(data_path), 'output_clinvar')
            self.save_checkpoint(save_dir, save_config, model_id)
            self.emit_complete(final_metrics)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                msg = "GPU out of memory. Try reducing batch size."
            else:
                msg = str(e)
            self.log(f"ERROR: {msg}")
            self.emit_error(msg)
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            self.emit_error(str(e))
        finally:
            self._is_training = False
            self._training_event.clear()

    def _train_epoch(self, X_train, y_train, X_val, y_val, batch_size):
        self.model.train()
        indices = torch.randperm(len(X_train))
        total_loss = 0
        num_batches = 0
        total_correct = 0
        total_samples = 0

        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self, 'gradient_clip', 1.0))
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            y_pred_binary = (y_pred >= 0.5).float()
            total_correct += (y_pred_binary == y_batch).sum().item()
            total_samples += len(y_batch)

        train_acc = total_correct / max(total_samples, 1)

        self.model.eval()
        with torch.no_grad():
            y_val_pred = self.model(X_val)
            val_loss = self.criterion(y_val_pred, y_val).item()

            y_pred_binary_val = (y_val_pred.cpu().numpy() >= 0.5).astype(int).flatten()
            y_true = y_val.cpu().numpy().flatten()

            val_acc = accuracy_score(y_true, y_pred_binary_val)
            try:
                auc_score = roc_auc_score(y_true, y_val_pred.cpu().numpy().flatten())
            except Exception:
                auc_score = 0.5
            f1 = f1_score(y_true, y_pred_binary_val)
            precision = precision_score(y_true, y_pred_binary_val, zero_division=0)
            recall = recall_score(y_true, y_pred_binary_val, zero_division=0)

        avg_train_loss = total_loss / max(num_batches, 1)
        return avg_train_loss, train_acc, val_loss, val_acc, auc_score, f1, precision, recall

    def save_charts(self, save_dir: str, model_id: str):
        chart_dir = os.path.join(save_dir, 'charts')
        os.makedirs(chart_dir, exist_ok=True)
        self.chart_dir = chart_dir

        epochs = np.array(self.epoch_history)
        y_true = np.array(self._y_true)
        y_pred = np.array(self._y_pred)
        y_proba = np.array(self._y_pred_proba)

        # 1. Training Curves (loss + accuracy)
        fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(epochs, self.loss_history, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.val_loss_history, 'b--', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, self.acc_history, 'g-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, self.val_acc_history, 'g--', label='Val Acc', linewidth=2)
        axes[1].axhline(y=max(self.val_acc_history), color='orange', linestyle=':', label=f"Best Val {max(self.val_acc_history):.3f}")
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        fig1.suptitle(f'Training History — {model_id}', fontsize=14, y=1.02)
        fig1.tight_layout()
        curves_path = os.path.join(chart_dir, f'curves_{model_id}.png')
        fig1.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.close(fig1)
        self.chart_files['curves'] = f'curves_{model_id}.png'

        # 2. Confusion Matrix
        fig2, ax_cm = plt.subplots(figsize=(5, 4.5))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        labels_disp = ['Benign', 'Pathogenic']
        im = ax_cm.imshow(cm, cmap='Blues', interpolation='nearest')
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(labels_disp)
        ax_cm.set_yticklabels(labels_disp)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center',
                           fontsize=16, fontweight='bold',
                           color='white' if cm[i, j] > cm.max()/2 else 'black')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        ax_cm.set_title('Confusion Matrix (Validation Set)')
        fig2.tight_layout()
        cm_path = os.path.join(chart_dir, f'cm_{model_id}.png')
        fig2.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        self.chart_files['cm'] = f'cm_{model_id}.png'

        # 3. Class Distribution
        fig3, ax_dist = plt.subplots(figsize=(6, 4))
        counts = [int((y_true == 0).sum()), int((y_true == 1).sum())]
        colors = ['#4CAF50', '#f44336']
        bars = ax_dist.bar(labels_disp, counts, color=colors, edgecolor='white', width=0.5)
        for bar, count in zip(bars, counts):
            ax_dist.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                         f'{count:,}', ha='center', va='bottom', fontweight='bold')
        ax_dist.set_title('Class Distribution (Validation Set)')
        ax_dist.set_ylabel('Count')
        ax_dist.grid(axis='y', alpha=0.3)
        fig3.tight_layout()
        dist_path = os.path.join(chart_dir, f'distribution_{model_id}.png')
        fig3.savefig(dist_path, dpi=150, bbox_inches='tight')
        plt.close(fig3)
        self.chart_files['distribution'] = f'distribution_{model_id}.png'

        # 4. ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = getattr(self, 'final_auc', auc(fpr, tpr))

        fig4, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
        ax_roc.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random Classifier')
        ax_roc.fill_between(fpr, tpr, alpha=0.15, color='blue')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(alpha=0.3)
        ax_roc.set_xlim(-0.02, 1.02)
        ax_roc.set_ylim(-0.02, 1.02)
        fig4.tight_layout()
        roc_path = os.path.join(chart_dir, f'roc_{model_id}.png')
        fig4.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close(fig4)
        self.chart_files['roc'] = f'roc_{model_id}.png'

        # 5. Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        pr_auc_val = getattr(self, 'final_pr_auc', auc(recall_vals, precision_vals))

        fig5, ax_pr = plt.subplots(figsize=(6, 5))
        ax_pr.plot(recall_vals, precision_vals, 'b-', linewidth=2, label=f'PR (AUC = {pr_auc_val:.4f})')
        baseline = y_true.mean()
        ax_pr.axhline(y=baseline, color='r', linestyle='--', alpha=0.5,
                      label=f'Baseline ({baseline:.3f})')
        ax_pr.fill_between(recall_vals, precision_vals, alpha=0.15, color='blue')
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curve')
        ax_pr.legend(loc='upper right')
        ax_pr.grid(alpha=0.3)
        fig5.tight_layout()
        pr_path = os.path.join(chart_dir, f'pr_{model_id}.png')
        fig5.savefig(pr_path, dpi=150, bbox_inches='tight')
        plt.close(fig5)
        self.chart_files['pr'] = f'pr_{model_id}.png'

        # 6. Prediction Probability Distribution
        fig6, (ax_hist0, ax_hist1) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, (label, mask), color in zip(
            [ax_hist0, ax_hist1],
            [('Benign', y_true == 0), ('Pathogenic', y_true == 1)],
            ['#4CAF50', '#f44336']
        ):
            probs = y_proba[mask]
            if len(probs) > 0:
                ax.hist(probs, bins=30, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
                ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Threshold=0.5')
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Count')
            ax.set_title(f'{label} (n={len(probs)})')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        fig6.suptitle('Prediction Probability Distribution', fontsize=13)
        fig6.tight_layout()
        prob_path = os.path.join(chart_dir, f'probability_dist_{model_id}.png')
        fig6.savefig(prob_path, dpi=150, bbox_inches='tight')
        plt.close(fig6)
        self.chart_files['probability_dist'] = f'probability_dist_{model_id}.png'

        self.log(f"Charts saved to: {chart_dir}")
        return self.chart_files

    def save_checkpoint(self, save_dir: str, config: dict, model_id: str = '') -> str:
        if self.model is None:
            raise ValueError("No model to save")

        if not model_id:
            model_id = _uuid.uuid4().hex[:12]
        model_path = os.path.join(save_dir, f"model_{model_id}.pt")

        final_metrics = {
            'accuracy': round(getattr(self, 'final_accuracy', 0), 4),
            'auc': round(getattr(self, 'final_auc', 0), 4),
            'f1': round(getattr(self, 'final_f1', 0), 4),
            'pr_auc': round(getattr(self, 'final_pr_auc', 0), 4),
        }
        if self.val_acc_history:
            final_metrics['best_val_acc'] = round(max(self.val_acc_history), 4)

        full_config = {
            **config,
            'model_id': model_id,
            'final_metrics': final_metrics,
        }

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'input_dim': self.input_dim,
            'config': full_config,
        }, model_path)

        config_path = os.path.join(save_dir, f"config_{model_id}.json")
        with open(config_path, 'w') as f:
            json.dump(full_config, f, indent=2)

        self.log(f"Model saved to: {model_path}")
        return model_path
