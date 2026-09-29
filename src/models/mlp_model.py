# MLP model script

import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import average_precision_score

class LightningMLPRegressor(pl.LightningModule):
    def __init__(self, input_dim=2214, hidden_dim=128, output_dim=1, learning_rate=1e-4, optimizer_name='adam'):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),     # Layer 0
            nn.ReLU(),                            # Layer 1
            nn.Dropout(p = 0.5), # droupout rate  # Layer 2
            nn.Linear(hidden_dim, output_dim)     # Layer 3
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = self.loss_fn(preds, y)
        self.log("valid_loss", loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = self.loss_fn(preds, y)
        self.log("test_loss", loss, logger=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

    def load_pretrained_weights(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))



class LightningMLPClassifier(pl.LightningModule):
    def __init__(self, input_dim=2214, hidden_dim=128, output_dim=1, learning_rate=1e-4, optimizer_name='adam'):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),     # Layer 0
            nn.ReLU(),                            # Layer 1
            nn.Dropout(p = 0.5), # droupout rate  # Layer 2
            nn.Linear(hidden_dim, output_dim)     # Layer 3
        )

        self.loss_fn = nn.BCEWithLogitsLoss() # Binary classification loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)                 # [B,1]
        logits = logits.view(-1, 1)      # Safety
        y = y.view(-1, 1).float()        # Safety
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits = logits.view(-1, 1)
        y = y.view(-1, 1).float()
        loss = self.loss_fn(logits, y)
        self.log("valid_loss", loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_targets = []
        self.test_losses = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits = logits.view(-1, 1)
        y = y.view(-1, 1).float()
        loss = self.loss_fn(logits, y)

        self.test_preds.append(logits.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        self.test_losses.append(loss.detach().cpu())
        return loss

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_preds, dim=0).view(-1)   # [N]
        all_y = torch.cat(self.test_targets, dim=0).view(-1)     # [N]

        probs = torch.sigmoid(all_preds)
        auprc = average_precision_score(all_y.numpy(), probs.numpy())
        baseline = all_y.float().mean().item()
        delta_auprc = auprc - baseline
        mean_loss = torch.stack(self.test_losses).mean().item()

        self.log("test_auprc", auprc)
        self.log("test_fraction_pos", baseline)
        self.log("test_delta_auprc", delta_auprc)
        self.log("test_loss", mean_loss)  

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

    def load_pretrained_weights(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))



class LightningMLPRegressorTransfer(pl.LightningModule):
    def __init__(self, input_dim=2214, hidden_dim=128, output_dim=1, learning_rate=1e-4, optimizer_name='adam'):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5) # droupout rate
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.backbone(x)
        return self.output_layer(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # This function is added for transfer learning
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = self.loss_fn(preds, y)
        self.log("valid_loss", loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = self.loss_fn(preds, y)
        self.log("test_loss", loss, logger=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Only optimize the parameters that require grad
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)


class LightningMLPClassifierTransfer(pl.LightningModule):
    def __init__(self, input_dim=2214, hidden_dim=128, output_dim=1, learning_rate=1e-4, optimizer_name='adam'):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5) # droupout rate
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.loss_fn = nn.BCEWithLogitsLoss() # Binary classification loss)

    def forward(self, x):
        x = self.backbone(x)
        return self.output_layer(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).view(-1, 1)      # [B,1]
        y = y.view(-1, 1).float()         # [B,1]
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).view(-1, 1)
        y = y.view(-1, 1).float()
        loss = self.loss_fn(logits, y)
        self.log("valid_loss", loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_targets = []
        self.test_losses = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).view(-1, 1)
        y = y.view(-1, 1).float()
        loss = self.loss_fn(logits, y)

        self.test_preds.append(logits.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        self.test_losses.append(loss.detach().cpu())
        return loss

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_preds, dim=0).view(-1)   # [N]
        all_y = torch.cat(self.test_targets, dim=0).view(-1)     # [N]

        probs = torch.sigmoid(all_preds)
        auprc = average_precision_score(all_y.numpy(), probs.numpy())
        baseline = all_y.float().mean().item()
        delta_auprc = auprc - baseline
        mean_loss = torch.stack(self.test_losses).mean().item()

        self.log("test_auprc", auprc)
        self.log("test_fraction_pos", baseline)
        self.log("test_delta_auprc", delta_auprc)
        self.log("test_loss", mean_loss)  

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

    def load_pretrained_weights(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))



class LightningMLPRegressorMultiTask(pl.LightningModule):
    def __init__(self, input_dim=2214, hidden_dim=128, output_dims=[1, 1], learning_rate=1e-4, optimizer_name='adam'):
        super().__init__()
        self.save_hyperparameters()

        # Shared backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, out_dim) for out_dim in output_dims
        ])

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        shared_features = self.shared_backbone(x)
        return [head(shared_features) for head in self.task_heads]

    def training_step(self, batch, batch_idx):
        total_loss = 0.0 # for backpropagation
        for dataset_key, (x, y) in batch.items():
            task_idx = 0 if dataset_key == "dataset0" else 1
            preds = self(x)[task_idx].squeeze()
            loss = self.loss_fn(preds, y)
            self.log(f"train_loss/{dataset_key}", loss, on_step=False, on_epoch=True, prog_bar=True)
            total_loss += loss
        return total_loss

    def validation_step(self, batch, batch_idx):
        for dataset_key, (x, y) in batch.items():
            task_idx = 0 if dataset_key == "dataset0" else 1
            preds = self(x)[task_idx].squeeze()
            loss = self.loss_fn(preds, y)
            self.log(f"val_loss/{dataset_key}", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        for dataset_key, (x, y) in batch.items():
            task_idx = 0 if dataset_key == "dataset0" else 1
            preds = self(x)[task_idx].squeeze()
            loss = self.loss_fn(preds, y)
            self.log(f"test_loss/{dataset_key}", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# Added more layers
class LightningMLPClassifierDeeper(pl.LightningModule):
    def __init__(self, input_dim=2214, hidden_dim=128, output_dim=1, learning_rate=1e-4, optimizer_name='adam'):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),     # Layer 0
            nn.ReLU(),                            # Layer 1
            nn.Dropout(p = 0.5), # droupout rate  # Layer 2
            nn.Linear(hidden_dim, hidden_dim),     # Layer 3
            nn.ReLU(),                            # Layer 4
            nn.Dropout(p = 0.5), # droupout rate  # Layer 5
            nn.Linear(hidden_dim, output_dim)     # Layer 6
        )

        self.loss_fn = nn.BCEWithLogitsLoss() # Binary classification loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = self.loss_fn(preds, y)
        self.log("valid_loss", loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_targets = []
        self.test_losses = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = self.loss_fn(preds, y)

        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        self.test_losses.append(loss.detach().cpu())

        return loss

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_preds)
        all_y = torch.cat(self.test_targets)

        probs = torch.sigmoid(all_preds)
        auprc = average_precision_score(all_y.numpy(), probs.numpy())
        baseline = all_y.float().mean().item()
        delta_auprc = auprc - baseline
        mean_loss = torch.stack(self.test_losses).mean().item()

        self.log("test_auprc", auprc)
        self.log("test_fraction_pos", baseline)
        self.log("test_delta_auprc", delta_auprc)
        self.log("test_loss", mean_loss)  

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

    def load_pretrained_weights(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))


# Added more layers
class LightningMLPClassifierTransferDeeper(pl.LightningModule):
    def __init__(self, input_dim=2214, hidden_dim=128, output_dim=1, learning_rate=1e-4, optimizer_name='adam'):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5), # droupout rate
            nn.Linear(hidden_dim, hidden_dim),     # Layer 3
            nn.ReLU(),                            # Layer 4
            nn.Dropout(p = 0.5), # droupout rate  # Layer 5
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.loss_fn = nn.BCEWithLogitsLoss() # Binary classification loss)

    def forward(self, x):
        x = self.backbone(x)
        return self.output_layer(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = self.loss_fn(preds, y)
        self.log("valid_loss", loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_targets = []
        self.test_losses = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = self.loss_fn(preds, y)

        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        self.test_losses.append(loss.detach().cpu())

        return loss

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_preds)
        all_y = torch.cat(self.test_targets)

        probs = torch.sigmoid(all_preds)
        auprc = average_precision_score(all_y.numpy(), probs.numpy())
        baseline = all_y.float().mean().item()
        delta_auprc = auprc - baseline
        mean_loss = torch.stack(self.test_losses).mean().item()

        self.log("test_auprc", auprc)
        self.log("test_fraction_pos", baseline)
        self.log("test_delta_auprc", delta_auprc)
        self.log("test_loss", mean_loss)  

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

    def load_pretrained_weights(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))


# Can adjust number of tasks
class LightningMLPClassifierMultitaskTransfer(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, lr, num_pretrain_tasks, with_target_head=True):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5)
        )
        self.pre_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_pretrain_tasks)])
        self.target_head = nn.Linear(hidden_dim, 1) if with_target_head else None
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def _forward_head(self, h, head_idx: int):
        if head_idx < len(self.pre_heads):
            return self.pre_heads[head_idx](h)
        elif self.target_head is not None and head_idx == len(self.pre_heads):
            return self.target_head(h)
        else:
            raise ValueError(f"Bad head idx {head_idx}")

    def forward(self, x, head_idx: int):
        h = self.backbone(x)
        return self._forward_head(h, head_idx)

    # ------------ Pretrain (multi-loader) ------------
    # CombinedLoader(dict) → batch는 dict: {"t0": (x0,y0), "t1": (x1,y1), ...}
    def _xy_from_batch(self, b):
    # b: (x,y) or (x,y,task_id)
        if isinstance(b, (list, tuple)):
            if len(b) == 2:
                x, y = b
            elif len(b) == 3:
                x, y, _ = b  
            else:
                raise ValueError(f"Unexpected batch format len={len(b)}")
        else:
            raise ValueError(f"Unexpected batch type: {type(b)}")
        return x, y

    def training_step(self, batch, batch_idx):
        losses = []
        for i, (_, b) in enumerate(batch.items()):
            x, y = self._xy_from_batch(b)
            logit = self.forward(x, head_idx=i).view(-1)
            loss  = self.loss_fn(logit, y.float().view(-1))
            self.log(f"train_loss/t{i}", loss, on_step=False, on_epoch=True, prog_bar=(i==0))
            losses.append(loss)
        return torch.stack(losses).mean()

    def validation_step(self, batch, batch_idx):
        losses = []
        for i, (_, b) in enumerate(batch.items()):
            x, y = self._xy_from_batch(b)
            logit = self.forward(x, head_idx=i).view(-1)
            loss  = self.loss_fn(logit, y.float().view(-1))
            self.log(f"valid_loss/t{i}", loss, on_step=False, on_epoch=True, prog_bar=(i==0))
            losses.append(loss)
        return torch.stack(losses).mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # ------------ Finetune/Test on TARGET ONLY ------------
    def finetune_step(self, batch, batch_idx):
        x, y = self._xy_from_batch(batch)
        target_idx = len(self.pre_heads)
        logit = self.forward(x, head_idx=target_idx).view(-1)
        loss  = self.loss_fn(logit, y.float().view(-1))
        self.log("train_loss/target", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validate_target_step(self, batch, batch_idx):
        x, y = self._xy_from_batch(batch)
        target_idx = len(self.pre_heads)
        logit = self.forward(x, head_idx=target_idx).view(-1)
        loss  = self.loss_fn(logit, y.float().view(-1))
        self.log("valid_loss/target", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"logit": logit.detach(), "y": y.detach(), "loss": loss.detach()}
        
    def validation_epoch_end(self, outputs):
        if outputs and isinstance(outputs[0], dict) and "logit" in outputs[0]:
            self._aggregate_target(outputs, prefix="valid")

    def test_epoch_end(self, outputs):
        if outputs and isinstance(outputs[0], dict) and "logit" in outputs[0]:
            self._aggregate_target(outputs, prefix="test")

    def _aggregate_target(self, outputs, prefix="valid"):
        logits = torch.cat([o["logit"] for o in outputs]).cpu()
        ys     = torch.cat([o["y"]    for o in outputs]).float().cpu()
        probs  = torch.sigmoid(logits).numpy()
        y_np   = ys.numpy()
        auprc  = average_precision_score(y_np, probs)
        baseline = float(ys.mean().item())
        self.log(f"{prefix}_auprc/target", auprc)
        self.log(f"{prefix}_frac_pos/target", baseline)
        self.log(f"{prefix}_delta_auprc/target", auprc - baseline)
