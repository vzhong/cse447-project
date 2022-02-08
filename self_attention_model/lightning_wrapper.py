import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


class LightningWrapper(pl.LightningModule):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

    def training_step(self, batch: torch.Tensor, _):
        true = self.f.embed.one_hot(batch)[:, 1:, :]
        pred = F.log_softmax(self.f(batch)[:, :-1, :], dim=-1)
        loss = -(true * pred).mean()

        self.log("train_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters(), lr=1.0)
        return optimizer