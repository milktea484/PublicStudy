import torch.nn as nn
import torch
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import LinearLR
import pandas as pd
from metrics import contact_f1
from utils import mat2bp, bp2matrix, outer_concat
from tqdm import tqdm

class ResNet2DBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, bias=False):
        super().__init__()

        # Bottleneck architecture
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, bias=bias, padding="same"),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        residual = x

        x = self.conv_net(x)
        x = x + residual

        return x

class ResNet2D(nn.Module):
    def __init__(self, embed_dim, num_blocks, kernel_size=3, bias=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResNet2DBlock(embed_dim, kernel_size, bias=bias) for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

class SecondaryStructurePredictor(nn.Module):
    def __init__(
        self, embed_dim, num_blocks=2,
        conv_dim=64, kernel_size=3,
        #negative_weight=0.1,
        positive_weight=10,
        device='cpu', lr=1e-5
    ):
        super().__init__()
        self.lr = lr
        self.threshold = 0.1
        self.norm = nn.InstanceNorm2d(embed_dim)
        self.linear_in = nn.Linear(embed_dim, (int) (conv_dim/2))
        self.resnet = ResNet2D(conv_dim, num_blocks, kernel_size)
        self.conv_out = nn.Conv2d(conv_dim, 1, kernel_size=kernel_size, padding="same")
        self.device = device
        #self.class_weight = torch.tensor([negative_weight, 1.0]).float().to(self.device)
        self.positive_weight = positive_weight
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.lr_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=2000)

        self.to(device)

    def loss_func(self, yhat, y):
        """yhat and y are [N, M, M]"""

        binary_y = y.clone()
        binary_y[binary_y==-1] = 0
        #pos_weight = torch.full([binary_y.shape[0], binary_y.shape[1], binary_y.shape[2]], self.positive_weight).float().to(self.device)
        #loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_function = nn.BCEWithLogitsLoss()
        loss = loss_function(yhat, binary_y.float())

        return loss

    def forward(self, x):
        x = self.norm(x)
        x = self.linear_in(x) 

        x = outer_concat(x, x) 
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) 

        x = self.resnet(x)
        x = self.conv_out(x)
        x = x.squeeze(-3) 

        x = torch.triu(x, diagonal=1)
        x = x + x.transpose(-1, -2)

        return x.squeeze(-1)

    def fit(self, loader):
        self.train()
        loss_acum = 0
        f1_acum = 0
        for batch in tqdm(loader):
            X = batch["seq_embs_pad"].to(self.device)
            y = batch["contacts"].to(self.device)
            y_pred = self(X)
            # print(f"y_pred size: {y_pred.shape}") # torch.Size([4, 512, 512])
            # print(f"y size: {y.shape}") # torch.Size([4, 512, 512])
            loss = self.loss_func(y_pred, y)
            loss_acum += loss.item()
            # f1_acum += contact_f1(y.cpu(), y_pred.detach().cpu(), batch["Ls"], method="triangular")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.lr_scheduler.step()
        loss_acum /= len(loader)
        f1_acum /= len(loader)
        return {"loss": loss_acum, "f1": f1_acum}

    def test(self, loader):
        self.eval()
        loss_acum = 0
        y_prob = {} #device関連の小言いわれそう
        for batch in loader:
            seq_ids = batch["seq_ids"]
            X = batch["seq_embs_pad"].to(self.device)
            y = batch["contacts"].to(self.device)
            with torch.no_grad():
                y_pred = self(X)
                loss = self.loss_func(y_pred, y)
            loss_acum += loss.item()
            for k in range(len(y_pred)):
                y_prob[seq_ids[k]] = y_pred[k]

            # f1_acum += contact_f1(y.cpu(), y_pred.detach().cpu(), batch["Ls"], method="triangular")
        loss_acum /= len(loader)

        return {"loss": loss_acum, "y_prob": y_prob}

    def pred(self, loader, y_prob, y_ref):
        predictions = []
        y_truth = {}
        f1_acum = 0
        for batch in tqdm(loader):
            Ls = batch["Ls"]
            seq_ids = batch["seq_ids"]
            sequences = batch["sequences"]
            ys = batch["contacts"].to(self.device)
            f1_list = []

            for l, seq_id, sequence, y in zip(Ls, seq_ids, sequences, ys):
                ind = torch.where(y != -1)
                y = y[ind].view(l, l)
                y_truth[seq_id] = y
                y_prob[seq_id] = y_prob[seq_id][ind].view(l, l)
                y_ref[seq_id] = y_ref[seq_id][ind].view(l,l)

                y_prob[seq_id] = torch.sigmoid(y_prob[seq_id])
                y_ref[seq_id] = torch.sigmoid(y_ref[seq_id])

                pred_bp = mat2bp(y_prob[seq_id].cpu(), y_ref[seq_id].cpu())

                predictions.append((
                    seq_id,
                    sequence,
                    pred_bp                    
                ))
                f1_list.append(contact_f1(y.cpu(), bp2matrix(l, pred_bp).cpu().detach()))
            f1_acum += torch.tensor(f1_list).mean().item()

        predictions = pd.DataFrame(predictions, columns=["id", "sequence", "base_pairs"])
        f1_acum /= len(loader)

        return predictions, f1_acum, y_prob, y_truth
