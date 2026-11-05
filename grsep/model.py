import torch
from torch import nn


def block(cin, cout, bn=True, dropout=0.0):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, padding_mode="reflect", bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.LeakyReLU(0.1, inplace=True))
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class Head(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.body = nn.Sequential(block(width, width, dropout=dropout), block(width, width, dropout=dropout))
        self.out = nn.Conv2d(width, 1, 3, padding=1, padding_mode="reflect")

    def forward(self, x):
        return self.out(self.body(x))


class GroundRollSeparator(nn.Module):
    def __init__(self, width=64, trunk_layers=7, dropout=0.1):
        super().__init__()
        trunk = [block(2, width, bn=False)]
        trunk += [block(width, width) for _ in range(trunk_layers - 1)]
        self.trunk = nn.Sequential(*trunk)
        self.head_r = Head(width, dropout)
        self.head_g = Head(width, dropout)
        self.config = {"width": width, "trunk_layers": trunk_layers, "dropout": dropout}

    def forward(self, x):
        h = self.trunk(x)
        return self.head_r(h), self.head_g(h)

    @property
    def receptive_field(self):
        return 2 * (self.config["trunk_layers"] + 3) + 1


def save_model(model, path, band, extra=None):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config,
            "band": {"f_pass": band.f_pass, "f_stop": band.f_stop},
            "extra": extra or {},
        },
        path,
    )


def load_model(path, device="cpu"):
    from .bandsplit import Band

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GroundRollSeparator(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, Band(**ckpt["band"]), ckpt.get("extra", {})
