"""Models tested in our repository over the features and signals, to predict the future returns of CH.DVD (log return end of the day)
   and tested over the baseline of buy and hold strategy and some simple signal metric.
   Every model derives its tading strategy based on the predicted return over the NEXT day. 
   If log return is positive, we go long, if negative we go short, if zero we do nothing.

Architectures:

    LinearBaseline   - classic predictive regression (OLS, with ridge lasso regression (L1), scikit-learn)
    XGBoostBaseline  - gradient-boosted trees (xgboost)
    MaskedVAE        - masked variational auto-encoder (PyTorch). Both MLP and Transformer encoder variants are supported via config.


The Masked-VAE idea: we append the target return as one extra dimension to the
feature vector, randomly mask features (with some probability) + the target (always) during training
and ask the model to reconstruct everything. At inference the target token is always masked,
so its reconstruction is the forecast. A mask-indicator channel makes the model
robust to missing inputs (exactly the resilience the project asks for).


The loss are explained into each model architecture.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from utils.config import VAEConfig, XGBConfig
from utils.config import set_seed


# --------------------------------------------------------------------------- #
# Classic baselines
# --------------------------------------------------------------------------- #
class LinearBaseline:
    """Lasso predictive regression, L1 regularisation with time-series Cross validation

    Alpha is selected via 5-fold TimeSeriesSplit cross-validation on the
    training data (no future leakage).  Lasso shrinks irrelevant coefficients
    to exactly zero, which is appropriate for high-dimensional financial features.
    """

    def __init__(self):
        self.model = LassoCV(
            cv=TimeSeriesSplit(n_splits=5),
            max_iter=5000,
            random_state=0,
        )

    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        if verbose:
            print(f"  Architecture : Linear (Lasso, alpha via TimeSeriesSplit-5 CV)")
            print(f"  Features     : {X.shape[1]}")
            print(f"  Loss         : MSE + α·||β||₁  (L1 regularisation)")
        self.model.fit(X, y)
        if verbose:
            print(f"  Selected α   : {self.model.alpha_:.6f}")
            n_nonzero = (self.model.coef_ != 0).sum()
            print(f"  Non-zero coef: {n_nonzero} / {X.shape[1]}")
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "LinearBaseline":
        import joblib
        obj = cls()
        obj.model = joblib.load(path)
        return obj


class XGBoostBaseline:
    """Gradient-boosted trees (deterministic via fixed seed)."""

    def __init__(self, cfg: XGBConfig | None = None, seed: int = 42):
        self.cfg = cfg = cfg or XGBConfig()
        self.model = XGBRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            early_stopping_rounds=cfg.early_stopping_rounds,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=4,
        )

    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        if verbose:
            c = self.cfg
            print(f"  Architecture : XGBoost (gradient-boosted trees)")
            print(f"  Features     : {X.shape[1]}")
            print(f"  Config       : trees={c.n_estimators}  depth={c.max_depth}  "
                  f"lr={c.learning_rate}  subsample={c.subsample}  λ={c.reg_lambda}")
            print(f"  Loss         : MSE (reg:squarederror)")
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X, y, eval_set=eval_set, verbose=50 if verbose else False)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str) -> None:
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str) -> "XGBoostBaseline":
        obj = cls()
        obj.model.load_model(path)
        return obj


# --------------------------------------------------------------------------- #
# Masked Variational Auto-Encoder
# --------------------------------------------------------------------------- #
class _Encoder(nn.Module):
    """MLP or Transformer encoder producing (mu, logvar)."""

    def __init__(self, in_dim: int, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_attention:
            # Treat each (value, mask) pair as a token; attend across features.
            self.d_model = 32
            self.token_embed = nn.Linear(2, self.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=4, dim_feedforward=64,
                dropout=cfg.dropout, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=2)
            enc_out = self.d_model
            self.n_tokens = in_dim  # in_dim here = augmented feature count
        else:
            dims = [in_dim * 2] + list(cfg.hidden_dims)  # *2: value + mask channel
            layers: list[nn.Module] = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.ReLU(), nn.Dropout(cfg.dropout)]
            self.mlp = nn.Sequential(*layers)
            enc_out = dims[-1]
        self.fc_mu = nn.Linear(enc_out, cfg.latent_dim)
        self.fc_logvar = nn.Linear(enc_out, cfg.latent_dim)

    def forward(self, x_masked: torch.Tensor, mask: torch.Tensor):
        if self.cfg.use_attention:
            tokens = torch.stack([x_masked, mask], dim=-1)  # (B, T, 2)
            h = self.transformer(self.token_embed(tokens))  # (B, T, d_model)
            h = h.mean(dim=1)                                # pool over tokens
        else:
            h = self.mlp(torch.cat([x_masked, mask], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    def __init__(self, out_dim: int, cfg: VAEConfig):
        super().__init__()
        dims = [cfg.latent_dim] + list(reversed(cfg.hidden_dims))
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.ReLU(), nn.Dropout(cfg.dropout)]
        layers += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class MaskedVAE:
    """Masked-VAE forecaster with a scikit-style fit/predict wrapper."""

    def __init__(self, n_features: int, cfg: VAEConfig | None = None,
                 seed: int = 42, target_weight: float = 5.0,
                 device: str | None = None):
        self.cfg = cfg or VAEConfig()
        self.seed = seed
        self.target_weight = target_weight  # upweight target slot in recon loss
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.aug_dim = n_features + 1        # features + target token
        self.target_idx = n_features         # target is the last slot
        set_seed(seed)
        self.encoder = _Encoder(self.aug_dim, self.cfg).to(self.device)
        self.decoder = _Decoder(self.aug_dim, self.cfg).to(self.device)

    # -- internals --------------------------------------------------------- #
    def _augment(self, X: np.ndarray, y: np.ndarray | None) -> torch.Tensor:
        X = torch.as_tensor(X, dtype=torch.float32)
        if y is None:
            y = torch.zeros(len(X), 1)
        else:
            y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
        return torch.cat([X, y], dim=1).to(self.device)

    def _sample_mask(self, z: torch.Tensor, train: bool) -> torch.Tensor:
        """1 = masked/hidden. Target always masked; features masked in training."""
        mask = torch.zeros_like(z)
        mask[:, self.target_idx] = 1.0
        if train and self.cfg.mask_prob > 0:
            feat = torch.rand(z.size(0), self.aug_dim - 1, device=z.device) < self.cfg.mask_prob
            mask[:, : self.target_idx] = feat.float()
        return mask

    def _forward(self, z, mask, sample: bool):
        x_masked = z * (1.0 - mask)  # masked entries -> 0 (standardised mean)
        mu, logvar = self.encoder(x_masked, mask)
        if sample:
            std = torch.exp(0.5 * logvar)
            latent = mu + std * torch.randn_like(std)
        else:
            latent = mu
        recon = self.decoder(latent)
        return recon, mu, logvar

    def _loss(self, recon, z, mask, mu, logvar):
        # Weight the reconstruction of the (masked) target slot more heavily.
        w = torch.ones(self.aug_dim, device=z.device)
        w[self.target_idx] = self.target_weight
        recon_err = (((recon - z) ** 2) * w).mean()
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_err + self.cfg.beta * kl, recon_err

    # -- public API -------------------------------------------------------- #
    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        if verbose:
            enc_p = sum(p.numel() for p in self.encoder.parameters())
            dec_p = sum(p.numel() for p in self.decoder.parameters())
            attn = "TransformerEncoder" if self.cfg.use_attention else "MLP"
            print(f"  Architecture : MaskedVAE  encoder={attn}")
            print(f"  Features     : {self.aug_dim - 1} input  +  1 target token  =  {self.aug_dim} aug dim")
            print(f"  Encoder      : hidden={self.cfg.hidden_dims}  latent={self.cfg.latent_dim}  params={enc_p:,}")
            print(f"  Decoder      : hidden={tuple(reversed(self.cfg.hidden_dims))}  params={dec_p:,}")
            print(f"  Total params : {enc_p + dec_p:,}")
            print(f"  Loss         : MSE_recon (target_weight={self.target_weight})  +  β={self.cfg.beta} · KL")
            print(f"  Regularise   : mask_prob={self.cfg.mask_prob}  dropout={self.cfg.dropout}  wd={self.cfg.weight_decay}")
            print(f"  Training     : epochs={self.cfg.epochs}  batch={self.cfg.batch_size} samples  "
                  f"lr={self.cfg.lr}  patience={self.cfg.patience} epochs")
            print(f"  Device       : {self.device}")
        set_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        z = self._augment(X, y)
        ds = torch.utils.data.TensorDataset(z)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=True, generator=g
        )
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.Adam(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        best_val, best_state, wait = float("inf"), None, 0
        stopped_epoch = self.cfg.epochs
        for epoch in range(1, self.cfg.epochs + 1):
            self.encoder.train(); self.decoder.train()
            train_loss = 0.0
            for (zb,) in loader:
                mask = self._sample_mask(zb, train=True)
                recon, mu, logvar = self._forward(zb, mask, sample=True)
                loss, _ = self._loss(recon, zb, mask, mu, logvar)
                opt.zero_grad(); loss.backward(); opt.step()
                train_loss += loss.item()
            train_loss /= len(loader)

            val_loss = float("nan")
            if X_val is not None:
                vpred = self.predict(X_val)
                val_loss = float(np.mean((vpred - np.asarray(y_val)) ** 2))
                if val_loss < best_val - 1e-6:
                    best_val, wait = val_loss, 0
                    best_state = ({k: v.clone() for k, v in self.encoder.state_dict().items()},
                                  {k: v.clone() for k, v in self.decoder.state_dict().items()})
                else:
                    wait += 1
                    if wait >= self.cfg.patience:
                        stopped_epoch = epoch
                        if verbose:
                            print(f"  Early stop at epoch {epoch}/{self.cfg.epochs} "
                                  f"(best val_loss={best_val:.4f} [MSE, scaled])")
                        break

            if verbose and (epoch % 10 == 0 or epoch == 1):
                val_str = f"{val_loss:.4f}" if not np.isnan(val_loss) else "—"
                print(f"  Epoch {epoch:>4}/{self.cfg.epochs} | "
                      f"train_loss={train_loss:.4f} [MSE, scaled] | val_loss={val_str} [MSE, scaled]")

        if best_state is not None:
            self.encoder.load_state_dict(best_state[0])
            self.decoder.load_state_dict(best_state[1])
            if verbose:
                print(f"  Restored best weights (val_loss={best_val:.4f})")
        return self

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        """Forecast = reconstructed target token (features observed, target masked)."""
        self.encoder.eval(); self.decoder.eval()
        z = self._augment(X, y=None)
        mask = self._sample_mask(z, train=False)  # only target masked
        recon, _, _ = self._forward(z, mask, sample=False)
        return recon[:, self.target_idx].cpu().numpy()

    def save(self, path: str) -> None:
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "cfg": self.cfg,
            "aug_dim": self.aug_dim,
            "target_idx": self.target_idx,
            "seed": self.seed,
            "target_weight": self.target_weight,
        }, path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "MaskedVAE":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        n_features = ckpt["target_idx"]  # target_idx == n_features
        obj = cls(
            n_features=n_features,
            cfg=ckpt["cfg"],
            seed=ckpt["seed"],
            target_weight=ckpt["target_weight"],
            device=device,
        )
        obj.encoder.load_state_dict(ckpt["encoder"])
        obj.decoder.load_state_dict(ckpt["decoder"])
        return obj
