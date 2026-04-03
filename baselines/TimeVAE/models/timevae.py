import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

from models.vae_base import BaseVariationalAutoencoder, Sampling


class TrendLayer(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, trend_poly):
        super(TrendLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.trend_dense1 = nn.Linear(self.latent_dim, self.feat_dim * self.trend_poly)
        self.trend_dense2 = nn.Linear(self.feat_dim * self.trend_poly, self.feat_dim * self.trend_poly)

    def forward(self, z):
        # (b, l, dim)
        trend_params = F.relu(self.trend_dense1(z))
        trend_params = self.trend_dense2(trend_params)
        trend_params = trend_params.view(-1, self.feat_dim, self.trend_poly)

        lin_space = torch.arange(0, float(self.seq_len), 1, device=z.device) / self.seq_len 
        poly_space = torch.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], dim=0) 

        trend_vals = torch.matmul(trend_params, poly_space) 
        trend_vals = trend_vals.permute(0, 2, 1) 
        return trend_vals


class SeasonalLayer(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, custom_seas):
        super(SeasonalLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.custom_seas = custom_seas

        self.dense_layers = nn.ModuleList([
            nn.Linear(latent_dim, feat_dim * num_seasons)
            for num_seasons, len_per_season in custom_seas
        ])
        

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = torch.arange(num_seasons).unsqueeze(1) + torch.zeros(
            (num_seasons, len_per_season), dtype=torch.int32
        )
        season_indexes = season_indexes.view(-1)
        season_indexes = season_indexes.repeat(self.seq_len // len_per_season + 1)[: self.seq_len]
        return season_indexes

    def forward(self, z):
        N = z.shape[0]
        ones_tensor = torch.ones((N, self.feat_dim, self.seq_len), dtype=torch.int32, device=z.device)

        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)
            season_params = season_params.view(-1, self.feat_dim, num_seasons)

            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            ).to(z.device)

            dim2_idxes = ones_tensor * season_indexes_over_time.view(1, 1, -1)
            season_vals = torch.gather(season_params, 2, dim2_idxes)

            all_seas_vals.append(season_vals)

        all_seas_vals = torch.stack(all_seas_vals, dim=-1) 
        all_seas_vals = torch.sum(all_seas_vals, dim=-1)  
        all_seas_vals = all_seas_vals.permute(0, 2, 1)  

        return all_seas_vals

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len, self.feat_dim)
    

class LevelModel(nn.Module):
    def __init__(self, latent_dim, feat_dim, seq_len):
        super(LevelModel, self).__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.level_dense1 = nn.Linear(self.latent_dim, self.feat_dim)
        self.level_dense2 = nn.Linear(self.feat_dim, self.feat_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        level_params = self.relu(self.level_dense1(z))
        level_params = self.level_dense2(level_params)
        level_params = level_params.view(-1, 1, self.feat_dim)

        ones_tensor = torch.ones((1, self.seq_len, 1), dtype=torch.float32, device=z.device)
        level_vals = level_params * ones_tensor
        return level_vals


class ResidualConnection(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, encoder_last_dense_dim):
        super(ResidualConnection, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        
        self.dense = nn.Sequential(
            nn.Linear(latent_dim, encoder_last_dense_dim),
            nn.ReLU()
        )

        self.deconv_layers = nn.Sequential()
        in_channels = hidden_layer_sizes[-1]
        
        for i, num_filters in enumerate(reversed(hidden_layer_sizes[:-1])):
            self.deconv_layers.append(
                nn.ConvTranspose1d(in_channels, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            if i < len(hidden_layer_sizes) - 1:
                self.deconv_layers.append(nn.ReLU())
            in_channels = num_filters
            
        self.deconv_layers.append(
            nn.ConvTranspose1d(in_channels, feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        )


    def forward(self, z):
        x = self.dense(z)
        x = x.permute(0, 2, 1)
        x = self.deconv_layers(x)
        x = x.permute(0, 2, 1)
        return x


class CNN1DUpsample(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()

        self.in_proj = nn.Conv1d(dim, hidden_dim, kernel_size=3, padding=1)

        self.block1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        self.out_proj = nn.Conv1d(hidden_dim, dim, kernel_size=3, padding=1)

    def upsample(self, x):
        # x: (B, C, T)
        return F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)

    def forward(self, x):
        # x: (B, T, D)

        x = x.transpose(1, 2)  # → (B, D, T)

        x = self.in_proj(x)

        # ×2
        x = self.upsample(x)
        x = self.block1(x)

        # ×4
        x = self.upsample(x)
        x = self.block2(x)

        # ×8
        x = self.upsample(x)
        x = self.block3(x)

        x = self.out_proj(x)

        x = x.transpose(1, 2)  # → (B, 8T, D)

        return x


class TimeVAEEncoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim):
        super(TimeVAEEncoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = []
        self.layers.append(nn.Conv1d(feat_dim, hidden_layer_sizes[0], kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU())

        for i, num_filters in enumerate(hidden_layer_sizes[1:]):
            self.layers.append(nn.Conv1d(hidden_layer_sizes[i], num_filters, kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*self.layers)
        self.z_mean = nn.Linear(self.hidden_layer_sizes[-1], latent_dim)
        self.z_log_var = nn.Linear(self.hidden_layer_sizes[-1], latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1,2)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)

        epsilon = torch.randn_like(z_mean).to(z_mean.device)
        z =  z_mean + torch.exp(0.5 * z_log_var) * epsilon

        return z_mean, z_log_var, z


class TimeVAEDecoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, trend_poly=0, custom_seas=None, use_residual_conn=True, encoder_last_dense_dim=None):
        super(TimeVAEDecoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.encoder_last_dense_dim = encoder_last_dense_dim
        self.level_model = LevelModel(self.latent_dim * (seq_len//8), self.feat_dim, self.seq_len)

        self.trend_layer = None

        if trend_poly is not None and trend_poly > 0:
            self.trend_layer = TrendLayer(seq_len, feat_dim, latent_dim * (seq_len//8), trend_poly)

        self.seasonal_layer = None
        if custom_seas is not None and len(custom_seas) > 0:
            self.seasonal_layer = SeasonalLayer(seq_len, feat_dim, latent_dim, custom_seas)

        if use_residual_conn:
            self.residual_conn = ResidualConnection(seq_len, feat_dim, hidden_layer_sizes, latent_dim, encoder_last_dense_dim)

    def forward(self, z):
        # print(f"z.shape = {z.shape}")
        batch_size = z.shape[0] # (batch_size, seq_len//8, dim)
        outputs = self.level_model(z.reshape(batch_size, -1))
        if self.trend_layer is not None:
            outputs += self.trend_layer(z.reshape(batch_size, -1))

        if self.seasonal_layer is not None:
            outputs += self.seasonal_layer(z)

        if self.use_residual_conn:
            residuals = self.residual_conn(z)
            outputs += residuals

        return outputs


class TimeVAE(BaseVariationalAutoencoder):
    model_name = "TimeVAE"

    def __init__(
        self,
        hidden_layer_sizes=None,
        trend_poly=0,
        custom_seas=None,
        use_residual_conn=True,
        **kwargs,
    ):
        super(TimeVAE, self).__init__(**kwargs)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _get_encoder(self):
        return TimeVAEEncoder(self.seq_len, self.feat_dim, self.hidden_layer_sizes, self.latent_dim)

    def _get_decoder(self):
        return TimeVAEDecoder(self.seq_len, self.feat_dim, self.hidden_layer_sizes, self.latent_dim, self.trend_poly, self.custom_seas, self.use_residual_conn, self.encoder.hidden_layer_sizes[-1])

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, f"{self.model_name}_weights.pth"))

        if self.custom_seas is not None:
            self.custom_seas = [(int(num_seasons), int(len_per_season)) for num_seasons, len_per_season in self.custom_seas]

        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "kl_wt": self.kl_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "trend_poly": self.trend_poly,
            "custom_seas": self.custom_seas,
            "use_residual_conn": self.use_residual_conn,
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)

    @classmethod
    def load(cls, model_dir: str) -> "TimeVAE":
        params_file = os.path.join(model_dir, f"{cls.model_name}_parameters.pkl")
        dict_params = joblib.load(params_file)
        vae_model = TimeVAE(**dict_params)
        vae_model.load_state_dict(torch.load(os.path.join(model_dir, f"{cls.model_name}_weights.pth")))
        return vae_model