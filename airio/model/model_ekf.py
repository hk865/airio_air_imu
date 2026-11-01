import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==================== Base Modules ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_sizes, dilations, strides, dropout=0.5):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == len(dilations) == len(strides)

        layers = []
        prev_c = in_channels
        for c, k, d, s in zip(channels, kernel_sizes, dilations, strides):
            padding = 3
            layers.append(nn.Conv1d(prev_c, c, k, stride=s, padding=padding, dilation=d))
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.GELU())
            prev_c = c

        layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Input: (B, T, C) -> Convert to (B, C, T)
        x = x.transpose(1, 2)
        x = self.net(x)
        # Output: (B, C_out, T_out) -> Convert to (B, T_out, C_out)
        x = x.transpose(1, 2)
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2,
                 temporal_type='gru', bidirectional=False):
        super().__init__()
        self.temporal_type = temporal_type.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if self.temporal_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0,
                              bidirectional=bidirectional)
        elif self.temporal_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0,
                               bidirectional=bidirectional)
        elif self.temporal_type == 'transformer':
            self.pos_encoder = PositionalEncoding(input_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=8, dim_feedforward=hidden_dim,
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        else:
            raise ValueError(f"Unsupported temporal_type: {temporal_type}")

    def forward(self, x):
        if self.temporal_type in ['gru', 'lstm']:
            out, _ = self.rnn(x)
            return out
        else:  # transformer
            x = self.pos_encoder(x)
            out = self.transformer(x)
            out = self.proj(out)
            return out


class TemporalAttention(nn.Module):
    """Temporal attention mechanism"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        weighted = torch.sum(x * attention_weights, dim=1)
        return weighted


# ==================== AirIO Compatible Models ====================

class EkfMotionNetwork(nn.Module):
    """
    AirIO compatible EKF motion network with CNN + GRU architecture
    """
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.logvar_offset = getattr(conf, 'logvar_offset', 5.0)

        # IMU encoder config - flattened parameters
        self.kernel_sizes = list(getattr(conf, 'kernel_sizes', [7, 7]))
        self.strides = list(getattr(conf, 'strides', [3, 3]))
        self.dilations = list(getattr(conf, 'dilations', [1, 1]))
        self.channels = list(getattr(conf, 'imu_channels', [32, 64]))
        imu_dropout = getattr(conf, 'imu_dropout', 0.5)

        self.imu_encoder = CNNEncoder(
            in_channels=6,
            channels=self.channels,
            kernel_sizes=self.kernel_sizes,
            dilations=self.dilations,
            strides=self.strides,
            dropout=imu_dropout
        )

        # GRU layers
        self.gru1 = nn.GRU(64, 64, 1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(128, 128, 1, batch_first=True, bidirectional=True)

        # Output heads
        self.veldecoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        self.velcov_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )

        # Total stride calculation
        self.total_stride = 1
        for s in self.strides:
            self.total_stride *= s

    def encoder(self, x):
        """Encode IMU features"""
        x = self.imu_encoder(x)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x

    def decoder(self, x):
        """Decode velocity"""
        vel = self.veldecoder(x)
        return vel

    def cov_decoder(self, x):
        """Decode covariance"""
        cov = torch.exp(self.velcov_decoder(x) - self.logvar_offset)
        return cov

    def get_label(self, gt_label):
        """Align ground truth labels to network output"""
        padding_list = [3, 3]
        k_list = self.kernel_sizes
        s_list = self.strides

        s_idx = (k_list[0] - padding_list[0]) + s_list[0] * (k_list[1] - 1 - padding_list[1]) + 1
        step = s_list[0] * s_list[1]

        select_label = gt_label[:, s_idx::step, :]

        L1 = (gt_label.shape[1] + 2 * padding_list[0] - k_list[0]) // s_list[0] + 1
        L_out = (L1 + 2 * padding_list[1] - k_list[1]) // s_list[1] + 1

        diff = L_out - select_label.shape[1]

        if diff > 0:
            last = gt_label[:, -1:, :].repeat(1, diff, 1)
            select_label = torch.cat((select_label, last), dim=1)
        elif diff < 0:
            select_label = select_label[:, :L_out, :]

        return select_label

    def forward(self, data, rot=None):
        """
        Forward pass compatible with AirIO
        """
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)
        encoded = self.encoder(feature)
        net_vel = self.decoder(encoded)

        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(encoded)
        return {"cov": cov, "net_vel": net_vel}


class EkfMotionNetworkwithRot(EkfMotionNetwork):
    """
    AirIO compatible EKF motion network with rotation encoding
    """
    def __init__(self, conf):
        super().__init__(conf)

        # Attitude encoder - flattened parameters
        att_kernel_sizes = list(getattr(conf, 'att_kernel_sizes', [7, 7]))
        att_strides = list(getattr(conf, 'att_strides', [3, 3]))
        att_channels = list(getattr(conf, 'att_channels', [32, 64]))
        att_dropout = getattr(conf, 'att_dropout', 0.5)

        self.att_encoder = CNNEncoder(
            in_channels=3,
            channels=att_channels,
            kernel_sizes=att_kernel_sizes,
            dilations=[1, 1],
            strides=att_strides,
            dropout=att_dropout
        )

        # Fusion layers
        self.fusion_reduce_linear = nn.Linear(128, 64)
        self.fusion_reduce_bn = nn.BatchNorm1d(64)

        # Re-define GRUs for fused features
        self.gru1 = nn.GRU(64, 64, 1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(128, 128, 1, batch_first=True, bidirectional=True)

    def encoder(self, feature, ori):
        """Encode IMU features with orientation"""
        feat_imu = self.imu_encoder(feature)
        feat_att = self.att_encoder(ori)

        fused = torch.cat([feat_imu, feat_att], dim=-1)
        fused = self.fusion_reduce_linear(fused)
        fused = self.fusion_reduce_bn(fused.transpose(1, 2))
        fused = fused.transpose(1, 2)

        x, _ = self.gru1(fused)
        x, _ = self.gru2(x)
        return x

    def forward(self, data, rot=None):
        """
        Forward pass with rotation input
        """
        assert rot is not None, "rot input is required for EkfMotionNetworkwithRot"
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)
        encoded = self.encoder(feature, rot)
        net_vel = self.decoder(encoded)

        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(encoded)
        return {"cov": cov, "net_vel": net_vel}


class SimpleTransformerIMUModel(nn.Module):
    """
    AirIO compatible simplified Transformer-based IMU model
    """
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        input_dim = 6
        self.d_model = getattr(conf, 'd_model', 128)
        self.nhead = getattr(conf, 'nhead', 8)
        self.dim_feedforward = getattr(conf, 'dim_feedforward', 512)
        self.num_layers = getattr(conf, 'num_transformer_layers', 2)
        dropout = getattr(conf, 'dropout', 0.1)

        # Input projection
        self.input_projection = nn.Linear(input_dim, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Temporal attention
        self.temporal_attention = TemporalAttention(self.d_model)

        # Output heads
        self.veldecoder = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )
        self.velcov_decoder = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encoder(self, x):
        """Encode using Transformer"""
        # x: (B, T, 6)
        x = self.input_projection(x)  # (B, T, d_model)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Create causal mask
        seq_len = x.shape[1]
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)

        encoded = self.transformer_encoder(x, mask=mask)  # (B, T, d_model)
        return encoded

    def decoder(self, x):
        """Decode velocity"""
        # Use temporal attention to aggregate sequence
        aggregated = self.temporal_attention(x)  # (B, d_model)
        vel = self.veldecoder(aggregated)
        return vel.unsqueeze(1)  # (B, 1, 3)

    def cov_decoder(self, x):
        """Decode covariance"""
        aggregated = self.temporal_attention(x)
        logvar = self.velcov_decoder(aggregated)
        cov = torch.exp(logvar - 5.0)
        return cov.unsqueeze(1)  # (B, 1, 3)

    def get_label(self, gt_label):
        """Align ground truth labels - Transformer outputs single prediction per sequence"""
        # Take the mean of the sequence as the target
        target = gt_label.mean(dim=1, keepdim=True)  # (B, 1, 3)
        return target

    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, data, rot=None):
        """
        Forward pass compatible with AirIO
        """
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)
        encoded = self.encoder(feature)
        net_vel = self.decoder(encoded)

        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(encoded)
        return {"cov": cov, "net_vel": net_vel}
