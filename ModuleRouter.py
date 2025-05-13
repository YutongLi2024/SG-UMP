import torch
import torch.nn as nn
import torch.nn.functional as F


class Gate(nn.Module):
    def __init__(self, input_dim, low_rank):
        super(Gate, self).__init__()
        self.down = nn.Linear(input_dim, input_dim // low_rank)
        self.up = nn.Linear(input_dim // low_rank, input_dim)

    def forward(self, x):
        x = self.down(x)
        x = torch.sigmoid(x)
        x = self.up(x)
        return x


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, low_rank=1):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, int(hidden_size // low_rank))  
        self.fc2 = nn.Linear(int(hidden_size // low_rank), output_size)
        self.relu = nn.GELU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    
class FilterLayer(nn.Module):
    def __init__(self, hidden_size, num_layers=1, dropout_prob=0.1, ratio=0.8):
        super(FilterLayer, self).__init__()
        self.num_layers = num_layers
        self.ratio = ratio
        self.filter_layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.LayerNorm(hidden_size, eps=1e-12)
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        batch, seq_len, hidden_size = x.shape

        for layer_idx in range(self.num_layers):
            # 1. 频域变换
            fft_x = torch.fft.rfft(x, dim=1, norm='ortho')  # shape: [B, F, D]

            # 2. 构造频率选择 mask
            freqs = torch.fft.rfftfreq(seq_len).to(x.device)  # shape: [F]
            mask = freqs < freqs.quantile(self.ratio).item()  # shape: [F]
            # shape: [F, D] -> broadcast to all dims
            mask = torch.outer(mask, torch.ones(hidden_size, device=x.device))  # [F, D]
            mask = mask.unsqueeze(0)  # [1, F, D]

            # 3. 应用 mask
            fft_x = fft_x * mask

            # 4. 反变换
            filtered_x = torch.fft.irfft(fft_x, n=seq_len, dim=1, norm='ortho')  # [B, T, D]

            # 5. Dropout + LayerNorm + 残差
            x = self.filter_layers[layer_idx](filtered_x) + x

        return x


class MultiScaleFusionLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        low_rank,
        num_shared_experts,
        num_specific_experts,
        experts_shared,
        experts_task1,
        experts_task2,
        experts_task3,
        dnn_share,
        dnn1,
        dnn2,
        dnn3,
        attention_layer
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.low_rank = low_rank
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts

        self.experts_shared = experts_shared
        self.experts_task1 = experts_task1
        self.experts_task2 = experts_task2
        self.experts_task3 = experts_task3

        self.dnn_share = dnn_share
        self.dnn1 = dnn1
        self.dnn2 = dnn2
        self.dnn3 = dnn3

        self.attention_layer = attention_layer

    def forward(self, id_feat, img_feat, txt_feat):
        """
        Args:
            id_feat, img_feat, txt_feat: [B, L, D]
        Returns:
            fused_output: [B, L, D]
        """
        x = torch.cat([id_feat, img_feat, txt_feat], axis=-1)

        experts_shared_o = [e(x) for e in self.experts_shared]
        experts_shared_o = torch.stack(experts_shared_o) # [num of share experts, batch, seq, feature_dim]

        # gate_share
        selected_s = self.dnn_share(x)
        gate_share_out = torch.einsum('abcd, bca -> bcd', experts_shared_o, selected_s)

        experts_task1_o = [e(id_feat + gate_share_out) for e in self.experts_task1] 
        experts_task1_o = torch.stack(experts_task1_o) # [num of specific experts, batch, seq, feature_dim]
        experts_task2_o = [e(img_feat + gate_share_out) for e in self.experts_task2]
        experts_task2_o = torch.stack(experts_task2_o)
        experts_task3_o = [e(txt_feat + gate_share_out) for e in self.experts_task3]
        experts_task3_o = torch.stack(experts_task3_o)


        # gate1
        selected1 = self.dnn1(id_feat)
        gate_1_out = torch.einsum('abcd, bca -> bcd', experts_task1_o, selected1)

        # gate2
        selected2 = self.dnn2(img_feat)
        gate_2_out = torch.einsum('abcd, bca -> bcd', experts_task2_o, selected2)

        # gate3
        selected3 = self.dnn3(txt_feat)
        gate_3_out = torch.einsum('abcd, bca -> bcd', experts_task3_o, selected3)

        # gather
        combined_gate_outputs = torch.cat([gate_1_out, gate_2_out, gate_3_out, gate_share_out], dim=-1)
        attention_scores = self.attention_layer(combined_gate_outputs)
        attention_scores = F.softmax(attention_scores, dim=-1)
        weighted_gate_1_out = gate_1_out[:, 0, :] * attention_scores[:, 0, :]
        weighted_gate_2_out = gate_2_out[:, 0, :] * attention_scores[:, 1, :]
        weighted_gate_3_out = gate_3_out[:, 0, :] * attention_scores[:, 2, :]
        weighted_gate_share_out = gate_share_out[:, 0, :] * attention_scores[:, 3, :]
        task_out = weighted_gate_1_out + weighted_gate_2_out + weighted_gate_3_out + weighted_gate_share_out
        task_out = task_out.unsqueeze(1).expand(-1, 100, -1)  # (batch_size, 100, feature_dim)

        
        return task_out


class HierarchicalAttentionLayer(nn.Module):
    def __init__(self, hidden_size, low_rank, gate_v, gate_t):
        super().__init__()
        self.gate_v = gate_v
        self.gate_t = gate_t

        self.low_rank_layer = nn.Linear(hidden_size, hidden_size // low_rank)
        self.low_rank_layer1 = nn.Linear(hidden_size // low_rank, hidden_size)

        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_size // low_rank,
            nhead=4,
            dim_feedforward=hidden_size // low_rank
        )

    def global_embeddings(self, x):
        x = self.low_rank_layer(x.transpose(0, 1))      # [L, B, R]
        x = self.transformer(x)                         # [L, B, R]
        x = self.low_rank_layer1(x.transpose(0, 1))     # [B, L, D]
        return x

    def forward(self, id_feat, img_feat, txt_feat):
        global_id = self.global_embeddings(id_feat)
        global_img = self.global_embeddings(img_feat)
        global_txt = self.global_embeddings(txt_feat)

        img_refined = (
            torch.multiply(img_feat, self.gate_v(id_feat)) +
            torch.multiply(global_img, self.gate_v(global_id))
        )
        txt_refined = (
            torch.multiply(txt_feat, self.gate_v(id_feat)) +
            torch.multiply(global_txt, self.gate_t(global_id))
        )

        out = id_feat + img_refined + txt_refined
        return out


class MutualInformationLoss(nn.Module):
    """
    Compute mutual information loss to encourage diverse routing decisions.
    """
    def __init__(self, epsilon=1e-10):
        super(MutualInformationLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        B, M, N, P = phi.shape
        phi = phi.reshape(B, M * N * P)
        phi = F.softmax(phi, dim=1)
        phi = phi.reshape(B, M, N, P)

        p_m = phi.sum(dim=(2, 3))  # [B, M]
        p_t = phi.sum(dim=(1, 2))  # [B, P]
        p_mt = phi.sum(dim=2)      # [B, M, P]

        denominator = p_m.unsqueeze(2) * p_t.unsqueeze(1)  # [B, M, P]
        numerator = p_mt

        log_term = torch.log((numerator + self.epsilon) / (denominator + self.epsilon))
        mutual_info = torch.sum(p_mt * log_term, dim=(0, 1, 2))

        return -mutual_info


class ModuleRouter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(input_dim * 3, 3)
        self.mi_criterion = MutualInformationLoss()
        self.mi_loss = None

    def forward(self, id_feat, img_feat, txt_feat):
        pooled_id = self.pool(id_feat.transpose(1, 2)).squeeze(-1)
        pooled_img = self.pool(img_feat.transpose(1, 2)).squeeze(-1)
        pooled_txt = self.pool(txt_feat.transpose(1, 2)).squeeze(-1)

        combined = torch.cat([pooled_id, pooled_img, pooled_txt], dim=-1)  # [B, 3D]
        logits = self.linear(combined)  # [B, 3]

        # === 计算互信息损失 ===
        phi = F.softmax(logits, dim=-1).reshape(-1, 1, 1, 3)
        self.mi_loss = self.mi_criterion(phi.detach())

        avg_score = logits.mean(dim=0)  # [3]
        sorted_indices = torch.argsort(avg_score, descending=True)
        return [MODULE_NAMES[i] for i in sorted_indices.tolist()]
