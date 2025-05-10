import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, LayerNorm, DistSAEncoder
from modules import LayerNorm, DistSAEncoder




class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def finetune(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        sequence_output, attention_scores = item_encoded_layers[-1]
        return sequence_output, attention_scores

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


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


MODULE_NAMES = ["filter", "fusion", "attention"]

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




class STOSA(nn.Module):
    def __init__(self, args):
        super(STOSA, self).__init__()
        self.item_mean_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.item_cov_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_mean_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.position_cov_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.user_margins = nn.Embedding(args.num_users, 1)
        self.item_encoder = DistSAEncoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.num_layers = 1
        self.alpha = 1

        low_rank = args.low_rank
        self.transformer = nn.TransformerEncoderLayer(
            d_model=args.hidden_size // low_rank,
            nhead=args.global_transformer_nhead,
            dim_feedforward=args.hidden_size // low_rank
        )

        self.attention_layer = nn.Linear(args.hidden_size * 4, args.hidden_size)

        # Initialize learnable parameters
        self.w_mean = nn.Parameter(torch.rand(1))  # Learnable parameter for weighting similarity
        self.b_mean = nn.Parameter(torch.rand(1))  # Learnable parameter for biasing similarity
        self.w_cov = nn.Parameter(torch.rand(1))
        self.b_cov = nn.Parameter(torch.rand(1))

        self.low_rank_layer = nn.Linear(args.hidden_size, args.hidden_size // low_rank)
        self.low_rank_layer1 = nn.Linear(args.hidden_size // low_rank, args.hidden_size)

        # text
        self.text_mean_embeddings = nn.Embedding(args.item_size, args.pretrain_emb_dim, padding_idx=0)
        self.text_cov_embeddings = nn.Embedding(args.item_size, args.pretrain_emb_dim, padding_idx=0)

        # image
        self.image_mean_embeddings = nn.Embedding(args.item_size, args.pretrain_emb_dim, padding_idx=0)
        self.image_cov_embeddings = nn.Embedding(args.item_size, args.pretrain_emb_dim, padding_idx=0)


        self.fc_mean_image = nn.Linear(args.pretrain_emb_dim, args.hidden_size)
        self.fc_cov_image = nn.Linear(args.pretrain_emb_dim, args.hidden_size)
        self.fc_mean_text = nn.Linear(args.pretrain_emb_dim, args.hidden_size)
        self.fc_cov_text = nn.Linear(args.pretrain_emb_dim, args.hidden_size)

        self.fc_mean_image_layernorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.fc_cov_image_layernorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.fc_mean_text_layernorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.fc_cov_text_layernorm = LayerNorm(args.hidden_size, eps=1e-12)

        # gate
        self.gate_v = Gate(args.hidden_size, low_rank)
        self.gate_t = Gate(args.hidden_size, low_rank)

        self.fc1_mean = nn.Sequential(
            nn.Linear(self.args.hidden_size * 2, 2 * self.args.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(2 * self.args.hidden_size, self.args.hidden_size, bias=False),
            nn.GELU()
        )
        self.fc2_mean = nn.Sequential(
            nn.Linear(self.args.hidden_size * 2, 2 * self.args.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(2 * self.args.hidden_size, self.args.hidden_size, bias=False),
            nn.GELU()
        )

        self.fc1_cov = nn.Sequential(
            nn.Linear(self.args.hidden_size * 2, 2 * self.args.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(2 * self.args.hidden_size, self.args.hidden_size, bias=False),
            nn.GELU()
        )
        self.fc2_cov = nn.Sequential(
            nn.Linear(self.args.hidden_size * 2, 2 * self.args.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(2 * self.args.hidden_size, self.args.hidden_size, bias=False),
            nn.GELU()
        )

        # Multi-scale Multimodal Fusion Layer
        self.num_shared_experts = args.num_shared_experts
        self.num_specific_experts = args.num_specific_experts

        self.experts_shared_mean = nn.ModuleList([Expert(self.args.hidden_size * 3, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_shared_experts)])
        self.experts_task1_mean = nn.ModuleList([Expert(self.args.hidden_size, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_specific_experts)])
        self.experts_task2_mean = nn.ModuleList([Expert(self.args.hidden_size, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_specific_experts)])
        self.experts_task3_mean = nn.ModuleList([Expert(self.args.hidden_size, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_specific_experts)])

        self.experts_shared_cov = nn.ModuleList([Expert(self.args.hidden_size * 3, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_shared_experts)])
        self.experts_task1_cov = nn.ModuleList([Expert(self.args.hidden_size, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_specific_experts)])
        self.experts_task2_cov = nn.ModuleList([Expert(self.args.hidden_size, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_specific_experts)])
        self.experts_task3_cov = nn.ModuleList([Expert(self.args.hidden_size, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_specific_experts)])


        # By using a low-rank expert network and gating, the model parameters can be reduced without sacrificing performance.
        self.dnn_share_mean = nn.Sequential(
            nn.Linear(self.args.hidden_size * 3, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_shared_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.dnn_share_cov = nn.Sequential(
            nn.Linear(self.args.hidden_size * 3, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_shared_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.dnn1_mean = nn.Sequential(
            nn.Linear(self.args.hidden_size, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_specific_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.dnn2_mean = nn.Sequential(
            nn.Linear(self.args.hidden_size, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_specific_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.dnn3_mean = nn.Sequential(
            nn.Linear(self.args.hidden_size, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_specific_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.dnn1_cov = nn.Sequential(
            nn.Linear(self.args.hidden_size, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_specific_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.dnn2_cov = nn.Sequential(
            nn.Linear(self.args.hidden_size, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_specific_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.dnn3_cov = nn.Sequential(
            nn.Linear(self.args.hidden_size, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_specific_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.multi_scale_fusion_mean = MultiScaleFusionLayer(
            hidden_size=args.hidden_size,
            low_rank=args.low_rank,
            num_shared_experts=args.num_shared_experts,
            num_specific_experts=args.num_specific_experts,
            experts_shared=self.experts_shared_mean,
            experts_task1=self.experts_task1_mean,
            experts_task2=self.experts_task2_mean,
            experts_task3=self.experts_task3_mean,
            dnn_share=self.dnn_share_mean,
            dnn1=self.dnn1_mean,
            dnn2=self.dnn2_mean,
            dnn3=self.dnn3_mean,
            attention_layer=self.attention_layer
        )

        self.multi_scale_fusion_cov = MultiScaleFusionLayer(
            hidden_size=args.hidden_size,
            low_rank=args.low_rank,
            num_shared_experts=args.num_shared_experts,
            num_specific_experts=args.num_specific_experts,
            experts_shared=self.experts_shared_cov,
            experts_task1=self.experts_task1_cov,
            experts_task2=self.experts_task2_cov,
            experts_task3=self.experts_task3_cov,
            dnn_share=self.dnn_share_cov,
            dnn1=self.dnn1_cov,
            dnn2=self.dnn2_cov,
            dnn3=self.dnn3_cov,
            attention_layer=self.attention_layer
        )

        self.hierarchical_attention = HierarchicalAttentionLayer(
            hidden_size=args.hidden_size,
            low_rank=args.low_rank,
            gate_v=self.gate_v,
            gate_t=self.gate_t
        )

        self.item_filter = FilterLayer(hidden_size=args.hidden_size, num_layers=self.num_layers)
        self.image_filter = FilterLayer(hidden_size=args.hidden_size, num_layers=self.num_layers)
        self.text_filter = FilterLayer(hidden_size=args.hidden_size, num_layers=self.num_layers)

        self.module_router = ModuleRouter(args.hidden_size)
        self.module_order = None  # Will be set during first add_position_* call
        self.has_logged_module_order = False


        self.apply(self.init_weights)

        print("----------start loading multi_modality -----------")
        self.replace_embedding()

    

    def replace_embedding(self):
        text_features_list = torch.load(self.args.text_emb_path)
        image_features_list = torch.load(self.args.image_emb_path)
        self.image_mean_embeddings.weight.data[1:-1, :] = image_features_list
        self.image_cov_embeddings.weight.data[1:-1, :] = image_features_list
        self.text_mean_embeddings.weight.data[1:-1, :] = text_features_list
        self.text_cov_embeddings.weight.data[1:-1, :] = text_features_list


    def apply_modules_in_order(self, modules_dict, order, item_embeddings, img_feat, txt_feat):
        task_out = None
        for name in order:
            if name == "filter":
                id_feat = modules_dict["filter"](item_embeddings)
                img_feat = modules_dict["filter"](img_feat)
                txt_feat = modules_dict["filter"](txt_feat)
                item_embeddings = item_embeddings + img_feat + txt_feat + id_feat
            elif name == "fusion":
                task_out = modules_dict["fusion"](item_embeddings, img_feat, txt_feat)
                item_embeddings = item_embeddings + img_feat + txt_feat + task_out
            elif name == "attention":
                item_embeddings = modules_dict["attention"](item_embeddings, img_feat, txt_feat)
        return item_embeddings

    def get_module_routing_loss(self):
            if hasattr(self, "module_router") and self.module_router.mi_loss is not None:
                return self.module_router.mi_loss
            return torch.tensor(0.0, device=next(self.parameters()).device)

    def add_position_mean_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_mean_embeddings(position_ids)

        item_embeddings = self.item_mean_embeddings(sequence)  # (256,100,64)
        item_image_embeddings = self.fc_mean_image(self.image_mean_embeddings(sequence))
        item_text_embeddings = self.fc_mean_text(self.text_mean_embeddings(sequence))

        # 手动顺序优先，否则使用 module_router 自动生成
        if self.module_order is None:
            if hasattr(self.args, "manual_module_order") and self.args.manual_module_order:
                self.module_order = self.args.manual_module_order
                print("===> Using manually specified Module Order:", self.module_order)
            else:
                self.module_order = self.module_router(item_embeddings, item_image_embeddings, item_text_embeddings)
                print("===> Selected Module Order by router:", self.module_order)

            # 只记录一次模块顺序到日志文件
            if not self.has_logged_module_order and hasattr(self.args, 'log_file'):
                with open(self.args.log_file, 'a') as f:
                    f.write("===> Module Order: " + str(self.module_order) + '\n')
                self.has_logged_module_order = True


        item_embeddings = self.apply_modules_in_order(
            modules_dict={
                "filter": self.item_filter,
                "fusion": self.multi_scale_fusion_mean,
                "attention": self.hierarchical_attention
            },
            order=self.module_order,
            item_embeddings=item_embeddings,
            img_feat=item_image_embeddings,
            txt_feat=item_text_embeddings
        )

        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(sequence_emb)

        return sequence_emb

    def add_position_cov_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_cov_embeddings(position_ids)

        item_embeddings = self.item_cov_embeddings(sequence)
        item_image_embeddings = self.fc_cov_image(self.image_cov_embeddings(sequence))
        item_text_embeddings = self.fc_cov_text(self.text_cov_embeddings(sequence))

        # 手动顺序优先，否则使用 module_router 自动生成
        if self.module_order is None:
            if hasattr(self.args, "manual_module_order") and self.args.manual_module_order:
                self.module_order = self.args.manual_module_order
                print("===> Using manually specified Module Order:", self.module_order)
            else:
                self.module_order = self.module_router(item_embeddings, item_image_embeddings, item_text_embeddings)
                print("===> Selected Module Order by router:", self.module_order)

            # 只记录一次模块顺序到日志文件
            if not self.has_logged_module_order and hasattr(self.args, 'log_file'):
                with open(self.args.log_file, 'a') as f:
                    f.write("===> Module Order: " + str(self.module_order) + '\n')
                self.has_logged_module_order = True


        item_embeddings = self.apply_modules_in_order(
            modules_dict={
                "filter": self.item_filter,
                "fusion": self.multi_scale_fusion_cov,
                "attention": self.hierarchical_attention
            },
            order=self.module_order,
            item_embeddings=item_embeddings,
            img_feat=item_image_embeddings,
            txt_feat=item_text_embeddings
        )


        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(self.dropout(sequence_emb)) + 1
        
        return sequence_emb

    def finetune(self, input_ids, user_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * (-2 ** 32 + 1)

        mean_sequence_emb = self.add_position_mean_embedding(input_ids)
        cov_sequence_emb = self.add_position_cov_embedding(input_ids)
        
        mi_loss = self.get_module_routing_loss()
        
        item_encoded_layers = self.item_encoder(mean_sequence_emb,
                                                cov_sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        mean_sequence_output, cov_sequence_output, att_scores = item_encoded_layers[-1]

        margins = self.user_margins(user_ids)

        return mean_sequence_output, cov_sequence_output, att_scores, margins, mi_loss

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            module.weight.data.normal_(mean=0.01, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class DistMeanSAModel(STOSA):
    def __init__(self, args):
        super(DistMeanSAModel, self).__init__(args)
        self.item_encoder = STOSA(args)

