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
    def __init__(self, hidden_size, num_layers=2, dropout_prob=0.1):
        super(FilterLayer, self).__init__()
        self.num_layers = num_layers

        
        self.filter_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size, eps=1e-12),
                nn.Dropout(dropout_prob)
            )
            for _ in range(num_layers)
        ])

        
        self.complex_weights = None

    def initialize_weights(self, seq_len, hidden_size, device):
        
        freq_dim = seq_len // 2 + 1
        self.complex_weights = nn.ParameterList([
            nn.Parameter(
                torch.randn(1, freq_dim, hidden_size, dtype=torch.cfloat, device = device) * 0.02
            )
            for _ in range(self.num_layers)
        ])
    def forward(self, x, q=0.8):
        batch, seq_len, hidden_size = x.shape
        
        if self.complex_weights is None:
            self.initialize_weights(seq_len, hidden_size, device=x.device)

        for layer_idx in range(self.num_layers):
            fft_x = torch.fft.rfft(x, dim=1, norm="ortho")

            weight = self.complex_weights[layer_idx]
            fft_x = fft_x * weight

            fft_magnitude = torch.abs(fft_x)  
            q_value = torch.quantile(fft_magnitude.reshape(-1), q)  
            # print(f"Layer {layer_idx + 1}, q-{q} quantile: {q_value.item()}")  

            x_fft_filtered = torch.fft.irfft(fft_x, n=seq_len, dim=1, norm="ortho")

            x = self.filter_layers[layer_idx](x_fft_filtered) + x

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

        # experts_shared_o = [e(x) for e in self.experts_shared_mean]
        experts_shared_o = [e(x) for e in self.experts_shared]
        experts_shared_o = torch.stack(experts_shared_o)
        experts_shared_o = experts_shared_o.squeeze()

        # gate_share
        # selected_s = self.dnn_share_mean(x)
        selected_s = self.dnn_share(x)
        gate_share_out = torch.einsum('abcd, bca -> bcd', experts_shared_o, selected_s)

        experts_task1_o = [e(id_feat + gate_share_out) for e in self.experts_task1]
        experts_task1_o = torch.stack(experts_task1_o)
        experts_task2_o = [e(img_feat + gate_share_out) for e in self.experts_task2]
        experts_task2_o = torch.stack(experts_task2_o)
        experts_task3_o = [e(txt_feat + gate_share_out) for e in self.experts_task3]
        experts_task3_o = torch.stack(experts_task3_o)

        # experts_shared_o = experts_shared_o.squeeze()
        experts_task1_o = experts_task1_o.squeeze()
        experts_task2_o = experts_task2_o.squeeze()
        experts_task3_o = experts_task3_o.squeeze()

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

MODULE_NAMES = ["filter", "fusion", "attention"]


class ModuleRouter(nn.Module):
    """
    Automatically determines execution order of modules (filter, fusion, attention)
    based on average modal features. Only used once per dataset.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(input_dim * 3, 3)  # Predict score for each module

    def forward(self, id_feat, img_feat, txt_feat):
        """
        Args:
            id_feat:   [B, L, D]
            img_feat:  [B, L, D]
            txt_feat:  [B, L, D]

        Returns:
            A fixed order list of module names, e.g., ['fusion', 'filter', 'attention']
        """
        # [B, D] → 特征池化得到单个表示向量
        pooled_id = self.pool(id_feat.transpose(1, 2)).squeeze(-1)   # [B, D]
        pooled_img = self.pool(img_feat.transpose(1, 2)).squeeze(-1) # [B, D]
        pooled_txt = self.pool(txt_feat.transpose(1, 2)).squeeze(-1) # [B, D]

        # 拼接模态特征
        combined = torch.cat([pooled_id, pooled_img, pooled_txt], dim=-1)  # [B, 3D]
        scores = self.linear(combined)  # [B, 3]

        # 平均所有样本的排序结果（全局决定）
        avg_score = scores.mean(dim=0)  # [3]
        sorted_indices = torch.argsort(avg_score, descending=True)  # 排序模块索引

        return [MODULE_NAMES[i] for i in sorted_indices.tolist()]






class DistSAModel(nn.Module):
    def __init__(self, args):
        super(DistSAModel, self).__init__()
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
                item_embeddings = modules_dict["filter"](item_embeddings)
            elif name == "fusion":
                task_out = modules_dict["fusion"](item_embeddings, img_feat, txt_feat)
                item_embeddings = item_embeddings + img_feat + txt_feat + task_out
            elif name == "attention":
                item_embeddings = modules_dict["attention"](item_embeddings, img_feat, txt_feat)
        return item_embeddings


    def add_position_mean_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_mean_embeddings(position_ids)

        item_embeddings = self.item_mean_embeddings(sequence)  # (256,100,64)
        item_image_embeddings = self.fc_mean_image(self.image_mean_embeddings(sequence))
        item_text_embeddings = self.fc_mean_text(self.text_mean_embeddings(sequence))


        # # === Frequency Filter Layerr ===
        # item_embeddings = self.item_filter(item_embeddings)

        # # === Multi-Scale Fusion Layer ===
        # if self.args.is_use_mm and self.args.is_use_cross:
        #     task_out = self.multi_scale_fusion_mean(item_embeddings, item_image_embeddings, item_text_embeddings)
        #     item_embeddings = item_embeddings + item_image_embeddings + item_text_embeddings + task_out
        # elif self.args.is_use_mm:
        #     item_embeddings = item_embeddings + item_image_embeddings + item_text_embeddings
        # elif self.args.is_use_text:
        #     item_embeddings = item_embeddings + item_text_embeddings
        # elif self.args.is_use_image:
        #     item_embeddings = item_embeddings + item_image_embeddings

        # # === Hierarchical Attention Layer ===
        # item_embeddings = self.hierarchical_attention(item_embeddings, item_image_embeddings, item_text_embeddings)

        if self.module_order is None:
            self.module_order = self.module_router(item_embeddings, item_image_embeddings, item_text_embeddings)
            print("===> Selected Module Order:", self.module_order)

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

        # === Frequency Filter Layerr ===
        item_embeddings = self.item_filter(item_embeddings)

        # === Multi-Scale Fusion Layer ===
        if self.args.is_use_mm and self.args.is_use_cross:
            task_out = self.multi_scale_fusion_cov(item_embeddings, item_image_embeddings, item_text_embeddings)
            item_embeddings = item_embeddings + item_image_embeddings + item_text_embeddings + task_out
        elif self.args.is_use_mm:
            item_embeddings = item_embeddings + item_image_embeddings + item_text_embeddings
        elif self.args.is_use_text:
            item_embeddings = item_embeddings + item_text_embeddings
        elif self.args.is_use_image:
            item_embeddings = item_embeddings + item_image_embeddings

        # === Hierarchical Attention Layer ===
        item_embeddings = self.hierarchical_attention(item_embeddings, item_image_embeddings, item_text_embeddings)

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


        item_encoded_layers = self.item_encoder(mean_sequence_emb,
                                                cov_sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        mean_sequence_output, cov_sequence_output, att_scores = item_encoded_layers[-1]

        margins = self.user_margins(user_ids)

        return mean_sequence_output, cov_sequence_output, att_scores, margins

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


class DistMeanSAModel(DistSAModel):
    def __init__(self, args):
        super(DistMeanSAModel, self).__init__(args)
        self.item_encoder = DistSAModel(args)

