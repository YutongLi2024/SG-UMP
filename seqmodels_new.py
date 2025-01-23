import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, LayerNorm, DistSAEncoder
from modules import LayerNorm, DistSAEncoder



def global_embeddings(self, x1):

    x1 = self.low_rank_layer((x1.transpose(0, 1)))
    x1 = self.transformer(x1)
    x1 = self.low_rank_layer1(x1.transpose(0, 1))

    return x1


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

        self.item_filter = FilterLayer(hidden_size=args.hidden_size, num_layers=self.num_layers)
        self.image_filter = FilterLayer(hidden_size=args.hidden_size, num_layers=self.num_layers)
        self.text_filter = FilterLayer(hidden_size=args.hidden_size, num_layers=self.num_layers)

        self.apply(self.init_weights)

        print("----------start loading multi_modality -----------")
        # self.replace_embedding()

    

    def replace_embedding(self):
        text_features_list = torch.load(self.args.text_emb_path)
        image_features_list = torch.load(self.args.image_emb_path)
        self.image_mean_embeddings.weight.data[1:-1, :] = image_features_list
        self.image_cov_embeddings.weight.data[1:-1, :] = image_features_list
        self.text_mean_embeddings.weight.data[1:-1, :] = text_features_list
        self.text_cov_embeddings.weight.data[1:-1, :] = text_features_list

    def add_position_mean_embedding(self, sequence, remove_image, remove_text):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_mean_embeddings(position_ids)

        item_embeddings = self.item_mean_embeddings(sequence)  # (256,100,64)
        item_image_embeddings = self.fc_mean_image(self.image_mean_embeddings(sequence))
        item_text_embeddings = self.fc_mean_text(self.text_mean_embeddings(sequence))

        item_embeddings = self.item_filter(item_embeddings)

        # Multi-scale Multimodal Fusion Layer
        if self.args.is_use_mm:

            if self.args.is_use_cross:
                x = torch.cat([item_embeddings, item_image_embeddings, item_text_embeddings], axis=-1)

                experts_shared_o = [e(x) for e in self.experts_shared_mean]
                experts_shared_o = torch.stack(experts_shared_o)
                experts_shared_o = experts_shared_o.squeeze()

                # gate_share
                selected_s = self.dnn_share_mean(x)
                gate_share_out = torch.einsum('abcd, bca -> bcd', experts_shared_o, selected_s)

                experts_task1_o = [e(item_embeddings + gate_share_out) for e in self.experts_task1_mean]
                experts_task1_o = torch.stack(experts_task1_o)
                experts_task2_o = [e(item_image_embeddings + gate_share_out) for e in self.experts_task2_mean]
                experts_task2_o = torch.stack(experts_task2_o)
                experts_task3_o = [e(item_text_embeddings + gate_share_out) for e in self.experts_task3_mean]
                experts_task3_o = torch.stack(experts_task3_o)

                # experts_shared_o = experts_shared_o.squeeze()
                experts_task1_o = experts_task1_o.squeeze()
                experts_task2_o = experts_task2_o.squeeze()
                experts_task3_o = experts_task3_o.squeeze()

                # gate1
                selected1 = self.dnn1_mean(item_embeddings)
                gate_1_out = torch.einsum('abcd, bca -> bcd', experts_task1_o, selected1)

                # gate2
                selected2 = self.dnn2_mean(item_image_embeddings)
                gate_2_out = torch.einsum('abcd, bca -> bcd', experts_task2_o, selected2)

                # gate3
                selected3 = self.dnn3_mean(item_text_embeddings)
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

                item_embeddings = item_embeddings + item_image_embeddings + item_text_embeddings + task_out

            else:
                item_embeddings = item_embeddings + item_image_embeddings + item_text_embeddings

        elif self.args.is_use_text:
            item_embeddings = item_embeddings + item_text_embeddings
        elif self.args.is_use_image:
            item_embeddings = item_embeddings + item_image_embeddings


        # Multi-view Multimodal Refine Layer
        item_global_embeddings = global_embeddings(self, item_embeddings)
        item_image_global_embeddings = global_embeddings(self, item_image_embeddings)
        item_text_global_embeddings = global_embeddings(self, item_text_embeddings)

        item_image_embeddings = (torch.multiply(item_image_embeddings, self.gate_v(item_embeddings))
                                + torch.multiply(item_image_global_embeddings, self.gate_v(item_global_embeddings)))

        item_text_embeddings = (torch.multiply(item_text_embeddings, self.gate_v(item_embeddings))
                                + torch.multiply(item_text_global_embeddings, self.gate_t(item_global_embeddings)))

        item_embeddings = item_embeddings + item_image_embeddings + item_text_embeddings + task_out   

        item_embeddings = self.item_filter(item_embeddings)
        
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(sequence_emb)

        return sequence_emb

    def add_position_cov_embedding(self, sequence, remove_image, remove_text):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_cov_embeddings(position_ids)

        item_embeddings = self.item_cov_embeddings(sequence)
        item_image_embeddings = self.fc_cov_image(self.image_cov_embeddings(sequence))
        item_text_embeddings = self.fc_cov_text(self.text_cov_embeddings(sequence))

        

        # Multi-view Multimodal Refine Layer
        item_global_embeddings = global_embeddings(self, item_embeddings)
        item_image_global_embeddings = global_embeddings(self, item_image_embeddings)
        item_text_global_embeddings = global_embeddings(self, item_text_embeddings)

        item_image_embeddings = (torch.multiply(item_image_embeddings, self.gate_v(item_embeddings))
                                 + torch.multiply(item_image_global_embeddings,
                                                  self.gate_v(item_global_embeddings)))

        item_text_embeddings = (torch.multiply(item_text_embeddings, self.gate_v(item_embeddings))
                                + torch.multiply(item_text_global_embeddings,
                                                 self.gate_t(item_global_embeddings)))

        item_embeddings = self.item_filter(item_embeddings)
        
        # Multi-scale Multimodal Fusion Layer
        if self.args.is_use_mm:

            if self.args.is_use_cross:
                # print("use cross")

                x = torch.cat([item_embeddings, item_image_embeddings, item_text_embeddings], axis=-1)

                # share 
                experts_shared_o = [e(x) for e in self.experts_shared_cov]
                experts_shared_o = torch.stack(experts_shared_o)
                experts_shared_o = experts_shared_o.squeeze()

                selected_s = self.dnn_share_cov(x)
                gate_share_out = torch.einsum('abcd, bca -> bcd', experts_shared_o, selected_s)

                # modality
                experts_task1_o = [e(item_embeddings + gate_share_out) for e in self.experts_task1_cov]
                experts_task1_o = torch.stack(experts_task1_o)
                experts_task2_o = [e(item_image_embeddings + gate_share_out) for e in self.experts_task2_cov]
                experts_task2_o = torch.stack(experts_task2_o)
                experts_task3_o = [e(item_text_embeddings + gate_share_out) for e in self.experts_task3_cov]
                experts_task3_o = torch.stack(experts_task3_o)

                experts_task1_o = experts_task1_o.squeeze()
                experts_task2_o = experts_task2_o.squeeze()
                experts_task3_o = experts_task3_o.squeeze()

                # gate1
                selected1 = self.dnn1_cov(item_embeddings)  # (256,100,2)
                gate_1_out = torch.einsum('abcd, bca -> bcd', experts_task1_o, selected1)  # (256,100,64)

                # gate2
                selected2 = self.dnn2_cov(item_image_embeddings)
                gate_2_out = torch.einsum('abcd, bca -> bcd', experts_task2_o, selected2)

                # gate3
                selected3 = self.dnn3_cov(item_text_embeddings)
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

                item_embeddings = item_embeddings + item_image_embeddings + item_text_embeddings + task_out                                     

            else:
                # print("use mm")
                item_embeddings = item_embeddings + item_image_embeddings + item_text_embeddings

        elif self.args.is_use_text:
            item_embeddings = item_embeddings + item_text_embeddings
        elif self.args.is_use_image:
            item_embeddings = item_embeddings + item_image_embeddings

        # item_embeddings = self.item_filter(item_embeddings)
        
        # Multi-view Multimodal Refine Layer
        item_global_embeddings = global_embeddings(self, item_embeddings)
        item_image_global_embeddings = global_embeddings(self, item_image_embeddings)
        item_text_global_embeddings = global_embeddings(self, item_text_embeddings)

        item_image_embeddings = (torch.multiply(item_image_embeddings, self.gate_v(item_embeddings))
                                + torch.multiply(item_image_global_embeddings, self.gate_v(item_global_embeddings)))

        item_text_embeddings = (torch.multiply(item_text_embeddings, self.gate_v(item_embeddings))
                                + torch.multiply(item_text_global_embeddings, self.gate_t(item_global_embeddings)))

        item_embeddings = self.item_filter(item_embeddings) 

        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(self.dropout(sequence_emb)) + 1

        return sequence_emb

    def finetune(self, input_ids, user_ids, remove_image, remove_text):
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

        mean_sequence_emb = self.add_position_mean_embedding(input_ids, remove_image=remove_image, remove_text=remove_text)
        cov_sequence_emb = self.add_position_cov_embedding(input_ids, remove_image=remove_image, remove_text=remove_text)


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

