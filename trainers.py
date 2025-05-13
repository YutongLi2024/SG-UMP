# -*- coding: utf-8 -*-
import numpy as np
import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, get_metric, cal_mrr
from modules import wasserstein_distance, kl_distance, wasserstein_distance_matmul
from STOSA import STOSA
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]), flush=True)
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix, flush=True)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix), None

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        recall_dict_list = []
        ndcg_dict_list = []
        
        for k in [10, 20]:
            recall_result, recall_dict_k = recall_at_k(answers, pred_list, k)
            recall.append(recall_result)
            recall_dict_list.append(recall_dict_k)
            
            ndcg_result, ndcg_dict_k = ndcg_k(answers, pred_list, k)
            ndcg.append(ndcg_result)
            ndcg_dict_list.append(ndcg_dict_k)
        
        # post_fix = {
        #     "Epoch": epoch,
        #     "Recall@10": '{:.8f}'.format(recall[0]), "NDCG@10": '{:.8f}'.format(ndcg[0]),
        #     "Recall@20": '{:.8f}'.format(recall[1]), "NDCG@20": '{:.8f}'.format(ndcg[1]),
        # }
        post_fix = {
            "Epoch": epoch,
            "Recall@10": '{:.4f}'.format(recall[0]), 
            "Recall@20": '{:.4f}'.format(recall[1]),
            "NDCG@10": '{:.4f}'.format(ndcg[0]), 
            "NDCG@20": '{:.4f}'.format(ndcg[1]),
        }
        
        print(post_fix, flush=True)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        
        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix), [recall_dict_list, ndcg_dict_list]


    def get_pos_items_ranks(self, batch_pred_lists, answers):
        num_users = len(batch_pred_lists)
        batch_pos_ranks = defaultdict(list)
        for i in range(num_users):
            pred_list = batch_pred_lists[i]
            true_set = set(answers[i])
            for ind, pred_item in enumerate(pred_list):
                if pred_item in true_set:
                    batch_pos_ranks[pred_item].append(ind+1)
        return batch_pos_ranks

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name, map_location='cuda:0'))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc

    def predict_sample(self, seq_out, test_neg_sample):
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class SASRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(SASRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        rec_data_iter = dataloader
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            rec_avg_auc = 0.0

            # for i, batch in rec_data_iter:
            for batch in tqdm(rec_data_iter):
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch
                # Binary cross_entropy
                sequence_output, _ = self.model.finetune(input_ids)
                loss, batch_auc = self.cross_entropy(sequence_output, target_pos, target_neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_auc += batch_auc.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_auc": '{:.4f}'.format(rec_avg_auc / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                #  for i, batch in rec_data_iter:
                i = 0
                for batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output, _ = self.model.finetune(input_ids)

                    recommend_output = recommend_output[:, -1, :]

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind = np.argpartition(rating_pred, -40)[:, -40:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                    i += 1
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                #  for i, batch in rec_data_iter:
                i = 0
                for batch in tqdm(rec_data_iter):
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    # print(input_ids)
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                    i += 1

                return self.get_sample_scores(epoch, pred_list)


class STOSATrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(STOSATrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def bpr_optimization(self, seq_mean_out, seq_cov_out, pos_ids, neg_ids):  
        # print("bpr_optimization")
        # [batch seq_len hidden_size]
        activation = nn.ELU()
        pos_mean_emb = self.model.item_mean_embeddings(pos_ids)
        pos_cov_emb = activation(self.model.item_cov_embeddings(pos_ids)) + 1
        neg_mean_emb = self.model.item_mean_embeddings(neg_ids)
        neg_cov_emb = activation(self.model.item_cov_embeddings(neg_ids)) + 1

        # [batch*seq_len hidden_size]
        pos_mean = pos_mean_emb.view(-1, pos_mean_emb.size(2))
        pos_cov = pos_cov_emb.view(-1, pos_cov_emb.size(2))
        neg_mean = neg_mean_emb.view(-1, neg_mean_emb.size(2))
        neg_cov = neg_cov_emb.view(-1, neg_cov_emb.size(2))
        seq_mean_emb = seq_mean_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        seq_cov_emb = seq_cov_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]

        if self.args.distance_metric == 'wasserstein':
            pos_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = wasserstein_distance(pos_mean, pos_cov, neg_mean, neg_cov)

        else:
            pos_logits = kl_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = kl_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = kl_distance(pos_mean, pos_cov, neg_mean, neg_cov)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(-torch.log(torch.sigmoid(neg_logits - pos_logits + 1e-24)) * istarget) / torch.sum(istarget)
        pvn_loss = self.args.pvn_weight * torch.sum(torch.clamp(pos_logits - pos_vs_neg, 0) * istarget) / torch.sum(istarget)
        auc = torch.sum(
            ((torch.sign(neg_logits - pos_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc, pvn_loss

    def dist_predict_full(self, seq_mean_out, seq_cov_out):  
        # print("dist_predict_full")
        elu_activation = torch.nn.ELU()
        test_item_mean_emb = self.model.item_mean_embeddings.weight
        test_item_cov_emb = elu_activation(self.model.item_cov_embeddings.weight) + 1

        return wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        rec_data_iter = dataloader

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            rec_avg_pvn_loss = 0.0
            rec_avg_auc = 0.0

            # for batch in rec_data_iter:
            for batch in tqdm(rec_data_iter, desc="Training Progress"):
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, _ = batch
                # bpr optimization
                sequence_mean_output, sequence_cov_output, _, _, mi_loss = self.model.finetune(input_ids, user_ids)
                loss, batch_auc, pvn_loss = self.bpr_optimization(sequence_mean_output, sequence_cov_output, target_pos, target_neg)

                # loss = loss + pvn_loss
                loss = loss + pvn_loss + mi_loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_auc += batch_auc.item()
                rec_avg_pvn_loss += pvn_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_auc": '{:.6f}'.format(rec_avg_auc / len(rec_data_iter)),
                "rec_avg_pvn_loss": '{:.6f}'.format(rec_avg_pvn_loss / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')
        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                with torch.no_grad():
                    # for i, batch in rec_data_iter:
                    i = 0
                    for batch in tqdm(rec_data_iter):
                        # 0. batch_data will be sent into the device(GPU or cpu)
                        batch = tuple(t.to(self.device) for t in batch)
                        user_ids, input_ids, target_pos, target_neg, answers = batch
                        recommend_mean_output, recommend_cov_output, _, _, _= self.model.finetune(input_ids, user_ids)

                        recommend_mean_output = recommend_mean_output[:, -1, :]
                        recommend_cov_output = recommend_cov_output[:, -1, :]

                        rating_pred = self.dist_predict_full(recommend_mean_output, recommend_cov_output)
                        rating_pred = rating_pred.cpu().data.numpy().copy()
                        batch_user_index = user_ids.cpu().numpy()
                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 1e+24
                        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                        ind = np.argpartition(rating_pred, 40)[:, :40]
                        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                        # ascending order
                        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::]
                        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                        if i == 0:
                            pred_list = batch_pred_list
                            answer_list = answers.cpu().data.numpy()
                        else:
                            pred_list = np.append(pred_list, batch_pred_list, axis=0)
                            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                        i += 1
                    return self.get_full_sort_score(epoch, answer_list, pred_list)


class OracleTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):
        super().__init__(model, train_dataloader, eval_dataloader, test_dataloader, args)
        
        # 重新配置优化器 - 释放基类创建的默认优化器
        del self.optim
        
        # 创建Oracle模型特有的双优化器
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim_future = Adam(
            self.model.future_ae.parameters(),
            lr=self.args.lr, 
            betas=betas, 
            weight_decay=self.args.weight_decay
        )
        
        self.optim_past = Adam(
            list(self.model.past_ae.parameters()) + list(self.model.transition.parameters()),
            lr=self.args.lr, 
            betas=betas, 
            weight_decay=self.args.weight_decay
        )

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        """实现基类要求的iteration接口"""
        if train:
            return self._train_epoch(epoch, dataloader)
        else:
            return self._evaluate(epoch, dataloader, full_sort)

    def _train_epoch(self, epoch, dataloader):
        """Oracle双阶段训练核心逻辑"""
        self.model.train()
        total_loss = 0.0
        
        rec_data_iter = tqdm(
            enumerate(dataloader), 
            desc=f'Train in {epoch}-th epoch', 
            total=len(dataloader),
            ncols=120
        )
        
        for i, batch in rec_data_iter:
            # 设备转移
            batch = tuple(t.to(self.device) for t in batch)
            _, input_ids, answer, neg_answer, input_ids_future, answer_future, neg_answer_future = batch
            
            # ---- 阶段1：训练Future编码器 ----
            # 确保参数同步 - 从past到future
            self._sync_embeddings(src=self.model.past_ae, tgt=self.model.future_ae)
            
            # 前向计算
            future_loss, _, _ = self.model.future_forward(
                input_ids_future, answer_future, neg_answer_future
            )
            
            # 反向传播
            self.optim_future.zero_grad()
            future_loss.backward()
            self.optim_future.step()
            
            # 获取未来表征
            _, z_future, z_future_mask = self.model.future_forward(
                input_ids_future, answer_future, neg_answer_future
            )
            z_future = z_future.detach()  # 分离计算图
            
            # ---- 阶段2：训练Past编码器 ----
            # 确保参数同步 - 从future到past
            self._sync_embeddings(src=self.model.future_ae, tgt=self.model.past_ae)
            
            # 过去编码器前向
            past_loss, z_past = self.model.past_forward(input_ids, answer, neg_answer)
            
            # Transition计算
            transition_loss = self.model.transition_forward(z_past, z_future, z_future_mask)
            
            # 组合损失
            total_batch_loss = past_loss + 0.01 * transition_loss
            
            # 反向传播
            self.optim_past.zero_grad()
            total_batch_loss.backward()
            self.optim_past.step()
            
            # 更新统计
            batch_loss = future_loss.item() + past_loss.item() + 0.01 * transition_loss.item()
            total_loss += batch_loss
            
            # 更新进度条
            post_fix = {'rec_loss': '{:.4f}'.format(total_loss / (i+1))}
            rec_data_iter.set_postfix(post_fix)
        
        return total_loss / len(dataloader)

    def _sync_embeddings(self, src, tgt):
        """动态参数同步工具"""
        # Item embeddings同步
        tgt.item_embeddings.weight.data = src.item_embeddings.weight.data.detach()
        
        # Position embeddings同步
        tgt.position_embeddings.weight.data = src.position_embeddings.weight.data.detach()

    def _evaluate(self, epoch, dataloader, full_sort):
        """评估函数"""
        self.model.eval()
        
        if full_sort:
            return self._full_sort_eval(epoch, dataloader)
        else:
            return self._sample_eval(epoch, dataloader)

    def _sample_eval(self, epoch, dataloader):
        """采样评估"""
        pred_list = None
        rec_data_iter = tqdm(
            enumerate(dataloader), 
            desc=f'{"Valid" if "valid" in str(dataloader.dataset) else "Test"} in {epoch}-th epoch', 
            total=len(dataloader),
            ncols=120
        )
        
        for i, batch in rec_data_iter:
            batch = tuple(t.to(self.device) for t in batch)
            user_ids, input_ids, answers, _, sample_negs = batch
            
            # 生成预测
            with torch.no_grad():
                test_neg_items = torch.cat((answers.unsqueeze(-1), sample_negs), -1)
                test_logits = self.model.predict(input_ids, test_neg_items)
                test_logits = test_logits.cpu().detach().numpy().copy()
            
            # 收集结果
            if i == 0:
                pred_list = test_logits
            else:
                pred_list = np.append(pred_list, test_logits, axis=0)
        
        # 调用评分方法
        return self.get_sample_scores(epoch, pred_list)

    def _full_sort_eval(self, epoch, dataloader):
        """全排序评估"""
        answer_list = []
        pred_list = []
        
        for batch in tqdm(dataloader, desc=f'Full Sort Eval in {epoch}-th epoch', ncols=120):
            batch = tuple(t.to(self.device) for t in batch)
            user_ids, input_ids, answers = batch[:3]
            
            # 生成预测
            with torch.no_grad():
                rating_pred = self.model.predict_full(input_ids)
            
            # 屏蔽训练数据
            rating_pred[self.args.train_matrix[user_ids.cpu().numpy()].toarray() > 0] = -np.inf
            
            # 收集结果
            pred_list.append(rating_pred.cpu().numpy())
            answer_list.append(answers.cpu().numpy())
        
        # 调用评分方法
        return self.get_full_sort_score(
            epoch, 
            np.concatenate(answer_list, axis=0),
            np.concatenate(pred_list, axis=0)
        )