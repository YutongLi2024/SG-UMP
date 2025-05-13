# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import SASRecDataset, OracleDataset
from trainers import SASRecTrainer, STOSATrainer, OracleTrainer
from STOSA import STOSA
from SASRec import SASRecModel
from utils import EarlyStopping, get_user_seqs, check_path, set_seed
from Oracle4Rec import Oracle4Rec
import time
import os

base_dir = os.path.dirname(os.path.abspath(__file__))  
data_name = 'Beauty'

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=f'./data/Features/{data_name}/', type=str)
    parser.add_argument('--output_dir', default=f'outputs/{data_name}', type=str)
    parser.add_argument('--data_name', default=f'reviews_{data_name}', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")
    parser.add_argument('--patience', default=10, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--model_name", default='STOSA', type=str)
    # parser.add_argument("--model_name", default='SASRec', type=str)
    parser.add_argument("--hidden_size", type=int, default=256, help="hidden size of transformer model") #64
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="number of layers")
    parser.add_argument('--num_attention_heads', default=4, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.0, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=100, type=int)
    parser.add_argument('--distance_metric', default='wasserstein', type=str)
    parser.add_argument('--pvn_weight', default=0.005, type=float)
    parser.add_argument('--kernel_param', default=1.0, type=float)

    # multimodal args
    parser.add_argument('--image_emb_path', default=f'data/Features/{data_name}/clip_image_features_{data_name}.pt', type=str)
    parser.add_argument('--text_emb_path', default=f'data/Features/{data_name}/clip_text_features_{data_name}.pt', type=str)
    parser.add_argument('--mm_emb_dim', default=512, type=int)
    parser.add_argument("--is_use_mm", type=bool, default=True, help="is use mm embedding")
    parser.add_argument("--is_use_text", type=bool, default=False, help="is use text embedding")
    parser.add_argument("--is_use_image", type=bool, default=False, help="is use image embedding")
    parser.add_argument("--pretrain_emb_dim", type=int, default=512, help="pretrain_emb_dim of clip model")
    parser.add_argument("--is_use_cross", type=bool, default=True, help="is use mm cross")
    parser.add_argument('--num_shared_experts', default=4, type=int, help="shared experts for multi-modal fusion")
    parser.add_argument('--num_specific_experts', default=4, type=int, help="specific experts for multi-modal fusion")
    parser.add_argument('--low_rank', default=4, type=int, help="low_rank matrix")
    parser.add_argument('--global_transformer_nhead', default=4, type=int)
    parser.add_argument("--prediction", type=bool, default=False, help="activate prediction mode")
    parser.add_argument(
        "--manual_module_order",
        nargs='+',  # 接收一个或多个字符串参数
        default=["attention", "filter", "fusion"], 
        # default=None,  # 此时自动选择
        help="Specify manual module order, e.g., --manual_module_order filter fusion attention"
    )


    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--device", type=str, default="cuda:0", help="train device")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'
    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = \
        get_user_seqs(args.data_file)

    args.item_size = max_item + 2
    args.num_users = num_users
    args.mask_id = max_item + 1

    # data_name is 'reviews_Home'，save 'Home'
    clean_data_name = args.data_name.replace('reviews_', '')

    # 定义模块名到缩写字母的映射
    module_short_map = {
        "filter": "a",
        "attention": "b",
        "fusion": "c"
    }

    # 处理模块顺序
    if args.manual_module_order is None:
        order_str = "auto"
    else:
        # 转换成对应的缩写字母顺序，例如 ['fusion', 'attention', 'filter'] -> 'c-b-a'
        short_order = [module_short_map[m] for m in args.manual_module_order]
        order_str = "-".join(short_order)

    # 拼接 log 文件名
    args_str = f'{args.model_name}-{clean_data_name}-{args.num_shared_experts}-{args.num_specific_experts}-order-{order_str}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)

    test_dataset = SASRecDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)

    if args.model_name == 'STOSA':
        model = STOSA(args=args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=100)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)
        trainer = STOSATrainer(model, train_dataloader, eval_dataloader,
                                    test_dataloader, args)
    elif args.model_name == 'SASRec':
        model = SASRecModel(args=args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

        trainer = SASRecTrainer(model, train_dataloader, eval_dataloader,
                                test_dataloader, args)
        
    else:
        train_dataset = OracleDataset(args, user_seq, data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

        eval_dataset = OracleDataset(args, user_seq, data_type='valid')
        eval_sampler = SequentialSampler(eval_dataset)

        test_dataset = OracleDataset(args, user_seq, data_type='test')
        test_sampler = SequentialSampler(test_dataset)

        model = Oracle4Rec(args=args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

        trainer = OracleTrainer(model, train_dataloader, eval_dataloader,
                                test_dataloader, args)

    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info, _ = trainer.test(0, full_sort=True)

    else:
        if args.model_name == 'STOSA':
            early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        else:
            early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):

            trainer.train(epoch)
            start_time_epoch_valid = time.time()
            scores, _, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)

            end_time_epoch_valid = time.time()
            epoch_duration = end_time_epoch_valid - start_time_epoch_valid
            x_epoch_duration = format(epoch_duration, ".2f")
            print(f"Epoch {epoch} valid in {epoch_duration:.2f} seconds.")

            with open(args.log_file, 'a') as f:
                f.write(f"Epoch {epoch} duration: {epoch_duration:.2f} seconds\n")
                f.write(x_epoch_duration + '\n')

            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('---------------Change to test_rating_matrix!-------------------')

        # Load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        valid_scores, _, _ = trainer.valid('best', full_sort=True)
        trainer.args.train_matrix = test_rating_matrix

        # Start timing the testing phase
        start_time_test = time.time()
        scores, result_info, _ = trainer.test('best', full_sort=True)
        end_time_test = time.time()

        # Calculate and log the prediction time
        prediction_duration = end_time_test - start_time_test
        print(f"Prediction time: {prediction_duration:.2f} seconds.")
        with open(args.log_file, 'a') as f:
            f.write(f"Prediction time: {prediction_duration:.2f} seconds\n")

    # Log total training time
    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
        f.write(f"Total training time: {int(minutes):02d}:{int(seconds):02d}" + '\n')

    print(f"Total training time: {int(minutes):02d}:{int(seconds):02d}")


main()
