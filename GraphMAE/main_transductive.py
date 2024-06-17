import logging
import sys 
import numpy as np
from tqdm import tqdm
import torch
import os
import dgl
from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
sys.path.append("..")
from data_utils.dataset import check_candidate_lists,modified_edge_index_tensor,modify_edge_index_one_time,modify_edge_index_one_time_ratio
from graphmae.datasets.data_util import scale_feats
from data_utils.load import load_llm_feature_and_data
import data_utils.logistic_regression_eval as eval
from data_utils.logistic_regression_eval import Multi_K_Shot_Evaluator
from graphmae.models import build_model
from graphmae.evaluation import node_classification_evaluation

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

    return model


def remove_del_candidate_from_edges(edge_index, del_edges_tensor):
    device = edge_index.device

    del_set = set()
    for node_index, sub_tensor in enumerate(del_edges_tensor):
        for index in sub_tensor[0]:
            if index != -1:
                del_set.add((node_index, index.item()))

    original_edges = set()
    for col in range(edge_index.shape[1]):
        original_edges.add((edge_index[0, col].item(), edge_index[1, col].item()))
    
    updated_edge_set = original_edges - del_set
    # 分离源节点和目标节点
    source_nodes = [pair[0] for pair in updated_edge_set]
    target_nodes = [pair[1] for pair in updated_edge_set]

    # 转换为张量
    updated_edge_index_tensor = torch.tensor([source_nodes, target_nodes]).to(device)

    return updated_edge_index_tensor

def add_candidate_edges_to_edges(edge_index, add_edges_tensor):
    device = edge_index.device

    # 将添加的边转换为集合，以便进行操作
    add_set = set()
    for node_index, sub_tensor in enumerate(add_edges_tensor):
        for index in sub_tensor[0]:
            if index != -1:
                add_set.add((node_index, index.item()))

    # 将原始的边转换为集合
    original_edges = set()
    for col in range(edge_index.shape[1]):
        original_edges.add((edge_index[0, col].item(), edge_index[1, col].item()))
    
    # 更新边集合，将需要添加的边加入
    updated_edge_set = original_edges.union(add_set)
    
    # 分离源节点和目标节点
    source_nodes = [pair[0] for pair in updated_edge_set]
    target_nodes = [pair[1] for pair in updated_edge_set]

    # 转换为张量
    updated_edge_index_tensor = torch.tensor([source_nodes, target_nodes]).to(device)

    return updated_edge_index_tensor

import torch
import dgl
import gc

def update_dgl_edges(original_graph, new_edge_tensor, device, batch_size=1000):
    gc.collect()  # Collect garbage to free up as much memory as possible
    torch.cuda.empty_cache()  # Release unoccupied cached memory
    print(f"Initial GPU memory usage: {torch.cuda.memory_allocated(device) / 1e6} MB")

    # Remove all edges from the original graph
    graph = dgl.remove_edges(original_graph, torch.arange(0, original_graph.number_of_edges()).to(device))
    print(f"GPU memory usage after removing edges: {torch.cuda.memory_allocated(device) / 1e6} MB")

    # Determine the number of batches
    num_edges = new_edge_tensor.size(1)
    num_batches = (num_edges + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        # Calculate the start and end indices for this batch
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, num_edges)

        # Extract the batch of source and destination nodes
        src_nodes_batch = new_edge_tensor[0, start_idx:end_idx].to(device)
        dst_nodes_batch = new_edge_tensor[1, start_idx:end_idx].to(device)
        print(f"GPU memory usage after loading batch {batch_num}: {torch.cuda.memory_allocated(device) / 1e6} MB")

        # Add this batch of new edges to the graph
        graph = dgl.add_edges(graph, src_nodes_batch, dst_nodes_batch)
        print(f"GPU memory usage after adding batch {batch_num} edges: {torch.cuda.memory_allocated(device) / 1e6} MB")

        # Clear the memory of the current batch variables if they are no longer needed
        del src_nodes_batch
        del dst_nodes_batch
        torch.cuda.memory_summary(device)
        torch.cuda.empty_cache()  # Release unoccupied cached memory

    # Note: This operation does not retain the original edge features.
    # If you need to keep or update edge features, further operations are needed.

    return graph

def update_graph_with_new_edges(old_graph, new_edge_tensor, device):
    # 创建新图
    src_nodes = new_edge_tensor[0, :].to(device)
    dst_nodes = new_edge_tensor[1, :].to(device)
    new_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=old_graph.number_of_nodes(), device=device)
    
    # 复制节点特征
    for feature_name in old_graph.ndata:
        new_graph.ndata[feature_name] = old_graph.ndata[feature_name].to(device)
    return new_graph

def dgl_to_torch_tensor(graph, device):
    # Get edge indices
    src, dst = graph.edges()
    
    # Convert to PyTorch tensor and stack for PyTorch tensor format
    edge_tensor = torch.stack([src, dst], dim=0).to(device)
    
    return edge_tensor


hyperparams_space = {
    'lr': [3e-4],
    'emb_epoch': [1],
    'max_epoch': [500],
    'num_hidden': [1024],  
    'num_layers': [2],
    'mask_rate':[0.9],
    'drop_edge_rate':[0.2],
    'ratio':[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
}

results = {
    "best_acc_search": 0.0,
    "variance": 0.0,
    "best_hyperparams": {}
}

accs = []

def main(args):
    #Grid Search
    global results
    global accs
    var_with_best_acc = 0.0
    best_acc_search = 0.0
    best_hyperparams = {}

    for lr in hyperparams_space['lr']:
        for emb_epoch in hyperparams_space['emb_epoch']:
            for max_epoch in hyperparams_space['max_epoch']:
                for num_hidden in hyperparams_space['num_hidden']:
                    for num_layers in hyperparams_space['num_layers']:
                        for mask_rate in hyperparams_space['mask_rate']:
                            for drop_edge_rate in hyperparams_space['drop_edge_rate']:
                                for ratio in hyperparams_space['ratio']:
                                    hyperparams = {
                                        'lr': lr,
                                        'max_epoch': max_epoch,
                                        'num_hidden': num_hidden,
                                        'num_layers': num_layers,
                                        'emb_epoch':emb_epoch,
                                        'mask_rate':mask_rate,
                                        'drop_edge_rate':drop_edge_rate
                                        }
                                    ratio_this_round = ratio
                                    args.lr = lr
                                    args.max_epoch = max_epoch
                                    args.num_hidden = num_hidden
                                    args.num_layers = num_layers
                                    args.emb_epoch = emb_epoch
                                    args.mask_rate = mask_rate
                                    args.drop_edge_rate = drop_edge_rate
                                    #args.encoder = 'gcn'
                                    args.eval_multi_k = True
                                    device = args.device if args.device >= 0 else "cpu"
                                    
                                    data_seeds = args.data_seeds
                                    model_seeds = args.model_seeds
                                    dataset_name = args.dataset
                                    max_epoch = args.max_epoch
                                    max_epoch_f = args.max_epoch_f
                                    num_hidden = args.num_hidden
                                    num_layers = args.num_layers
                                    encoder_type = args.encoder
                                    decoder_type = args.decoder
                                    replace_rate = args.replace_rate
                                
                                    optim_type = args.optimizer 
                                    loss_fn = args.loss_fn
                                
                                    lr = args.lr
                                    weight_decay = args.weight_decay
                                    lr_f = args.lr_f
                                    weight_decay_f = args.weight_decay_f
                                    linear_prob = args.linear_prob
                                    load_model = args.load_model
                                    save_model = args.save_model
                                    logs = args.logging
                                    use_scheduler = args.scheduler
            
                                    print(args)
            
            
                                    
                                    if args.eval_multi_k:
                                        multi_k_eval=Multi_K_Shot_Evaluator()
                                
                                    graph = load_llm_feature_and_data(dataset_name=args.dataset,LLM_feat_seed=model_seeds[0],lm_model_name='microsoft/deberta-base',feature_type=f'attention-{emb_epoch}', use_dgl = True , device = device, sclae_feat= True if dataset_name == "ogbn-arxiv" else False )
                                    
                                    edge_index = dgl_to_torch_tensor(graph,args.device)
                                    print(edge_index.shape)
                                    candidate_lists = check_candidate_lists(args.dataset)
                                    edge_index = modify_edge_index_one_time_ratio(dataset_name,edge_index,candidate_lists,ratio = ratio_this_round)
                                    print("ratio this round is ",ratio_this_round)
                                    print("edge_index this round is ",edge_index.shape)                         
                                    graph = update_graph_with_new_edges(graph,edge_index,args.device)
                                    
                                    graph = dgl.add_self_loop(graph)
                                    
            
                                    features = graph.ndata['feat'] 
                                    
                                    # graph, (num_features, num_classes) = load_dataset(dataset_name)
                                    
                                    (num_features, num_classes)= (features.shape[1],graph.ndata['label'].unique().size(0))
                                    args.num_features = num_features
                                
                                    final_acc_list = []
                                    early_stp_acc_list=[]
                                    for i, seed in enumerate(model_seeds):
                                        print(f"####### Run {i} for seed {seed}")
                                        set_random_seed(seed)
                                
                                        if logs:
                                            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
                                        else:
                                            logger = None
                                
                                        model = build_model(args)
                                        model.to(device)
                                        optimizer = create_optimizer(optim_type, model, lr, weight_decay)
                                
                                        if use_scheduler:
                                            logging.info("Use schedular")
                                            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                                            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                                        else:
                                            scheduler = None
                                
                                        x = features
                                
                                        if not load_model:
                                            model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
                                            model = model.cpu()
                                
                                        if load_model:
                                            logging.info("Loading Model ... ")
                                            model.load_state_dict(torch.load('C:/Users/YI/Desktop/cora_checkpoint.pt'))
                                        if save_model:
                                            logging.info("Saveing Model ...")
                                            torch.save(model.state_dict(), "checkpoint.pt")
                                
                                        model = model.to(device)
                                        model.eval()
                                
                                
                                        #acc_list = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f,max_epoch_f, device,dataset_name=args.dataset,data_random_seeds=args.data_seeds,mute=False)
                                        with torch.no_grad():
                                            feat = model.embed(graph.to(device), x.to(device))
                                            in_feat = x.shape[1]
                                        
                                        if args.eval_multi_k:
                                            multi_k_eval.multi_k_fit_logistic_regression_new(features=feat,labels=graph.ndata['label'],data_random_seeds=args.data_seeds,dataset_name=args.dataset,device=device)
                                        else:
                                            final_acc , early_stp_acc = eval.fit_logistic_regression_new(features=feat,labels=graph.ndata['label'],data_random_seeds=args.data_seeds,dataset_name=args.dataset,device=device)
                                            final_acc_list.extend(final_acc)
                                            early_stp_acc_list.extend(early_stp_acc)
                                        if logger is not None:
                                            logger.finish()
                                
                                    if args.eval_multi_k:
                                        acc_1 = multi_k_eval.save_csv_results(dataset_name=args.dataset, experience_name="GraphMAE",feature_type=args.feature_type + 'attention')
                                        second_part_1 = acc_1.split('/')[1]
                                        number_str_1 = second_part_1.split('±')[0]
                                        acc = float(number_str_1)
                                        var = float(second_part_1.split('±')[1])

                                        accs.append(acc)
                                        if acc > best_acc_search:
                                            best_acc_search = acc
                                            var_with_best_acc = var
                                            best_hyperparams = hyperparams
            
                                            print("best_hyperparams updated!")
                                            results["best_acc_search"] = best_acc_search
                                            results["best_hyperparams"] = best_hyperparams
                                            results["variance"] = var_with_best_acc
                                            # 输出结果到文件
                                            output_file_name = "GraphMAE_" + args.dataset + "_new_attention_struct"+".txt"
                                            with open(output_file_name, "w") as file:
                                                file.write("Best Accuracy: {}\n".format(results["best_acc_search"]))
                                                file.write("Variance with Best Acc: {}\n".format(results["variance"]))
                                                file.write("Best Hyperparameters:\n")
                                                for param, value in results["best_hyperparams"].items():
                                                    file.write("{}: {}\n".format(param, value))
            
                                            print(f"Results saved to {output_file_name}")

    print(accs)

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print("Original Args")
    print(args)
    main(args)
