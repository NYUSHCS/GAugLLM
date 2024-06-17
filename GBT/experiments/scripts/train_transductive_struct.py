import json
import os
import sys
import yaml
from torch_geometric.utils.sparse import to_edge_index
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import sys
# sys.path.append('/LLMs-with-GSSL')  # 替换为实际的项目路径
sys.path.append("../../..") # TODO merge TAPE into current repo
from data_utils.dataset import check_candidate_lists,modified_edge_index_tensor
from data_utils.load import load_llm_feature_and_data
import argparse
from GBT.gssl import DATA_DIR
from GBT.gssl.datasets import load_dataset
from GBT.gssl.utils import seed
import copy
from copy import deepcopy
import sys
import data_utils.logistic_regression_eval as evaluate
from data_utils.logistic_regression_eval import Multi_K_Shot_Evaluator
sys.path.append("..") # TODO merge TAPE into current repo
from BGRL.bgrl.transforms_new import get_graph_drop_transform_new,remove_del_candidate_from_edges
from torch_geometric.utils import degree

hyperparams_space = {
    'lr': [1e-3],
    'p_x_1': [0.0],
    'p_x_2': [0.0],
    'add_edge_p': [0.3],
    'del_edge_p': [0.7],  
    'emb_epoch': [1]
}

results = {
    "best_acc_search": 0.0,
    "variance":0.0,
    "best_hyperparams": {}
}




def main():
    global results

    best_acc_search = 0.0
    best_hyperparams = {}


    parser = argparse.ArgumentParser(description='GraphCL')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument("--feature_type", type=str, required=True)
    parser.add_argument("--eval_multi_k", action="store_false", default=True)
    #parser.add_argument("--use_LLM_emb", action="store_true")
    parser.add_argument('--data_seeds', type=list, default=[1,2,0,3,4])
    parser.add_argument('--loss_name', type=str, default='GraphCL_loss')
    #parser.add_argument('--feat_aug',type=int,required=1)
    #parser.add_argument('--edge_aug',type=int,required=0)
    parser.add_argument('--p_x_1',type=float,default=None)
    parser.add_argument('--p_e_1',type=float,default=None)
    parser.add_argument('--p_x_2',type=float,default=None)
    parser.add_argument('--p_e_2',type=float,default=None)
    parser.add_argument('--lr_base',type=float,default=None)
    args = parser.parse_args()

    seed()

    # Read dataset name
    #dataset_name = sys.argv[1]
    dataset_name = args.dataset_name
    loss_name = args.loss_name
    if loss_name == 'GraphCL_loss':
        print("Using GraphCL Model")
        from GBT.gssl.transductive_model_GraphCL import Model
        from GBT.gssl.transductive_model_arxiv_GraphCL import ArxivModel
        
    elif loss_name == 'barlow_twins':
        print("Using GBT Model")
        from GBT.gssl.transductive_model_GBT import Model
        from GBT.gssl.transductive_model_arxiv_GBT import ArxivModel
        
    else:
        raise NotImplementedError
    
    if args.eval_multi_k:
        multi_k_eval_1=Multi_K_Shot_Evaluator(result_path='/LLM4GCL_update/results_1.csv')
        multi_k_eval_2=Multi_K_Shot_Evaluator(result_path='/LLM4GCL_update/results_2.csv')
    #Read params
    with open("../configs/train_transductive.yaml", "r") as fin:
        params = yaml.safe_load(fin)[loss_name][dataset_name]

    

    import os
    sys.path.append('/LLM4GCL_update')
    os.chdir('/LLM4GCL_update/GBT')
    data_1 = load_llm_feature_and_data(
        dataset_name=args.dataset_name,
        lm_model_name='microsoft/deberta-base',
        feature_type="GIA",
        device=args.device,
        use_text=False,
    )

    device = args.device
    

    for lr in hyperparams_space['lr']:
        for p_x_1 in hyperparams_space['p_x_1']:
            for p_x_2 in hyperparams_space['p_x_2']:
                for add_p in hyperparams_space['add_edge_p']:
                    for del_p in hyperparams_space['del_edge_p']:
                        for emb_epoch in hyperparams_space['emb_epoch']:
                            hyperparams = {
                                'lr': lr,
                                'p_x_1': p_x_1,
                                'p_x_2': p_x_2,
                                'add_edge_p': add_p,
                                'del_edge_p': del_p,
                                'emb_epoch':emb_epoch,
                            }
                            params["p_x_1"] = p_x_1
                            params["p_e_1"] = 0
                            params["p_x_2"] = p_x_2
                            params["p_e_2"] = 0
                            params["add_p"] = add_p
                            params["del_p"] = del_p
                            params["lr_base"] = lr

                            data_2 = load_llm_feature_and_data(
                                dataset_name=args.dataset_name,
                                lm_model_name='microsoft/deberta-base',
                                feature_type='GIA_text',
                                #feature_type="GIA",
                                #feature_type = f'attention-{emb_epoch}',
                                device=args.device,
                                use_text=False,
                            )
            
                            if args.dataset_name == 'ogbn-arxiv':
                                data_1.edge_index, _ = to_edge_index(data_1.edge_index)
                                data_2.edge_index, _ = to_edge_index(data_2.edge_index)
                            degrees_list = degree(data_1.edge_index[0], num_nodes=data_1.num_nodes).tolist()
                            data_2 = data_2.to(device)
                            
                            if 1 == 1:
                                candidate_lists = check_candidate_lists(args.dataset_name)
                                edges_to_del,edges_to_add = modified_edge_index_tensor(args.dataset_name, candidate_lists)
                                data_2.edge_index = remove_del_candidate_from_edges(data_2.edge_index,edges_to_del)
                            
                            outs_dir = os.path.join(
                                DATA_DIR, f"ssl/{loss_name}/{dataset_name}/"
                            )

                            emb_dir = os.path.join(outs_dir, "embeddings/")
                            os.makedirs(emb_dir, exist_ok=True)

                            model_dir = os.path.join(outs_dir, "models/")
                            os.makedirs(model_dir, exist_ok=True)

                            # Which model to use
                            if dataset_name == "ogbn-arxiv":
                                model_cls = ArxivModel
                            else:
                                model_cls = Model

                          
                            for i in tqdm(range(1), desc="Splits"):
                                model = model_cls(
                                    feature_dim=data_1.x.size(-1),
                                    emb_dim=params["emb_dim"],
                                    loss_name=loss_name,
                                    p_x_1=params["p_x_1"],
                                    p_e_1=params["p_e_1"],
                                    p_x_2=params["p_x_2"],
                                    p_e_2=params["p_e_2"],
                                    lr_base=params["lr_base"],
                                    total_epochs=params["total_epochs"],
                                    warmup_epochs=params["warmup_epochs"],
                                    edges_to_add = edges_to_add,
                                    edges_to_del = edges_to_del,
                                    degrees_list = degrees_list,
                                    add_p = params["add_p"],
                                    del_p = params["del_p"],
                                )

                                z1,z2 = model.fit(
                                    data_1=data_1,
                                    data_2=data_2,
                                )
                                # Save model
                                torch.save(obj=model, f=os.path.join(model_dir, f"{i}.pt"))

                                representations1 = model.predict(data_1)
                                z1.append(representations1)

                                representations2 = model.predict(data_2)
                                z2.append(representations2)
                                labels = data_1.y
                                if args.eval_multi_k:
                                    for feat in z1:
                                        multi_k_eval_1.multi_k_fit_logistic_regression_new(features=feat,labels=labels,
                                                                                        dataset_name=args.dataset_name,data_random_seeds=args.data_seeds,
                                                                                        device=args.device
                                                                                        )    
                                    for feat in z2:
                                        multi_k_eval_2.multi_k_fit_logistic_regression_new(features=feat,labels=labels,
                                                                                        dataset_name=args.dataset_name,data_random_seeds=args.data_seeds,
                                                                                        device=args.device
                                                                                        )    
                                                                         

                                if args.eval_multi_k:
                                    extra_name = '_attention'
                                    acc_1 = multi_k_eval_1.save_csv_results(dataset_name=args.dataset_name, experience_name="GraphCL" if loss_name=='GraphCL_loss' else "GBT",feature_type=args.feature_type+extra_name)
                                    second_part_1 = acc_1.split('/')[1]
                                    acc_number_str_1 = second_part_1.split('±')[0]
                                    acc_1 = float(acc_number_str_1)
                                    var_1 = float(second_part_1.split('±')[1])

                                    acc_2 = multi_k_eval_2.save_csv_results(dataset_name=args.dataset_name, experience_name="GraphCL" if loss_name=='GraphCL_loss' else "GBT",feature_type=args.feature_type+extra_name)
                                    second_part_2 = acc_2.split('/')[1]
                                    number_str_2 = second_part_2.split('±')[0]
                                    acc_2 = float(number_str_2)
                                    var_2 = float(second_part_2.split('±')[1])

                                    acc = max(acc_1,acc_2)
                                    if acc == acc_1:
                                        var = var_1
                                    else:
                                        var = var_2

                                    if acc > best_acc_search:
                                        best_acc_search = acc
                                        best_hyperparams = hyperparams
                                        var_with_best_acc = var
                                        print("best_hyperparams updated!")
                                    results["best_acc_search"] = best_acc_search
                                    results["best_hyperparams"] = best_hyperparams
                                    results["variance"] = var_with_best_acc
                                    # 输出结果到文件
                                    if args.loss_name == 'GraphCL_loss':
                                        output_file_name = 'GraphCL_' + args.dataset_name + "_new_attention_struct"+"_1.txt"
                                    elif args.loss_name == 'barlow_twins':
                                        output_file_name = 'GBT_' + args.dataset_name + "_new_attention_struct"+"_1.txt"
                                    with open(output_file_name, "w") as file:
                                        file.write("Best Accuracy: {}\n".format(results["best_acc_search"]))
                                        file.write("Variance with Best Acc: {}\n".format(results["variance"]))
                                        file.write("Best Hyperparameters:\n")
                                        for param, value in results["best_hyperparams"].items():
                                            file.write("{}: {}\n".format(param, value))
                                    print(f"Results saved to {output_file_name}")


if __name__ == "__main__":
    main()
