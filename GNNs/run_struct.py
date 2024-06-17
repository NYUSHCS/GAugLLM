import itertools
import subprocess


def run_GNNs_script(dataset_list, model_type_list, feature_type_list, epoch_list, k_shot_list, lr_list, weight_decay_list,dim_hidden_list,number_of_layers_list,edge_ratio_list):
    # 生成所有可能的参数组合
    all_combinations = itertools.product(dataset_list, model_type_list, feature_type_list, epoch_list, k_shot_list,
                                         lr_list, weight_decay_list, dim_hidden_list,number_of_layers_list,edge_ratio_list)
    count = 0
    for combination in all_combinations:
        count += 1
        dataset, model_type, feature_type, epoch, k_shot,lr,weight_decay,dim_hidden,number_of_layers,edge_ratio = combination

        # 构建指令
        cmd = f"python GNNs_struct.py --dataset {dataset} --model_type {model_type} --feature_type {feature_type} --epoch {epoch} --k_shot {k_shot} --lr {lr} --weight_decay {weight_decay} --dim_hidden {dim_hidden} --num_layers {number_of_layers} --edge_ratio {edge_ratio}"

        print(f"Executing: {cmd}")

        # 执行指令
        subprocess.run(cmd, shell=True)


# 参数列表
dataset_list = ["pubmed"]
dim_hidden_list = [128]
model_type_list = ["GCN"]
feature_type_list = ["attention"]
number_of_layers_list = [2]
epoch_list = [200]
lr_list = [0.0005]
weight_decay_list = [0.00001]
k_shot_list = [20]
edge_ratio_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]



# 执行脚本
run_GNNs_script(dataset_list, model_type_list, feature_type_list, epoch_list, k_shot_list, lr_list, weight_decay_list, dim_hidden_list,number_of_layers_list,edge_ratio_list)
