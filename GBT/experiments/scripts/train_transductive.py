import json
import os
import sys
import yaml
from torch_geometric.utils.sparse import to_edge_index
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import argparse

import sys
sys.path.append("../../..") # TODO merge TAPE into current repo
from data_utils.load import load_llm_feature_and_data
import data_utils.logistic_regression_eval as evaluate

from GBT.gssl import DATA_DIR
from GBT.gssl.datasets import load_dataset
from GBT.gssl.transductive_model_arxiv_GBT import ArxivModel
from GBT.gssl.utils import seed
import sys
import data_utils.logistic_regression_eval as evaluate
from data_utils.logistic_regression_eval import Multi_K_Shot_Evaluator
sys.path.append("..") # TODO merge TAPE into current repo

def main():
    parser = argparse.ArgumentParser(description='GraphCL')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument("--feature_type", type=str, required=True)
    parser.add_argument("--eval_multi_k", action="store_true")
    parser.add_argument("--use_LLM_emb", action="store_true")
    parser.add_argument('--data_seeds', type=list, default=[0,1,2,3,4])
    parser.add_argument('--loss_name', type=str, default='GraphCL_loss')
    
    args = parser.parse_args()
    print(args)

    if args.eval_multi_k:
        multi_k_eval=Multi_K_Shot_Evaluator(result_path='/LLM4GCL/results.csv')
        
    seed()

    # Read dataset name
    #dataset_name = sys.argv[1]
    dataset_name = args.dataset
    loss_name = args.loss_name
    
    if loss_name == 'GraphCL_loss':
        from GBT.gssl.transductive_model_ori import Model
        from GBT.gssl.transductive_model_arxiv_GBT import ArxivModel
        
    elif loss_name == 'barlow_twins':
        from GBT.gssl.transductive_model_ori import Model
        from GBT.gssl.transductive_model_arxiv_GBT import ArxivModel
        
    else:
        raise NotImplementedError
    
    if args.eval_multi_k:
        multi_k_eval=Multi_K_Shot_Evaluator()

    # Read params
    with open("../configs/train_transductive.yaml", "r") as fin:
        params = yaml.safe_load(fin)[loss_name][dataset_name]

    # data, masks = load_dataset(name=dataset_name)
    # load data

    import os
    # os.chdir('/LLMs-with-GSSL/dataset')
    #sys.path.append('/mnt/home/tangxiaqiang/code/graph/LLMs-with-GSSL/GBT')
    sys.path.append('/LLM4GCL_update')
    os.chdir('/LLM4GCL_update/GBT')
    dataset = load_llm_feature_and_data(
        dataset_name=args.dataset,
        lm_model_name='microsoft/deberta-base',
        feature_type=args.feature_type,
        device=args.device,
        use_text=False,
    )



    if args.dataset == 'ogbn-arxiv':
        dataset.edge_index, _ = to_edge_index(dataset.edge_index)

    data = dataset
    masks = None

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

    log_epochs = None
    final_acc_list = []
    early_stp_acc_list = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    #20 改为1
    for i in tqdm(range(1), desc="Splits"):
        logger = SummaryWriter(log_dir=os.path.join(outs_dir, f"logs/{i}"))

        model = model_cls(
            feature_dim=data.x.size(-1),
            emb_dim=params["emb_dim"],
            loss_name=loss_name,
            p_x=params["p_x"],
            p_e=params["p_e"],
            lr_base=params["lr_base"],
            total_epochs=params["total_epochs"],
            warmup_epochs=params["warmup_epochs"],
        )

        logs,z = model.fit(
            data=data,
            logger=logger,
            log_interval=params["log_interval"],
            masks=None,
        )

        log_epochs = logs["log_epoch"]
        train_accuracies.append(logs["train_accuracies"])
        val_accuracies.append(logs["val_accuracies"])
        test_accuracies.append(logs["test_accuracies"])

        # Save model
        torch.save(obj=model, f=os.path.join(model_dir, f"{i}.pt"))

        representations = model.predict(data)
        z.append(representations)
        assert len(z) == 11 , "z should have 11 , since we save many representation"
        labels = data.y
        if args.eval_multi_k:
            for feat in z:
                multi_k_eval.multi_k_fit_logistic_regression_new(features=feat,labels=labels,
                                                                dataset_name=args.dataset,data_random_seeds=args.data_seeds,
                                                                device=args.device
                                                                )
        else:
            final_acc, early_stp_acc = evaluate.fit_logistic_regression_new(features=representations, labels=labels,
                                                                            data_random_seeds=args.data_seeds,
                                                                            dataset_name=args.dataset, device=args.device,
                                                                            mute=True)
            final_acc_list.extend(final_acc)
            early_stp_acc_list.extend(early_stp_acc)

    if args.eval_multi_k:
        multi_k_eval.save_csv_results(dataset_name=args.dataset,experience_name="GraphCL" if loss_name=='GraphCL_loss' else "GBT",feature_type=args.feature_type)
    else:
        final_acc, final_acc_std = np.mean(final_acc_list), np.std(final_acc_list)
        estp_acc, estp_acc_std = np.mean(early_stp_acc_list), np.std(early_stp_acc_list)
        print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
        print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")

if __name__ == "__main__":
    main()
