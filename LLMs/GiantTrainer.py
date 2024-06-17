import os
import sys
import copy
import numpy as np
import scipy.sparse as smat
from pecos.utils import smat_util
from pecos.utils.featurization.text.vectorizers import Vectorizer
import torch
from torch_geometric.utils import degree, is_undirected, to_undirected
from torch_geometric.utils.convert import to_scipy_sparse_matrix
#from pecos.xmc.xtransformer.model import XTransformer
from modelori import XTransformer
from pecos.xmc.xtransformer.module import MLProblemWithText
from pecos.utils.featurization.text.preprocess import Preprocessor
sys.path.append("..")

from LLMs.utils import time_logger
from LLMs.utils import init_path, time_logger

import argparse
from torch_geometric.utils.sparse import to_edge_index

from data_utils.load import load_llm_feature_and_data

    
class GiantTrainer():
    def __init__(self,data,text,filename,device):
        self.data = data
        self.text = text
        self.dataset = filename
        self.device = device
        self.max_deg = 1000
        
        self.output_dir = f'output/{self.dataset}/GIA_SAS' # use related dir 
        self.ckpt_dir = f'prt_lm/{self.dataset}/GIA_SAS'

        xrt_data_dir = './proc_data_xrt'
        self.save_data_dir = os.path.join(xrt_data_dir, self.dataset)
        os.makedirs(self.save_data_dir, exist_ok=True)
        edge_index = self.data.edge_index
        # Make sure edge_index is undirected!!!
        if not is_undirected(edge_index):
            edge_index = to_undirected(edge_index)
        # Filtering nodes whose number of edges >= max_degree
        Degree = degree(edge_index[0])
        Filtered_idx = torch.where(Degree < self.max_deg)[0]
        print('Number of original nodes:{}'.format(data.x.shape[0]))
        print('Number of filtered nodes:{}'.format(len(Filtered_idx)))

        # # Construct and save label matrix (adjacencey matrix) Y.
        Y_csr_all = smat.csr_matrix(to_scipy_sparse_matrix(edge_index))
        Y_csr_trn = Y_csr_all[Filtered_idx.cpu()]
        smat_util.save_matrix(f"{self.save_data_dir}/Y.trn.npz", Y_csr_trn)
        smat_util.save_matrix(f"{self.save_data_dir}/Y.all.npz", Y_csr_all)
        print("Saved Y.trn.npz and Y.all.npz")

        # Apply the same filtering for raw text
        sanitized_lines = [line.replace('\n', ' ') for line in self.text]
        self.text = sanitized_lines
        
        node_text_list = copy.deepcopy(self.text)
        print("|node_text_list={}".format(len(node_text_list)))
        count = 0
        with open(f"{self.save_data_dir}/X.trn.txt", "w") as fout:
            for cur_idx, line in enumerate(node_text_list):
                if Filtered_idx[count].item() == cur_idx:
                    fout.writelines(line+"\n")
                    count += 1
        assert count == len(Filtered_idx), "count={}, len(Filtered_idx)={}".format(count, len(Filtered_idx))
        print("Saved X.trn.txt")
        


        # Apply the same filtering for tfidf features
        parser = argparse.ArgumentParser(description='Prepare data for Giant-XRT')
        parser.add_argument('--vectorizer-config-path', type=str, default=f"./proc_data_xrt/vect_config.json",
                            help="a path to a json file that specify the tfidf hyper-paramters")
        
        # avoid conflict with argparse from trainModel.py 
        parser.add_argument('--dataset', type=str, default='cora')  
        parser.add_argument('--LLM_type', type=str, default='cora')
        
        # parser.add_argument('--flagfile', type=str) 
        # parser.add_argument('--feature_type', type=str) 
        # parser.add_argument('--LLM_type', type=str) 
        # parser.add_argument('--eval_epochs', type=str) 
        # parser.add_argument("--eval_multi_k", action="store_true", default=False)
        # parser.add_argument("--fine_tune_LM", action="store_true", default=False)
        
        
        
        
        
        args = parser.parse_args()

        vectorizer_config = Vectorizer.load_config_from_args(args)  # usiang args.vectorizer_config_pth
        preprocessor = Preprocessor.train(node_text_list, vectorizer_config, dtype=np.float32)
        preprocessor.save(f"{self.save_data_dir}/tfidf-model")
        X_tfidf_all = preprocessor.predict(node_text_list)
        X_tfidf_trn = X_tfidf_all[Filtered_idx.cpu()]
        smat_util.save_matrix(f"{self.save_data_dir}/X.all.tfidf.npz", X_tfidf_all)
        smat_util.save_matrix(f"{self.save_data_dir}/X.trn.tfidf.npz", X_tfidf_trn)
        print("Saved X.trn.npz and X.all.npz")

        X = smat_util.load_matrix(f"{self.save_data_dir}/X.trn.tfidf.npz")
        Y = smat_util.load_matrix(f"{self.save_data_dir}/Y.trn.npz")

        # also filter the raw text 
        Filtered_idx_list = Filtered_idx.tolist()
        node_text_list= [node_text_list[i] for i in Filtered_idx_list]
        # load training text features
        self.prob = MLProblemWithText(node_text_list, Y, X_feat=X)

    def train(self):
        import json
        with open("proc_data_xrt/params.json") as f:
            params=json.load(f)
        
        xtf = XTransformer.train(prob=self.prob,train_params=params["train_params"],pred_params=params["pred_params"])
        model_dir = os.path.join(self.save_data_dir, "model")
        xtf.save(model_dir)
        xtf = XTransformer.load(model_dir)
        self.model = xtf

    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        
        emb = self.model.encode(self.text)
        
        smat_util.save_matrix(f"{self.ckpt_dir}.emb", emb)
        
        return emb



def get_text_and_template(text, dataset_name, prompt_type):
    n = len(text)
    directory = f"../aug_data/Node_Feature/{prompt_type}/{dataset_name}"
    texts = []
    for i in range(n):
        file = f'generated_{i}.txt'
        if file.endswith('.txt'):
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                response = f.read().strip()
                texts.append(response)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.cuda.empty_cache()
    parser.add_argument('--dataset', type=str, default='cora')
    args = parser.parse_args()
    
    dataset_name = args.dataset

    dataset = load_llm_feature_and_data(dataset_name,feature_type="BOW" if 'amazon' in args.dataset else 'ogb' ,use_text=True,device=device)

    data = dataset
    text = data.text
    labels = data.y.tolist()

    seed = 0
    
    if "arxiv" in dataset_name:
        data.edge_index,_ = to_edge_index(data.edge_index)
    
    # first_neighbor,random_neighbor, all_neighbors = get_neighbors_list(data.edge_index)
    prompt_type = 'SAS'
    text = get_text_and_template(text,dataset_name,prompt_type)

    trainer = GiantTrainer(data,text,dataset_name, device)
    
    trainer.train()
    emb = trainer.eval_and_save()
    print(emb.shape)

