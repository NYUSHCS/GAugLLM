import os
import sys
import copy
import numpy as np
import scipy.sparse as smat
from pecos.utils import smat_util
from pecos.utils.featurization.text.vectorizers import Vectorizer
import torch
from natsort import natsorted
from data_utils.dataset import check_candidate_lists,modify_edge_index_one_time
from torch_geometric.utils import degree, is_undirected, to_undirected
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from pubmedModel import CustomXTransformer
from pecos.xmc.xtransformer.module import MLProblemWithText
from pecos.utils.featurization.text.preprocess import Preprocessor
sys.path.append("..")
import torch.nn.functional as F
import torch.nn as nn
from LLMs.utils import init_path, time_logger
from utils import get_neighbors_list
import argparse
import time
from torch_geometric.utils.sparse import to_edge_index
from scipy.sparse import csr_matrix
from data_utils.load import load_llm_feature_and_data
import json

def get_text_and_template(degree_list, dataset_name, prompt_type):
    # Define thresholds for tail and head nodes
    def find_thresholds(degree_list, lower_percentile, upper_percentile):
        t = np.percentile(degree_list, lower_percentile * 100)
        h = np.percentile(degree_list, (1 - upper_percentile) * 100)
        return t, h

    # Calculate thresholds
    t, h = find_thresholds(degree_list, 0.40, 0.20)

    # read GPT generated texts
    directory = f"../prompts/new_data/{dataset_name}/{prompt_type}-LLaMA/generated_texts"
    files = os.listdir(directory)
    files = natsorted(files)
    texts = []
    templates = []
    
    if prompt_type == "IDR":
        prompt = "This is the explanation for classification based on the original text of this node."
    elif prompt_type == "SAS":
        prompt = "This is the summarization of the original text with the understanding of its neighboring nodes."
    elif prompt_type == "SAR":
        prompt = "This is the explanation for classification based on the original text with the understanding of its neighboring nodes."

    n = len(degree_list)
    for i in range(n):
        file = f'generated_{i}.txt'
        if degree_list[i] <= t:
            node_info = f"The degree of this node is {int(degree_list[i])}. We consider nodes with degree less than or equal to {int(t)} as tail degree nodes. These nodes have relatively poor information from their linked nodes."
        elif degree_list[i] >= h:
            node_info = f"The degree of this node is {int(degree_list[i])}. We consider nodes with degree greater than or equal to {int(h)} as head degree nodes. These nodes have relatively rich structure information from their linked nodes."
        else:
            node_info = f"The degree of this node is {int(degree_list[i])}. Nodes with degrees between {int(t)+1} and {int(h)-1} are considered as mid-degree nodes."
            
        if file.endswith('.txt'):
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                response = f.read().strip()
                template = prompt + node_info
                texts.append(response)
                templates.append(template)
    print(texts[0])
    print(templates[0])
    return texts, templates

    
class GiantTrainer_new():
    def __init__(self,data,text,filename,device):
        self.data = data
        dataset_name = filename
        self.text = text
        self.dataset = filename
        self.device = device
        self.max_deg = 100000
        
        self.output_dir = f'output/{self.dataset}/debert-attention-train' # use related dir 
        self.ckpt_dir = f'prt_lm/{self.dataset}/debert-attention-train'
        self.checkpoints_dir = f'checkpoints/pubmed_LLaMA/'
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

        ######################### 
        number_of_nodes = len(self.data.y)
        degrees_list = degree(self.data.edge_index[0], num_nodes=number_of_nodes).tolist()
        with open('/LLM4GCL_update/LLMs/prt_lm/pubmed_LLaMA/Aug_degrees_list.json', 'w') as f:
            json.dump(degrees_list, f)
        print("degrees_list saved")

        text_ORI = self.text
        template_ORI = []
        def find_thresholds(degrees_list, lower_percentile, upper_percentile):
            t = np.percentile(degrees_list, lower_percentile * 100)
            h = np.percentile(degrees_list, (1 - upper_percentile) * 100)
            return t, h
        # Calculate thresholds
        t, h = find_thresholds(degrees_list, 0.40, 0.20)
        for i in range(len(text_ORI)):
            prompt = "This is the original text of this node."
            if degrees_list[i] <= t:
                node_info = f"The degree of this node is {int(degrees_list[i])}. We consider nodes with degree less than or equal to {int(t)} as tail degree nodes. These nodes have relatively poor information from their linked nodes."
            elif degrees_list[i] >= h:
                node_info = f"The degree of this node is {int(degrees_list[i])}. We consider nodes with degree greater than or equal to {int(h)} as head degree nodes. These nodes have relatively rich structure information from their linked nodes."
            else:
                node_info = f"The degree of this node is {int(degrees_list[i])}. Nodes with degrees between {int(t)+1} and {int(h)-1} are considered as mid-degree nodes."
            template_ORI.append(prompt + node_info)

        text_IDR,template_IDR = get_text_and_template(degrees_list,dataset_name,"IDR")
        text_SAS,template_SAS = get_text_and_template(degrees_list,dataset_name,"SAS")
        text_SAR,template_SAR = get_text_and_template(degrees_list,dataset_name,"SAR")
    
        prompt_types = ["ORI", "IDR", "SAS", "SAR"]
        texts = [text_ORI, text_IDR, text_SAS, text_SAR]  
        templates = [template_ORI,template_IDR,template_SAS,template_SAR]
        probs = []

        def text_process(text):
            return [line.replace('\n', ' ') for line in text]
        
        
        for i in range(4):
            probs_text = MLProblemWithText(text_process(texts[i]), Y, X_feat=preprocessor.predict(texts[i]))
            probs_template = MLProblemWithText(text_process(templates[i]),Y, X_feat=preprocessor.predict(templates[i]))
            probs.append(probs_text)
            probs.append(probs_template)
        
        '''
        # To test the pipeline with 8 same original text, should result in MLP weights all near 0.25
        prob_z = MLProblemWithText(text_process(texts[0]), Y, X_feat=X)
        for i in range(8):
            probs.append(prob_z)
        '''
        self.probs = probs


        
    

    def train(self):
        with open("proc_data_xrt/params.json") as f:
            params=json.load(f)
        
        xtf = CustomXTransformer.train(probs=self.probs,train_params=params["train_params"],pred_params=params["pred_params"])
        model_dir = os.path.join(self.save_data_dir, "model")
        #xtf.save(model_dir)
        #xtf = CustomXTransformer.load(model_dir)
        #self.model = xtf

    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        texts_list = [self.probs[i].X_text for i in [0,2,4,6]]
        for epoch_index in range(1,6):
            model_dir = self.checkpoints_dir + f'{epoch_index}'
            print("Trainer",model_dir)
            model = CustomXTransformer.load(model_dir)
            emb = model.encode(texts_list,epoch_index = epoch_index)
            #smat_util.save_matrix(f"{model_dir}/{epoch_index}.emb", emb)

        return emb




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

    #用新的augmented structure进行GIANT
    candidate_lists,candidate_neighbors_lists,_,_ = check_candidate_lists(dataset_name)
    new_edge_index = modify_edge_index_one_time(dataset.edge_index, candidate_lists, candidate_neighbors_lists)
    dataset.edge_index = new_edge_index
    data = dataset
    text = data.text
    labels = data.y.tolist()

    seed = 0
    
    if "arxiv" in dataset_name:
        data.edge_index,_ = to_edge_index(data.edge_index)
    
    # first_neighbor,random_neighbor, all_neighbors = get_neighbors_list(data.edge_index)

    text_c = text
    trainer = GiantTrainer(data,text,dataset_name, device)
    
    trainer.train()
    emb = trainer.eval_and_save()
    print(emb.shape)

