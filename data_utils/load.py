import os
import json
import torch
import csv
import numpy as np 
import dgl
import pandas as pd 
from data_utils.dataset import CustomDGLDataset
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import add_self_loops,remove_self_loops
from torch_geometric.data import Data
from pecos.utils import smat_util
from natsort import natsorted
import re

def bump(g):
    return Data.from_dict(g.__dict__)
            
def scale_feats(x):
    print("Scaling feat .....")
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

def load_gpt_preds(dataset, topk):
    preds = []
    with open(f'../gpt_preds/{dataset}.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            inner_list = []
            for value in row:
                inner_list.append(int(value))
            preds.append(inner_list)

    pl = torch.zeros(len(preds), topk, dtype=torch.long)
    for i, pred in enumerate(preds):
        pl[i][:len(pred)] = torch.tensor(pred[:topk], dtype=torch.long)+1
    return pl


def load_data(dataset, use_dgl=False, use_text=False, use_explanation=False, seed=0,use_generate_text=False,use_summery_text=False):
    if dataset == 'cora':
        from data_utils.load_cora import get_raw_text_cora as get_raw_text
        from data_utils.load_cora import add_generate_text_cora as add_generate_text
    elif dataset == 'pubmed':
        from data_utils.load_pubmed import get_raw_text_pubmed as get_raw_text
        from data_utils.load_pubmed import add_generate_text_pubmed as add_generate_text
        from data_utils.load_pubmed import add_summery_text_pubmed as add_summery_text
        
    elif dataset == 'ogbn-arxiv' or dataset == "arxiv":
        from data_utils.load_arxiv import get_raw_text_arxiv as get_raw_text
        from data_utils.load_arxiv import add_generate_text_arxiv as add_generate_text
    else:
        exit(f'Error: Dataset {dataset} not supported')

    # for training GNN
    if not use_text:
        data, _ = get_raw_text(use_text=False, seed=seed)
        if use_dgl:
            data = CustomDGLDataset(dataset, data)
        return data


        
    if use_text:
        data, text = get_raw_text(use_text=True, seed=seed) 
        if use_generate_text:
            add_generate_text(text)
        data.text = text
        if use_summery_text:
            add_summery_text(text)
    
    # use explanation 
    if use_explanation:
        folder_path = 'gpt_responses/{}'.format(dataset)
        print(f"use explanation: {folder_path}")
        n = data.y.shape[0]
        text = []
        for i in range(n):
            filename = str(i) + '.json'
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                content = json_data['choices'][0]['message']['content']
                data.text[i]+=content # append explation to text
                

    if use_dgl:
        data = CustomDGLDataset(dataset, data)
    return data

def load_amazon_data(dataset_name, feature_type, use_dgl, use_text=False,use_generate_text = False):
    # assert feature_type in ["BOW","W2V"], f"only BOW and W2V two kind of features in amazon data"
    if dataset_name == 'amazon-computers':
        '''
        csv_dir = 'C:/Users/YI/Desktop/dataset/computers/Computers_Final_with_W2V_embeddings.csv'
        data = build_data_from_csv(csv_dir)
        '''
        if feature_type=="BOW":
            data_path = '../dataset/computers/Computers_Final_with_BoW_embeddings.pt'
        else:
            data_path = '../dataset/computers/Computers_Final_with_BoW_embeddings.pt'
        data = torch.load(data_path)
    elif dataset_name == 'amazon-photo':
        '''
        csv_dir = 'C:/Users/YI/Desktop/dataset/photo/Photo_Final_with_W2V_embeddings.csv'
        data = build_data_from_csv(csv_dir)
        '''
        if feature_type=="BOW":
            data_path = '../dataset/photo/Photo_Final_with_BoW_embeddings.pt'
        else:
            data_path = '../dataset/photo/Photo_Final_with_BoW_embeddings.pt'
        data = torch.load(data_path)
    elif dataset_name == 'amazon-history':
        '''
        csv_dir = 'C:/Users/YI/Desktop/dataset/history/History_Final_with_BoW_embeddings.csv'
        data = build_data_from_csv(csv_dir)
        '''
        if feature_type=="BOW":
            data_path = '../dataset/history/History_Final_with_BoW_embeddings.pt'
        else:
            data_path = '../dataset/history/History_Final_with_BoW_embeddings.pt'

        #load old version pyg data
        data = torch.load(data_path)
        data = bump(data)
    else:
        assert False, "no such amazon dataset"
    
    #load old version pyg data
    data = torch.load(data_path)
    data = bump(data)
    
    if isinstance(data.x,torch.LongTensor):
        data.x=data.x.float() # special preprocess for amazon-photo-BOW

    if use_text:
        if dataset_name == 'amazon-computers':
            file_path = "../dataset/computers/Computers.csv"
        elif dataset_name == 'amazon-history':
            file_path = "../dataset/history/History_Final.csv"
        elif dataset_name == 'amazon-photo':
            file_path = "../dataset/photo/Photo_Final.csv"
        df = pd.read_csv(file_path)
        text = list(df["text"])
        data.text = text
    
    if use_generate_text:
        add_generate_amazon(data.text,dataset_name)
        
    if use_dgl:
        g = dgl.DGLGraph()
        edge_index = data.edge_index
        g.add_nodes(data.num_nodes)
        g.add_edges(edge_index[0], edge_index[1])
        g.ndata['feat'] = torch.FloatTensor(data.x)
        g.ndata['label'] = torch.LongTensor(data.y).squeeze()
        g.temp_edge_index = data.edge_index 
        if use_text:
            g.text = text 
        return g
    
    return data

def add_generate_amazon(text,dataset):
    # read GPT generated texts
    directory = f"../dataset/generated_texts/{dataset}"
    files = os.listdir(directory)
    files = natsorted(files)
    
    assert len(text) == len(files), "text length: {}, files length: {}".format(len(text), len(files))
    for i,file in enumerate(files):
        if file.endswith('.txt'):
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                text[i] = text[i] + first_line
        
def load_llm_feature_and_data(dataset_name, feature_type, use_dgl = False, LLM_feat_seed = 0, lm_model_name="microsoft/deberta-base",device = 0 , sclae_feat = False, use_text = False, use_explanation = False, use_generate_text=False, use_summery_text=False):
        '''
        args:
            feature_type: TA or E or P or Bow or Wov or ogb
            lm_model_name: "microsoft/deberta-base"
            device: gpu index 
        
        returns : 
        
        note : remove the seed for load_data since we will unify the split 
        
        '''
        # assert feature_type.upper() in ["TA","P","E","BOW","W2V","OGB","MLM",
        #                                 "TCL","BASE","TMLM","GIA","MLM_E","TCL_E","GIA_E",
        #                                 'MLM_FT_E','TCL_FT_E','GIA_FT_E','MLM_FT_G'], ValueError(feature_type)
        
        seed = LLM_feat_seed
        # ! Load data from ogb
        if use_generate_text:
            assert use_text, "use_generate_text must be used with use_text"
            
        if dataset_name in ('cora', 'pubmed', 'ogbn-arxiv', 'Cora', 'Pubmed','arxiv'):
            data = load_data(dataset_name, use_dgl=use_dgl, use_text=use_text, use_explanation = use_explanation,use_generate_text=use_generate_text,use_summery_text=use_summery_text)
        elif "amazon" in dataset_name:
            data = load_amazon_data(dataset_name, feature_type.upper(), use_dgl,use_text=use_text,use_generate_text=use_generate_text)
        else:
            raise ValueError(dataset_name)


        try:
            data=data[0]
        except:
            pass # self defined data skip this procedure
        
        if  use_dgl:
            num_nodes = data.num_nodes()
            data.x = data.ndata['feat'] # ref https://github.com/XiaoxinHe/TAPE/blob/241c93b735dcebbe2853414395c1559d5c2ce202/core/GNNs/dgl_gnn_trainer.py#L39C8-L39C8
            data = data.remove_self_loop().add_self_loop() # ! dgl add_self_loop will duplicate self_loop
        else:
            num_nodes = data.x.shape[0]
            num_classes = data.y.unique().size(0)
            data.y = data.y.squeeze()
            
            if "arxiv" not in dataset_name: # ogbn-arxiv already has self loop and is directed, do not support this 
                edge_index, _ = remove_self_loops(data.edge_index)
                data.edge_index,_ = add_self_loops(edge_index)# ! add self loop for pyg\dgl data 
            
        if sclae_feat:
            if feature_type == 'ogb': #"only scale original feature"
                data.x = scale_feats(data.x) # !the GraphMAE scaled feat '


        # ! Init gnn feature
        topk = 3 if dataset_name == 'pubmed' else 5
        if feature_type == 'ogb':
            print("Loading OGB features...")
            features = data.x

            
        elif feature_type in ["TCL","MLM","base","TMLM","MLM_E","TCL_E",
                              'MLM_FT_E','TCL_FT_E','MLM_FT_G','TCL_FT_G','MLM_FIX'
                              ]:
            print(f"Loading deberta {feature_type} feature....")
            LM_emb_path = f"../LLMs/prt_lm/{dataset_name}/deberta-{feature_type}.emb"
            # LM_emb_path = f"/mnt/home/tangxiaqiang/code/graph/LLMs-with-GSSL/LLMs/prt_lm/cora/deberta-MLM_40_epoch.emb"
            
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(num_nodes, 768)))
            ).to(torch.float32)
                        
        elif feature_type in ["GIA","GIA_E",'GIA_FT_E','GIA_FT_G','GIA_FT_S']:
            # GIA feature require special type of loading method 
            print(f"Loading deberta {feature_type} feature....")
            LM_emb_path = f"../LLMs/prt_lm/{dataset_name}/deberta-{feature_type}.emb"
            
            features=torch.from_numpy(smat_util.load_matrix(
                                    LM_emb_path).astype(np.float32))
            '''
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(num_nodes, 768)))
            ).to(torch.float32) 
            '''
            print(f"LM_emb_path: {LM_emb_path}")    
            print(features)
            

        elif feature_type in ["roberta","bert"]:
            print(f"Loading deberta {feature_type} feature....")
            LM_emb_path = f"../LLMs/prt_lm/{dataset_name}/{feature_type}-base-TCL.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(num_nodes, 768)))
            ).to(torch.float32) 
            
        elif "attention" in feature_type:
            emb_epoch = re.search(r'attention-(.*)', feature_type).group(1) if re.search(r'attention-(.*)', feature_type) else None
            print(f"Loading attention {emb_epoch} feature....")

            #LM_emb_path = f"../LLMs/checkpoints/{dataset_name}/1/similarity-embs.emb"
            LM_emb_path = f"../LLMs/prt_lm/{dataset_name}/mistral-embs.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            torch.cuda.init()
            features = torch.load(LM_emb_path).detach()
            print(features)

        elif feature_type == 'GIA_text':
            print(f"Loading {feature_type} feature....")
            LM_emb_path = f"../LLMs/prt_lm/{dataset_name}/{feature_type}.emb"
            
            features=torch.from_numpy(smat_util.load_matrix(LM_emb_path).astype(np.float32))

            print(f"LM_emb_path: {LM_emb_path}")    
            print(features)
            

        elif feature_type == 'TA':
            print("Loading pretrained LM features (title and abstract) ...")
            LM_emb_path = f"../prt_lm/{dataset_name}/{lm_model_name}-seed{seed}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(num_nodes, 768)))
            ).to(torch.float32)
        elif feature_type == 'E':
            print("Loading pretrained LM features (explanations) ...")
            LM_emb_path = f"../prt_lm/{dataset_name}2/{lm_model_name}-seed{seed}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(num_nodes, 768)))
            ).to(torch.float32)
        elif feature_type == 'P':
            print("Loading top-k prediction features ...")
            features = load_gpt_preds(dataset_name, topk)
            features=features.float()
        elif feature_type == 'BOW' or 'W2V':
            assert "amazon" in dataset_name
            print("Loading Amazon Dataset ...")
            features = data.x
        else:
            print(
                f'Feature type {feature_type} is not TAPE skip load LLM feature')
            features = data.x
            
        if use_dgl:
            data = data.to(device)
            features = features.to(device)
            data.ndata['feat'] = features
        else:
            data.x = features.to(device)  # to fit dgl , use to() instead of .cuda()
        data = data.to(device)
        
        return data
                                  
def emb2dataX(emb_file_path):
    emb = np.memmap(emb_file_path, mode='r', dtype=np.float16, shape=(19717, 768))
    return torch.from_numpy(emb).to(torch.float32)
