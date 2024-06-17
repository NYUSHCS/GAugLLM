from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import os
import torch
import pandas as pd
from natsort import natsorted



def get_raw_text_arxiv(use_text=False, seed=0):

    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv', transform=T.ToSparseTensor())
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        '../dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')

    raw_text = pd.read_csv('../dataset/ogbn_arxiv_orig/titleabs.tsv',
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])

    nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].astype(str)
    raw_text['paper id'] = raw_text['paper id'].astype(str)
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)
    return data, text


def add_generate_text_arxiv(text):
    # read GPT generated texts
    directory = "../dataset/generated_texts/ogbn-arxiv"
    files = os.listdir(directory)
    files = natsorted(files)
    
    assert len(text) == len(files), "text length: {}, files length: {}".format(len(text), len(files))
    for i,file in enumerate(files):
        if file.endswith('.txt'):
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                text[i] = text[i] + 'Summary:' + first_line
