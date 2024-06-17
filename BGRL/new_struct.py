import copy
import logging
import os
from absl import app
from absl import flags
import torch
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from torch_geometric.utils.sparse import to_edge_index
import json
from bgrl.transforms_new import get_graph_drop_transform_new,remove_del_candidate_from_edges
from bgrl import *
from bgrl import BGRL
import sys
sys.path.append("..")
from data_utils.dataset import check_candidate_lists,modified_edge_index_tensor,modify_edge_index_one_time,modify_edge_index_one_time_ratio
from data_utils.load import load_llm_feature_and_data
from data_utils.logistic_regression_eval import Multi_K_Shot_Evaluator
import cProfile
log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
flags.DEFINE_multi_integer('model_seeds', [0], 'Random seed used to generate train/val/test split.')
flags.DEFINE_multi_integer('data_seeds', [0,1,2,3,4], 'Random seed used to generate train/val/test split.')

# Dataset.
flags.DEFINE_enum('dataset', 'cora',
                  ['cora',  'pubmed','ogbn-arxiv','amazon-photo','amazon-computers',"amazon-history"],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')
flags.DEFINE_string('feature_type', 'TA', 'LLM feature type')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', [256], 'Conv layer sizes.')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')

# Training hyperparameters.
flags.DEFINE_integer('epochs', 500, 'The number of training epochs.')
flags.DEFINE_integer('device', 0, 'GPU index')

flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_integer('lr_warmup_epochs', 50, 'Warmup period for learning rate.')

# Augmentations.
flags.DEFINE_float('add_edge_p', 0., 'Probability of edge dropout 1.')
flags.DEFINE_float('drop_feat_p_1', 0., 'Probability of node feature dropout 1.')
flags.DEFINE_float('del_edge_p', 0., 'Probability of edge dropout 2.')
flags.DEFINE_float('drop_feat_p_2', 0., 'Probability of node feature dropout 2.')

# Logging and checkpoint.
flags.DEFINE_string('logdir', '/tmp', 'Where the checkpoint and logs are stored.')

# Evaluation
flags.DEFINE_integer('eval_epochs', 200, 'Evaluate every eval_epochs.')


#dataset setting
flags.DEFINE_integer('feat_aug', 0, '.')
flags.DEFINE_enum('struct_type', 'aug',
                  ['ori','mix','aug'],
                  'Which graph dataset to use.')

flags.DEFINE_bool('eval_multi_k',default=True,help='')

hyperparams_space = {
    'lr': [5e-4],
    'drop_feat_p_1': [0.0],
    'drop_feat_p_2': [0.0],
    'graph_encoder_layer': [[512]],
    'add_edge_p': [0.5],
    'del_edge_p': [0.5],
    'emb_epoch': [1]
}

results = {
    "best_acc_search": 0.0,
    "variance": 0.0,
    "best_hyperparams": {}
}

accs = []
def main(argv):
    #Grid Search
    global results
    global accs
    var_with_best_acc = 0.0
    best_acc_search = 0.0
    best_hyperparams = {}

    for lr in hyperparams_space['lr']:
        for drop_feat_p_1 in hyperparams_space['drop_feat_p_1']:
            for drop_feat_p_2 in hyperparams_space['drop_feat_p_2']:
                for add_edge_p in hyperparams_space['add_edge_p']:
                    for del_edge_p in hyperparams_space['del_edge_p']:
                        for graph_encoder_layer in hyperparams_space['graph_encoder_layer']:
                            for emb_epoch in hyperparams_space['emb_epoch']:
                                hyperparams = {
                                    'lr': lr,
                                    'drop_feat_p_1': drop_feat_p_1,
                                    'drop_feat_p_2': drop_feat_p_2,
                                    'graph_encoder_layer': graph_encoder_layer,
                                    'add_edge_p': add_edge_p,
                                    'del_edge_p': del_edge_p,
                                    'emb_epoch':emb_epoch,
                                }
                                FLAGS.lr = lr
                                FLAGS.drop_feat_p_1 = drop_feat_p_1
                                FLAGS.drop_feat_p_2 = drop_feat_p_2
                                FLAGS.graph_encoder_layer = graph_encoder_layer
                                FLAGS.add_edge_p = add_edge_p
                                FLAGS.del_edge_p = del_edge_p

                                



                                eval_multi_k = FLAGS.eval_multi_k
                                for model_seed in FLAGS.model_seeds:
                                    # use CUDA_VISIBLE_DEVICES to select gpu
                                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                                    log.info('Using {} for training.'.format(device))

                                    # set random seed
                                    if model_seed is not None:
                                        log.info('Random seed set to {}.'.format(model_seed))
                                        set_random_seeds(random_seed=model_seed)


                                    if eval_multi_k:
                                        multi_k_eval_1=Multi_K_Shot_Evaluator(result_path='/LLM4GCL_update/results_1.csv')
                                        multi_k_eval_2=Multi_K_Shot_Evaluator(result_path='/LLM4GCL_update/results_2.csv')

                                    
                                    # load data
                                    dataset = load_llm_feature_and_data(
                                        dataset_name = FLAGS.dataset,
                                        lm_model_name='microsoft/deberta-base',
                                        #feature_type=FLAGS.feature_type,
                                        feature_type = 'GIA',
                                        device=device,
                                        use_text=True,
                                        )

                                    data = dataset

                                    # Load LLM
                                    text = data.text
                                    labels = data.y.tolist()
                                    seed = 0

                                    #data.x = data.x.float()

                                    data1 = data.to(device)  # permanently move in gpy memory
                                    data2 = load_llm_feature_and_data(
                                        dataset_name = FLAGS.dataset,
                                        lm_model_name='microsoft/deberta-base',
                                        #feature_type='GIA_FT_E',
                                        #feature_type = 'GIA',
                                        feature_type=FLAGS.feature_type,
                                        device=device,
                                        use_text=True,
                                        )
                                    #print(data2.x)
                                    if FLAGS.dataset == 'ogbn-arxiv':
                                        data1.edge_index,_ = to_edge_index(data1.edge_index)
                                        data2.edge_index,_ = to_edge_index(data2.edge_index)
                                    
                                    if FLAGS.struct_type == 'aug':
                                        candidate_lists = check_candidate_lists(FLAGS.dataset)
                                        edges_to_del,edges_to_add = modified_edge_index_tensor(FLAGS.dataset, candidate_lists)
                                        data2.edge_index = remove_del_candidate_from_edges(data2.edge_index,edges_to_del)
                                    '''
                                    candidate_lists = check_candidate_lists(FLAGS.dataset)
                                    edge_index = data1.edge_index
                                    edge_index_aug = modify_edge_index_one_time_ratio(FLAGS.dataset,edge_index,candidate_lists,ratio=FLAGS.add_edge_p)
                                    data1.edge_index = edge_index_aug.to(FLAGS.device)
                                    data2.edge_index = edge_index_aug.to(FLAGS.device)
                                    '''
                                    #view 1 only do feature ratio aug
                                    transform_1 = get_graph_drop_transform(drop_edge_p=0.0, drop_feat_p=FLAGS.drop_feat_p_1)
                                    #view 2 new struct aug
                                    #transform_2 = get_graph_drop_transform(drop_edge_p=0.2, drop_feat_p=FLAGS.drop_feat_p_2)
                                    transform_2 = get_graph_drop_transform_new(del_edge_p= FLAGS.del_edge_p, add_edge_p = FLAGS.del_edge_p, drop_feat_p = FLAGS.drop_feat_p_2, del_tensor = edges_to_del, add_tensor = edges_to_add)


                                    # build networks
                                    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
                                    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True)   # 512, 256, 128
                                    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
                                    model: BGRL = BGRL(encoder, predictor).to(device)

                                    # optimizer
                                    # Combined trainable parameters from both models
                                    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

                                    # scheduler
                                    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
                                    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

                                    # setup tensorboard and make custom layout
                                    writer = SummaryWriter(FLAGS.logdir)
                                    #layout = {'accuracy': {'accuracy/test': ['Multiline', [f'accuracy/test_{i}' for i in range(num_eval_splits)]]}}
                                    #writer.add_custom_scalars(layout)

                                    def train(step):
                                        model.train()

                                        # update learning rate
                                        lr = lr_scheduler.get(step)
                                        for param_group in optimizer.param_groups:
                                            param_group['lr'] = lr

                                        # update momentum
                                        mm = 1 - mm_scheduler.get(step)

                                        # forward
                                        optimizer.zero_grad()
                                    
                                        #在这里控制view
                                        #x1, x2 = transform_1(data1).to(device), transform_2(data2).to(device)
                                        x1 = transform_1(data1).to(device)
                                        x2 = transform_2(data2).to(device)

                                        q1, y2 = model(x1, x2)
                                        q2, y1 = model(x2, x1)

                                        loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()

                                        loss.backward()

                                        # update online network
                                        optimizer.step()
                                        # update target network
                                        model.update_target_network(mm)

                                        # log scalars
                                        writer.add_scalar('params/lr', lr, step)
                                        writer.add_scalar('params/mm', mm, step)
                                        writer.add_scalar('train/loss', loss, step)

                                    def eval(epoch):
                                        # make temporary copy of encoder
                                        tmp_encoder = copy.deepcopy(model.online_encoder).eval()
                                        #######
                                        #这里可以试试改成data2
                                        #######
                                        representations1, labels1 = compute_representations(tmp_encoder, data1, device)
                                        representations2, labels2 = compute_representations(tmp_encoder, data2, device)

                                        # evaluate
                                        if eval_multi_k:
                                            multi_k_eval_1.multi_k_fit_logistic_regression_new(features=representations1, labels=labels1,
                                                                                                data_random_seeds=FLAGS.data_seeds,
                                                                                                dataset_name=FLAGS.dataset, device=device)
                                            multi_k_eval_2.multi_k_fit_logistic_regression_new(features=representations2, labels=labels2,
                                                                                                data_random_seeds=FLAGS.data_seeds,
                                                                                                dataset_name=FLAGS.dataset, device=device)

                                    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
                                        train(epoch - 1)
                                        if epoch % FLAGS.eval_epochs == 0:
                                            eval(epoch)

                                    if eval_multi_k:
                                        eval(99999)
                                        extra_name = "_ChatGAug"

                                        acc_1 = multi_k_eval_1.save_csv_results(dataset_name=FLAGS.dataset, experience_name="BGRL",
                                                                        feature_type=FLAGS.feature_type + extra_name)
                                        second_part_1 = acc_1.split('/')[1]
                                        number_str_1 = second_part_1.split('±')[0]
                                        acc_1 = float(number_str_1)
                                        var_1 = float(second_part_1.split('±')[1])

                                        acc_2 = multi_k_eval_2.save_csv_results(dataset_name=FLAGS.dataset, experience_name="BGRL",
                                                                        feature_type=FLAGS.feature_type + extra_name)
                                        second_part_2 = acc_2.split('/')[1]
                                        number_str_2 = second_part_2.split('±')[0]
                                        acc_2 = float(number_str_2)
                                        var_2 = float(second_part_2.split('±')[1])

                                        acc = max(acc_1,acc_2)

                                        if acc == acc_1:
                                            var = var_1
                                        else:
                                            var = var_2
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
                                        output_file_name = "BGRL_" + FLAGS.dataset + "_new_attention_struct_2"+".txt"
                                        with open(output_file_name, "w") as file:
                                            file.write("Best Accuracy: {}\n".format(results["best_acc_search"]))
                                            file.write("Variance with Best Acc: {}\n".format(results["variance"]))
                                            file.write("Best Hyperparameters:\n")
                                            for param, value in results["best_hyperparams"].items():
                                                file.write("{}: {}\n".format(param, value))

                                        print(f"Results saved to {output_file_name}")

                    
        print(accs)
if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)

