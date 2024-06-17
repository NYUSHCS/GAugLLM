dataset=$1
device=$2
feature_type=$3

[ -z "${dataset}" ] && dataset="cora"
[ -z "${device}" ] && device= 0
[ -z "${feature_type}" ] 


python main_transductive.py \
	--feature_type $feature_type \
	--device 0 \
	--dataset $dataset \
	--mask_rate 0.5 \
	--encoder "gat" \
	--decoder "gat" \
	--in_drop 0.2 \
	--attn_drop 0.1 \
	--num_layers 2 \
	--num_hidden 512 \
	--num_heads 4 \
	--max_epoch 1500 \
	--max_epoch_f 300 \
	--lr 0.001 \
	--weight_decay 0 \
	--lr_f 0.01 \
	--weight_decay_f 1e-4 \
	--activation prelu \
	--optimizer adam \
	--drop_edge_rate 0.0 \
	--loss_fn "sce" \
	--replace_rate 0.05 \
	--alpha_l 3 \
	--linear_prob \
	--scheduler \
	--use_cfg \
	--eval_multi_k \
