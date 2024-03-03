import argparse

parser = argparse.ArgumentParser()

# about seed and basic info
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)

# about dataset and dataloader
parser.add_argument('--input_data_dir', type=str, default='')
parser.add_argument('--dataset', type=str, default='bace')
parser.add_argument('--num_workers', type=int, default=8)

# about training strategies
parser.add_argument('--split', type=str, default='scaffold')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_scale', type=float, default=1)
parser.add_argument('--decay', type=float, default=0)

# about molecule GNN
parser.add_argument('--gnn_type', type=str, default='gin')
parser.add_argument('--num_layer', type=int, default=5)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--dropout_ratio', type=float, default=0.5)
parser.add_argument('--graph_pooling', type=str, default='mean')
parser.add_argument('--JK', type=str, default='last')
parser.add_argument('--gnn_lr_scale', type=float, default=1)
parser.add_argument('--model_3d', type=str, default='schnet', choices=['schnet'])

# for AttributeMask
parser.add_argument('--mask_rate', type=float, default=0.15)
parser.add_argument('--mask_edge', type=int, default=0)

# for ContextPred
parser.add_argument('--csize', type=int, default=3)
parser.add_argument('--contextpred_neg_samples', type=int, default=1)

# about if we would print out eval metric for training data
parser.add_argument('--eval_train', dest='eval_train', action='store_true')
parser.add_argument('--no_eval_train', dest='eval_train', action='store_false')
parser.set_defaults(eval_train=True)

# about loading and saving
parser.add_argument('--input_model_file', type=str, default='')
parser.add_argument('--output_model_dir', type=str, default='')

# verbosity
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--no_verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=False)

args = parser.parse_args()
print('arguments\t', args)
