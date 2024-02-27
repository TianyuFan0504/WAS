import argparse
parser = argparse.ArgumentParser()
# about seed and basic info
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
# about dataset and dataloader
parser.add_argument('--input_data_dir', type=str, default='./molecule_datasets/')
parser.add_argument('--dataset', type=str, default='bace')
parser.add_argument('--num_workers', type=int, default=8)
# about training strategies
parser.add_argument('--split', type=str, default='scaffold')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_scale', type=float, default=10)
parser.add_argument('--decay', type=float, default=0)
# about molecule GNN
parser.add_argument('--gnn_type', type=str, default='gin')
parser.add_argument('--num_layer', type=int, default=5)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--dropout_ratio', type=float, default=0.5)
parser.add_argument('--graph_pooling', type=str, default='mean')
parser.add_argument('--JK', type=str, default='last')
parser.add_argument('--gnn_lr_scale', type=float, default=1)
# about loading and saving
parser.add_argument('--input_model_file', type=str, default='./teachers')
parser.add_argument('--output_model_dir', type=str, default='./student')
# about WAS
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--tau", type=float, default=1.0)
parser.add_argument("--num_teachers", type=int, default=100)
parser.add_argument("--model_dict", type=dict, default=0)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--m", type=float, default=0.9)
parser.add_argument("--step_size", type=int, default=100)
parser.add_argument("--gamma", type=float, default=0.001)

args = parser.parse_args()
param = args.__dict__

    
print('arguments\t', args)