import os
import sys
sys.path.append(os.getcwd().split('src')[0])
from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import time
import argparse
import numpy as np
from torch_geometric.data import DataLoader
from src.model.gcn import GCN
from src.data_util.rna_family_graph_dataset import RNAFamilyGraphDataset
from src.util.visualization_util import plot_loss
from src.data_util.data_constants import word_to_ix, families
from src.evaluation.evaluation_util import evaluate_family_classifier, compute_metrics_family

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="test", help='model name')
parser.add_argument('--device', default="cuda", help='cpu or cuda')
parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to train on')
parser.add_argument('--n_epochs', type=int, default=10000, help='Number of samples to train on')
parser.add_argument('--embedding_dim', type=int, default=20, help='Dimension of nucleotide '
                                                                  'embeddings')
parser.add_argument('--seq_max_len', type=int, default=200, help='Maximum length of sequences '
                                                                 'used for training and testing')
parser.add_argument('--seq_min_len', type=int, default=1, help='Maximum length of sequences '                                                                 'used for training and testing')
parser.add_argument('--n_conv_layers', type=int, default=5, help='Number of convolutional layers')
parser.add_argument('--conv_type', type=str, default="MPNN", help='Type of convolutional layers')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--batch_norm', dest='batch_norm', action='store_true')
parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false')
parser.set_defaults(batch_norm=True)
parser.add_argument('--residuals', type=bool, default=False, help='Whether to use residuals')
parser.add_argument('--set2set_pooling', type=bool, default=True, help='Whether to use set2set '
                                                                        'pooling')
parser.add_argument('--early_stopping', type=int, default=30, help='Number of epochs for early '
                                                                   'stopping')
parser.add_argument('--verbose', type=bool, default=False, help='Verbosity')
parser.add_argument('--foldings_dataset', type=str,
                    default='../data/GM12878_200bp_fold.pkl', help='Path to foldings')
parser.add_argument('--train_dataset', type=str,
                    default='../data/GM12878_200bp_pos+neg_train.fa', help='Path to training '
                                                                          'dataset')
parser.add_argument('--val_dataset', type=str,
                    default='../data/GM12878_200bp_pos+neg_val.fa', help='Path to val dataset')

opt = parser.parse_args()
print(opt)

n_classes = len(families)

def create_model(hidden_dim,n_conv_layers,dropout,learning_rate):
    
    model = GCN(n_features=opt.embedding_dim, hidden_dim=hidden_dim, n_classes=n_classes,
            n_conv_layers=n_conv_layers,
            dropout=dropout, 
            batch_norm=opt.batch_norm,
            num_embeddings=len(word_to_ix),
            embedding_dim=opt.embedding_dim,
            node_classification=False,
            residuals=opt.residuals, device=opt.device,
            set2set_pooling=opt.set2set_pooling).to(opt.device)
    return model

def train_epoch(model, train_loader,loss_function,optimizer):
    model.train()
    losses = []
    accuracies = []

    for batch_idx, data in enumerate(train_loader):
        data.x = data.x.to(opt.device)
        data.edge_index = data.edge_index.to(opt.device)
        data.edge_attr = data.edge_attr.to(opt.device)
        data.batch = data.batch.to(opt.device)
        data.y = data.y.to(opt.device)

        model.zero_grad()

        out = model(data)

        # Loss is computed with respect to the target sequence
        loss = loss_function(out, data.y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # Metrics are computed with respect to generated folding
        pred = out.max(1)[1]
        accuracy = compute_metrics_family(data.y, pred)
        accuracies.append(accuracy)

    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(accuracies)

    print("training loss is {}".format(avg_loss))
    print("accuracy: {}".format(avg_accuracy))

    return avg_loss.item(), avg_accuracy

def run(model, n_epochs, train_loader,val_loader,results_dir, model_dir,loss_function,optimizer):

    print("The model contains {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(n_epochs):
        start = time.time()
        print("Epoch {}: ".format(epoch + 1))

        loss, accuracy = train_epoch(model, train_loader,loss_function,optimizer)
        val_loss, val_accuracy = evaluate_family_classifier(model, val_loader,
                                                                          loss_function, mode='val',
                                                                    device=opt.device, verbose=opt.verbose)
        end = time.time()
        print("Epoch took {0:.2f} seconds".format(end - start))

        if not val_accuracies or val_accuracy > max(val_accuracies):
            torch.save(model.state_dict(), model_dir + 'model.pt')
            print("Saved updated model")
        #
        train_losses.append(loss)
        val_losses.append(val_loss)
        train_accuracies.append(accuracy)
        val_accuracies.append(val_accuracy)

        # plot_loss(train_losses, val_losses,file_name=results_dir + 'loss.jpg')
        # plot_loss(train_accuracies, val_accuracies, file_name=results_dir + 'acc.jpg',
        #           y_label='accuracy')

        pickle.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
        }, open(results_dir + 'scores.pkl', 'wb'))

        if len(val_accuracies) > opt.early_stopping and max(val_accuracies[-opt.early_stopping:])\
                < max(val_accuracies):
            print("Training terminated because of early stopping")
            print("Best val_loss: {}".format(min(val_losses)))
            print("Best val_accuracy: {}".format(max(val_accuracies)))

            with open(results_dir + 'scores.txt', 'w') as f:
                f.write("Best val_accuracy: {}".format(max(
                    val_accuracies)))
            break

def main(hidden_dim,n_conv_layers,dropout,learning_rate):
    model_name='hid'+str(hidden_dim)+'c_conv'+str(n_conv_layers)+'dr'+str(dropout)+'lr'+str(learning_rate)
    results_dir = '../results_family_classification/{}/'.format(model_name)
    model_dir = '../models_family_classification/{}/'.format(model_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(results_dir + 'hyperparams.txt', 'a') as f:
        f.write(str(opt))

    with open(results_dir + 'hyperparams.pkl', 'wb') as f:
        pickle.dump(opt, f)
    
    n_train_samples = None if not opt.n_samples else int(opt.n_samples * 0.8)
    n_val_samples = None if not opt.n_samples else int(opt.n_samples * 0.1)

    train_set = RNAFamilyGraphDataset(opt.train_dataset, opt.foldings_dataset,
                                    seq_max_len=opt.seq_max_len,
                                        seq_min_len=opt.seq_min_len,
                                        n_samples=n_train_samples)
    val_set = RNAFamilyGraphDataset(opt.val_dataset, opt.foldings_dataset, seq_max_len=opt.seq_max_len,
                                    seq_min_len=opt.seq_min_len,
                                    n_samples=n_val_samples)

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False)
 
    model=create_model(hidden_dim,n_conv_layers,dropout,learning_rate)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    run(model, opt.n_epochs, train_loader,val_loader,results_dir, model_dir,loss_function,optimizer)

if __name__ == "__main__":
    #main(hidden_dim=80,n_conv_layers=5,dropout=0.1,learning_rate=0.0004)
    hidden_dim=[80]
    n_conv_layers=[2,4,6]
    dropout=[0.2]
    learning_rate=[0.0005]
    params=product(hidden_dim,n_conv_layers,dropout,learning_rate)
    for param in params:
        main(hidden_dim=param[0],n_conv_layers=param[1],dropout=param[2],learning_rate=param[3])
