import sys
import os
import random
import pickle
import argparse
from time import time

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import f1_score

from load_data import ProcessData, load_data_file
from label_bin import CustomLabelBinarizer

def main():
    parser = argparse.ArgumentParser(description='Train Neural Network.')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of updates to make.')
    parser.add_argument('--word_vectors', default=None, help='Word vecotors filepath.')
    parser.add_argument('--checkpoint_dir', default='./experiments/exp1/checkpoints/',
                        help='Checkpoint directory.')
    parser.add_argument('--checkpoint_name', default='checkpoint',
                        help='Checkpoint File Name.')
    parser.add_argument('--min_df', type=int, default=5, help='Min word count.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate.')
    parser.add_argument('--penalty', type=float, default=0.0, help='Regularization Parameter.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout Value.')
    parser.add_argument('--lr_decay', type=float, default=1e-6, help='Learning Rate Decay.')
    parser.add_argument('--minibatch_size', type=int, default=50, help='Mini-batch Size.')
    parser.add_argument('--val_minibatch_size', type=int, default=256, help='Val Mini-batch Size.')
    parser.add_argument('--train_data_X', help='Training Data.')
    parser.add_argument('--train_data_Y', help='Training Labels.')
    parser.add_argument('--val_data_X', help='Validation Data.')
    parser.add_argument('--val_data_Y', help='Validation Labels.')
    parser.add_argument('--seed', default=42, type=int, help='Random Seed.')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient Clip Value.')
    parser.add_argument('--cnn_conv_size', nargs='+', type=int, default=[3], help='CNN Covolution Sizes (widths)')
    parser.add_argument('--num_feat_maps', default=300, type=int, help='Number of CNN Feature Maps.')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load & Process Data
    train_txt, train_Y = load_data_file(args.train_data_X, args.train_data_Y)
    val_txt, val_Y = load_data_file(args.val_data_X, args.val_data_Y)

    data_processor = ProcessData(args.word_vectors, lower=True, min_df=args.min_df)
    X_train = data_processor.fit_transform(train_txt)
    X_val = data_processor.transform(val_txt)

    ml_vec = CustomLabelBinarizer()
    ml_vec.fit(train_Y)
    Y_train = ml_vec.transform(train_Y)
    Y_val = ml_vec.transform(val_Y)

    print("Init Model")
    sys.stdout.flush()
    # Init Model
    from models.cnn2 import CNN
    clf = CNN(data_processor.embs, nc=Y_train.shape[1], de=data_processor.embs.shape[1],
              lr=args.lr, p_drop=args.dropout, decay=args.lr_decay, clip=args.grad_clip,
              fs=args.cnn_conv_size, penalty=args.penalty, train_emb=args.learn_embeddings)
    print("CNN: hidden_state: %d word_vec_size: %d lr: %.5f decay: %.6f learn_emb: %s dropout: %.3f num_feat_maps: %d penalty: %.5f conv_widths: %s" % (args.hidden_state,
                data_processor.embs.shape[1], args.lr, args.lr_decay, args.learn_embeddings, args.dropout, args.num_feat_maps, args.penalty,
                args.cnn_conv_size))

    print("Training Model")
    sys.stdout.flush()
    train_idxs = list(range(len(X_train)))
    val_idxs = list(range(len(X_val)))
    # Train Model
    best_val_f1 = 0
    for epoch in range(1, args.num_epochs+1):
        mean_loss = []
        mean_f1 = []
        random.shuffle(train_idxs)
        epoch_t0 = time()
        for start, end in zip(range(0, len(train_idxs), args.minibatch_size),
             range(args.minibatch_size, len(train_idxs)+args.minibatch_size, args.minibatch_size)):
            if len(train_idxs[start:end]) == 0:
                continue
            mini_batch_sample = data_processor.pad_data([X_train[i] for i in train_idxs[start:end]])
            cost, preds = clf.train_batch(mini_batch_sample,
                    Y_train[train_idxs[start:end]].astype('int32'),
                    np.float32(0.))

            f1 = f1_score(Y_train[train_idxs[start:end]].argmax(axis=1), preds, average='binary')
            mean_f1.append(f1)
            mean_loss.append(cost)
            sys.stdout.write("Epoch: %d train_avg_loss: %.4f train_avg_f1: %.4f\r" %
                    (epoch, np.mean(mean_loss), np.mean(mean_f1)))
            sys.stdout.flush()

        # Validate Model
        final_preds = []
        val_loss = []
        for start, end in zip(range(0, len(val_idxs), args.val_minibatch_size),
             range(args.val_minibatch_size, len(train_idxs)+args.val_minibatch_size, args.val_minibatch_size)):
            if len(train_idxs[start:end]) == 0:
                continue
            mini_batch_sample = data_processor.pad_data([X_val[i] for i in val_idxs[start:end]])
            preds, cost = clf.predict_loss(mini_batch_sample, Y_val[val_idxs[start:end]], np.float32(1.))
            final_preds += list(preds.flatten())
            val_loss.append(cost)

        f1 = f1_score(Y_val.argmax(axis=1), final_preds, average='binary')
        sys.stdout.write("epoch: %d val_loss %.4f val_f1: %.4f train_avg_loss: %.4f train_avg_f1: %.4f time: %.1f\n" %
                (epoch, np.mean(val_loss), f1, np.mean(mean_loss), np.mean(mean_f1), time()-epoch_t0))
        sys.stdout.flush()

        # Checkpoint Model
        if f1 > best_val_f1:
            best_val_f1 = f1
            with open(os.path.abspath(args.checkpoint_dir)+'/'+args.checkpoint_name+'.pkl','wb') as out_file:
                pickle.dump({'model_params':clf.__getstate__(), 'token':data_processor,
                             'ml_bin':ml_vec, 'args':args, 'last_train_avg_loss': np.mean(mean_loss),
                             'last_train_avg_f1':np.mean(mean_f1), 'val_f1':f1}, out_file, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
