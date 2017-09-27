import random
import pickle
from time import time
import sys
import argparse

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.externals import joblib
from sklearn.metrics import f1_score, precision_recall_fscore_support


from load_data import ProcessData, load_data_file

def main():
    parser = argparse.ArgumentParser(description='Test Neural Network.')
    parser.add_argument('--checkpoint_model', help='Checkpoint Model.')
    parser.add_argument('--data_X', help='Test/Validation Data.')
    parser.add_argument('--data_Y', help='Test/Validation Labels.')
    parser.add_argument('--scoring', default='binary', help='Evaluation Measure.')
    parser.add_argument('--minibatch_size', type=int, default=256, help='Mini-batch Size.')

    args = parser.parse_args()

    if args.scoring not in ['binary','micro','macro', 'prf']:
        raise ValueError('Incorrect Evaluation  Measure Specified')

    # Load Checkpoint Model
    with open(args.checkpoint_model,'rb') as out_file:
        chk_pt = pickle.load(out_file)

    # Load & Process Data
    test_txt, test_Y = load_data_file(args.data_X, args.data_Y)
    X = chk_pt['token'].transform(test_txt)
    Y = chk_pt['ml_bin'].transform(test_Y)

    data_processor = chk_pt['token']

    print("Init Model")
    # Init Model
    from models.cnn2 import CNN
    clf = CNN(data_processor.embs, nc=Y.shape[1], de=data_processor.embs.shape[1],
              lr=chk_pt['args'].lr, p_drop=chk_pt['args'].dropout, decay=chk_pt['args'].lr_decay, clip=chk_pt['args'].grad_clip,
              fs=chk_pt['args'].cnn_conv_size, penalty=chk_pt['args'].penalty, train_emb=chk_pt['args'].learn_embeddings)
    clf.__setstate__(chk_pt['model_params'])
    print("CNN: hidden_state: %d word_vec_size: %d lr: %.5f decay: %.6f learn_emb: %s dropout: %.3f num_feat_maps: %d penalty: %.5f conv_widths: %s" % (chk_pt['args'].hidden_state,
                data_processor.embs.shape[1], chk_pt['args'].lr, chk_pt['args'].lr_decay, chk_pt['args'].learn_embeddings, chk_pt['args'].dropout, chk_pt['args'].num_feat_maps, chk_pt['args'].penalty,
                chk_pt['args'].cnn_conv_size))

    # Get Predictions
    idxs = list(range(len(X)))
    all_preds = []
    for start, end in zip(range(0, len(idxs), args.minibatch_size),
            range(args.minibatch_size, len(idxs)+args.minibatch_size, args.minibatch_size)):
        if len(idxs[start:end]) == 0:
            continue
        mini_batch_sample = data_processor.pad_data([X[i] for i in idxs[start:end]])
        preds = clf.predict(mini_batch_sample, np.float32(1.))
        all_preds += list(preds.flatten())

    # Evaluate
    prf1 = None
    if args.scoring == 'binary':
        f1 = f1_score(Y.argmax(axis=1), all_preds, average='binary')
    elif args.scoring == 'micro':
        f1 = f1_score(Y.argmax(axis=1), all_preds, average='micro')
    elif args.scoring == 'macro':
        f1 = f1_score(Y.argmax(axis=1), all_preds, average='macro')
    elif args.scoring == 'prf':
        prf1 = precision_recall_fscore_support(Y.argmax(axis=1), all_preds, average='binary')

    if prf1 is not None:
        print("Precision: %.4f Recall: %.4f F1: %.4f" % (prf1[0], prf1[1], prf1[2]))
        sys.stdout.flush()
    else:
        print("F1: %.4f" % (f1))
        sys.stdout.flush()

if __name__ == '__main__':
    main()
