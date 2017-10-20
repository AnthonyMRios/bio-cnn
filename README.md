# Convolutional Neural Networks for Biomedical Text Classification

Implementation of the Yoon Kim model [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) (EMNLP 2014). However, rather than training with Adadelta or Adagrad, we use the [Adam optimizer](https://arxiv.org/abs/1412.6980).

The model works at the word level by convolving over the successive word vectors for each word in a document. Then a document vector is created using max-over-time pooling which is then passed to a final softmax output layer.


We have extended the base Yoon Kim model in two ways. First, we have added the ability to pass auxiliary features to the final output layer (deep and wide). This allows us to use information that the CNN may  not be able to extract on its own. To use wide features, lines 36, 37, 65, 66, 78, 79, and 80  in cnn.py must be uncommented. Also, the wide features you want to use must be passed to the train_batch and predict functions in train.py and test.py. The second extension involves training an ordinal loss rather than the standard multi-class cross-entropy. The ordinal loss is useful when there is a natural order to the classes. For this extension only line 73 needs to be uncommented in cnn.py. In the train.py file, a multi-hot Y rather than one-hot encoded Y must be passed to train_batch. Specifically, given an ordinal scale (ABSENT, MILD, MODERATE, SEVERE), the multi-hot Y should be formated as [0,0,0], [1,0,0], [1,1,0], and [1,1,1], respectively.

Example data is provided in the data folder.

## Required Packages
- Python 2.7
- numpy 1.11.1+
- scipy 0.18.0+
- Theano
- gensim
- sklearn

## Usage

### Training

To simply train the model, run the following command:

```
python train.py --word_vectors '/path/to/word_vecs2/word_vectors' --train_data_X '/path/to/train.txt' --train_data_Y '/path/to/train_labels.txt' --val_data_X '/path/to/dev.txt' --val_data_Y '/path/to/dev_labels.txt' --checkpoint_dir '/where/to/save/model/checkpoints'
```

*note*: Word vectors must be in the gensim format.

### Training Options

```
usage: train.py [-h] [--num_epochs NUM_EPOCHS] [--word_vectors WORD_VECTORS]
                [--checkpoint_dir CHECKPOINT_DIR]
                [--checkpoint_name CHECKPOINT_NAME] [--min_df MIN_DF]
                [--lr LR] [--penalty PENALTY] [--dropout DROPOUT]
                [--lr_decay LR_DECAY] [--minibatch_size MINIBATCH_SIZE]
                [--val_minibatch_size VAL_MINIBATCH_SIZE]
                [--train_data_X TRAIN_DATA_X] [--train_data_Y TRAIN_DATA_Y]
                [--val_data_X VAL_DATA_X] [--val_data_Y VAL_DATA_Y]
                [--seed SEED] [--grad_clip GRAD_CLIP]
                [--cnn_conv_size CNN_CONV_SIZE [CNN_CONV_SIZE ...]]
                [--num_feat_maps NUM_FEAT_MAPS]

Train Neural Network.

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of updates to make.
  --word_vectors WORD_VECTORS
                        Word vectors filepath.
  --checkpoint_dir CHECKPOINT_DIR
                        Checkpoint directory.
  --checkpoint_name CHECKPOINT_NAME
                        Checkpoint File Name.
  --min_df MIN_DF       Min word count.
  --lr LR               Learning Rate.
  --penalty PENALTY     Regularization Parameter.
  --dropout DROPOUT     Dropout Value.
  --lr_decay LR_DECAY   Learning Rate Decay.
  --minibatch_size MINIBATCH_SIZE
                        Mini-batch Size.
  --val_minibatch_size VAL_MINIBATCH_SIZE
                        Val Mini-batch Size.
  --train_data_X TRAIN_DATA_X
                        Training Data.
  --train_data_Y TRAIN_DATA_Y
                        Training Labels.
  --val_data_X VAL_DATA_X
                        Validation Data.
  --val_data_Y VAL_DATA_Y
                        Validation Labels.
  --seed SEED           Random Seed.
  --grad_clip GRAD_CLIP
                        Gradient Clip Value.
  --cnn_conv_size CNN_CONV_SIZE [CNN_CONV_SIZE ...]
                        CNN Covolution Sizes (widths)
  --num_feat_maps NUM_FEAT_MAPS
                        Number of CNN Feature Maps.
```

### Testing

*Note*: The current test code is mainly for evaluation purposes

```
python test.py --data_X '/path/to/test.txt' --data_Y '/path/to/test_labels.txt' --checkpoint_model '/path/to/saved/model/checkpoint_cv3_3.pkl'
```

### Testing Options

```
usage: test.py [-h] [--checkpoint_model CHECKPOINT_MODEL] [--data_X DATA_X]
               [--data_Y DATA_Y] [--scoring SCORING]
               [--minibatch_size MINIBATCH_SIZE]

Test Neural Network.

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_model CHECKPOINT_MODEL
                        Checkpoint Model.
  --data_X DATA_X       Test/Validation Data.
  --data_Y DATA_Y       Test/Validation Labels.
  --scoring SCORING     Evaluation Measure.
  --minibatch_size MINIBATCH_SIZE
                        Mini-batch Size
```

## Acknowledgements

This repo is based on the following paper:

> Yoon Kim. "Convolutional neural networks for sentence classification." In EMNLP. 2014.

The wide features and CNN applications to biomedical text classification was used in the following paper:

> Anthony Rios and Ramakanth Kavuluru. "[Convolutional neural networks for biomedical text classification: application in indexing biomedical articles.](http://protocols.netlab.uky.edu/~rvkavu2/research/bcb-15.pdf)" Proceedings of the 6th ACM Conference on Bioinformatics, Computational Biology and Health Informatics. ACM, 2015.

```
@inproceedings{rios2015convolutional,
  title={Convolutional neural networks for biomedical text classification: application in indexing biomedical articles},
  author={Rios, Anthony and Kavuluru, Ramakanth},
  booktitle={Proceedings of the 6th ACM Conference on Bioinformatics, Computational Biology and Health Informatics},
  pages={258--267},
  year={2015},
  organization={ACM}
}
```

The ordinal extension is part of our method which ranked 3rd among 24 teams in the 2016 CEGS N-GRID NLP Shared Task. The details of the entire method can be found in the paper below:

> Anthony Rios and Ramakanth Kavuluru. "[Ordinal convolutional neural networks for predicting RDoC positive valence psychiatric symptom severity scores.](http://protocols.netlab.uky.edu/~rvkavu2/research/rdoc-rios-jbi-17.pdf)" Journal of Biomedical Informatics (2017).

```
@article{rios2017ordinal,
  title={Ordinal convolutional neural networks for predicting RDoC positive valence psychiatric symptom severity scores},
  author={Rios, Anthony and Kavuluru, Ramakanth},
  journal={Journal of Biomedical Informatics},
  year={2017},
  publisher={Elsevier}
}
```
