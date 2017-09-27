# Convolutional Neural Networks for Biomedical Text Classification

Implementation of a convolutional neural network for text classification.

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

```
python train.py --word_vectors '/path/to/word_vecs2/word_vectors' --train_data_X '/path/to/train.txt' --train_data_Y '/path/to/train_labels.txt' --val_data_X '/path/to/dev.txt' --val_data_Y '/path/to/dev_labels.txt' --checkpoint_dir '/where/to/save/model/checkpoints'
```

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
                        Word vecotors filepath.
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

## Cite

> Rios, Anthony, and Ramakanth Kavuluru. "Convolutional neural networks for biomedical text classification: application in indexing biomedical articles." Proceedings of the 6th ACM Conference on Bioinformatics, Computational Biology and Health Informatics. ACM, 2015.

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
