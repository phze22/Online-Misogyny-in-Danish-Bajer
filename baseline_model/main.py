import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from preprocessing import preprocess
from train import train_
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, help='For distinguishing between different experiment: creating directory paths/ file paths')
    parser.add_argument('--subtask', type=str,required=True, help='The subtask for training: subtask_A, subtask_B, subtask_C1 or subtask_C2; see codebook for more information')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model_tag', default='bert-base-multilingual-uncased')
    parser.add_argument('--reload', type=bool, default=False, help='Setting to True when loading a trained model from a saved model')
    parser.add_argument('--train', default=True) # if False, evaluate and explain - to be implemented
    parser.add_argument('--cuda', default=False)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs',type=int,default=10)
    # early stopping with decreasing learning rate. 0: direct exit when validation F1 decreases
    # parser.add_argument("--early_stop", default=5, type=int)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    args = parser.parse_args()

    print('Experiment:' + str(args.experiment))
    print('Subtask:' + str(args.subtask))
    print('Data:' + str(args.data_path))
    print('Batch size:' + str(args.batch_size))
    print('Epochs:' + str(args.epochs))

    if not os.path.exists(Path('experiment_' + str(args.experiment))):
        Path('experiment_' + str(args.experiment)).mkdir(parents=True, exist_ok=True)
    mode = 0o666
    parent_dir = 'experiment_' + str(args.experiment)
    subdir = args.subtask
    subsubdir = ['results', 'models']
    # results: metrics, fp/fn etc.
    # models: saved weights, configs
    subdir_path = Path(parent_dir, subdir)
    subdir_path.mkdir(parents=True, exist_ok=True)
    for dir in subsubdir:
        dir_path = Path(parent_dir, subdir, dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        print("Directory '% s' created" % dir)
    raw_data = pd.read_csv(args.data_path)
    data = raw_data[['text', args.subtask]]
    output_dir = Path(parent_dir, subdir, 'models')

    # 1 hot-encoding
    labels = data[args.subtask].unique()
    print(labels)

    #hard coded for fn/ fp for subtask_C1
    if args.subtask == 'subtask_C1':
        label_dict = {1:1, 0:0}
    else:
        label_dict = {}
        index = 0
        for label in labels:
            label_dict[label] = index
            index += 1
    assert set(labels)==set(label_dict)
    print(label_dict)
    data['label'] = data[args.subtask].replace(label_dict)


    # preprocessing data
    data['text'] = preprocess(data)
    print('preprocessing done:')
    print(raw_data['text'].iloc[0])
    print(data['text'].iloc[0])
    label_list = data.label.values

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.local_rank == -1:
            n_gpu = torch.cuda.device_count()
    n_gpu = torch.cuda.device_count()
    print(device)
    print('number of available gpus:' + str(n_gpu))

    label_list = data.label.values

    # Train
    if args.train == True:
        val_loss, val_f1, val_ids, predictions, true_vals, fp_ids, fn_ids = train_(data, args, label_dict, device, n_gpu, output_dir)

        # Metrics
        def accuracy_per_class(preds, labels):
            label_dict_inverse = {v: k for k, v in label_dict.items()}
            for label in np.unique(labels):
                y_preds = preds[labels == label]
                y_true = labels[labels == label]
                print(f'Class: {label_dict_inverse[label]}')
                print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')

        y_pred = np.argmax(predictions, axis=1).flatten()
        y_true = true_vals.flatten()

        #confusion matrix
        output = confusion_matrix(y_true, y_pred).ravel()
        conf = confusion_matrix(y_true, y_pred)
        print(conf)

        #accuracy oer class
        accuracy_per_class(y_true, y_pred)

        # Recover fp, fn
        if args.subtask == 'subtask_C1':
            fp_ = raw_data[raw_data.index.isin(fp_ids.tolist())]
            fn_ = raw_data[raw_data.index.isin(fn_ids.tolist())]
            fp_.to_csv(os.path.join(parent_dir, subdir, 'results/false_positives.csv'), index=False)
            fn_.to_csv(os.path.join(parent_dir, subdir, 'results/false_negatives.csv'), index=False)


        # output file
        f = open(os.path.join(parent_dir, subdir, 'results/output.txt'),'w+')
        f.write('Experiment:' + str(args.experiment)+'\n')
        f.write('Subtask:' + str(args.subtask) + '\n')
        #f.write('Accuracy per class:' + '\n')
        #f.write(accuracy_per_class(predictions, true_vals))
        #f.write('\n')
        f.write('Confusion_matrix:' + '\n')
        f.write(np.array2string(conf, separator=' ,'))
        f.write('\n')
        f.write('\n')
        f.write('classification_report:' + '\n')
        f.write(classification_report(y_true, y_pred))
        f.close()

        #classification_report
        print(classification_report(y_true, y_pred))

    # def evaluate()

    # Evaluate and Explain:
    #if args.train==False & args.run_explanations==True:
    #
    #    def load_model(model_path):
    #        loaded_model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    #        return loaded_model
    #
    #    model = load_model(args.output_dir)
    #
    #    def explain(args, model, tokenizer, label_list, device):