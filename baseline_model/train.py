import os
import numpy as np
import torch
import json
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data import TensorDataset
from tqdm import tqdm, tqdm_notebook
from transformers import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def processor(data, subtask, experiment, model_tag):

    print(data[0:5])
    print('dataset length: '+ str(len(data)))

    print('sub-sample data...')
    label_list = data.label.values


    # Train_Test_Split
    X_train, X_test, y_train, y_test = train_test_split(data.index.values,
                                                      data.label.values,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      stratify=data.label.values)
    X_train, X_val, y_train, y_val = train_test_split(data.index.values,
                                                        data.label.values,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=data.label.values)

    label_list = data.label.values


    data['data_type'] = ['not_set'] * data.shape[0]
    data.loc[X_train, 'data_type'] = 'train'
    data.loc[X_val, 'data_type'] = 'val'
    data.groupby([subtask, 'label', 'data_type']).count().to_csv(os.path.join('experiment_' + experiment, subtask,'results/test_train_split.csv'))
    data[data.data_type == 'val'].to_csv(os.path.join('experiment_' + experiment, subtask,'results/val_dataset.csv'))

    print('train/test split done with random state')


    # Tokenization
    tokenizer = BertTokenizer.from_pretrained(model_tag, do_lower_case=True, verbose=False)
    tokenizer.add_tokens(['@USER', '<has_cap>', '<all_cap>'])

    encoded_data_train = tokenizer.batch_encode_plus(
        data[data.data_type == 'train'].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        truncation=True,
        max_length=512, #len(posts) in dataset <512
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        data[data.data_type == 'val'].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        truncation=True,
        max_length=512, #len(posts) in dataset <512
        return_tensors='pt'
    )
    
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(data[data.data_type == 'train'].label.values)
    #labels_train.to(device)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(data[data.data_type == 'val'].label.values)
    #labels_val.to(device)


    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    #print(input_ids_train.shape)
    #print(attention_masks_train.shape)
    #print(labels_train.shape)
    print('dataset_train:' + str(len(dataset_train)))
    print('dataset_val:' + str(len(dataset_val)))
    print('Tokenization done')
    
    return label_list, tokenizer, dataset_train, dataset_val, X_train, X_val, y_train, y_val, labels_train
    

def train_(data, args, label_dict, device, n_gpu, output_dir):

    
    label_list, tokenizer, dataset_train, dataset_val, X_train, X_val, y_train, y_val, labels_train = processor(data, args.subtask, args.experiment, args.model_tag)

    # Load Model, DataLoader and Optimizer & Scheduler
    num_labels=len(label_dict)

    if args.reload:
        model = BertForSequenceClassification.from_pretrained(output_dir,
                                                          num_labels=len(label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False,
                                                          return_dict=True)
    else:
        model = BertForSequenceClassification.from_pretrained(args.model_tag,
                                                              num_labels=len(label_dict),
                                                              output_attentions=False,
                                                              output_hidden_states=False,
                                                              return_dict=True)


    # additional tokens added
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    count = 0
    for param in model.named_parameters():
        count += 1
    print('no. model parameters:' +str(count))

    batch_size = args.batch_size
    epochs = args.epochs


    # Handling class imbalance for training set when updating weights (WeightRandomSampler)
    labels_unique, counts = np.unique(labels_train, return_counts=True)
    print("Unique labels: {}".format(labels_unique))
    class_weights = [sum(counts) / c for c in counts]
    weights = [class_weights[e] for e in labels_train]
    sampler = WeightedRandomSampler(weights, len(labels_train), replacement=False)

    dataloader_train = DataLoader(dataset_train,
                                  sampler=sampler,
                                  batch_size=batch_size)

    dataloader_validation = DataLoader(dataset_val,
                                       sampler=SequentialSampler(dataset_val), # run predicitons for full data
                                       batch_size=batch_size)

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,
                      weight_decay=0.01,
                      correct_bias=False,
                      )
                #torch.BertAdam: implements weight decay fix, does not compensate for bias
        
        #Batch normalization?
        #tune learning-rate, random seed and dropout probability with grid-search?

    

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int((len(dataloader_train) * epochs)/10),
                                                num_training_steps=len(dataloader_train) * epochs)
        
    print('No. epochs: \n')
    print(args.epochs)
    print('Model initialization/ Data Loading finished')
    
        
    
    # Metrics
    def metrics_(preds, labels, target_label):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        
        def simple_accuracy(preds, labels):
            return (preds == labels).mean()
        
        acc = simple_accuracy(preds_flat, labels_flat)
        f1 = f1_score(y_true=labels_flat, y_pred=preds_flat, average='weighted')
        p, r = precision_score(y_true=labels_flat, y_pred=preds_flat, average='weighted'), recall_score(y_true=labels_flat, y_pred=preds_flat, average='weighted')
        
        if (args.subtask == 'subtask_C1') or (args.subtask == 'subtask_A'):
            f1_target = f1_score(labels_flat, preds_flat, pos_label=target_label, average='binary')
        else:
            f1_target = {}
            i = 0
            for label in label_dict.items():
                (name, encoded) = label
                scores = f1_score(labels_flat, preds_flat, average=None)
                f1_target[name] = scores[i]
                i+=1
                print(f1_target)
        
        return {
        "acc": acc,
        "f1": f1,
        "precision": p,
        "recall": r,
        "f1_target": f1_target}



    # Train
    print('Start training...')


    def evaluate(dataloader_val):
        model.eval() #back-propagation - no dropout, batch normalization

        loss_val_total = 0
        predictions, true_y, input_ids = [], [], []

        for batch in dataloader_val:
            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}

            with torch.no_grad(): #sets requires_grad=False
                outputs = model(**inputs)

            #logits = output["logits"]
            #loss = loss_func(logits, target=true_labels)
            #loss_func = nn.CrossEntropyLoss(weight=class_weights)

            loss = outputs[0] #CrossEntropyLoss() used in BertForSequenceClassification
            preds = outputs[1]
            loss_val_total += loss.item()

            preds = preds.detach().cpu().numpy()
            true_labels = inputs['labels'].cpu().numpy()
            input_ids = inputs['input_ids'].cpu().numpy()
            
            predictions.append(preds)
            true_y.append(true_labels)

        loss_val_avg = loss_val_total / len(dataloader_val)

        predictions = np.concatenate(predictions, axis=0)
        input_ids = np.concatenate(input_ids, axis=0)
        true_y = np.concatenate(true_y, axis=0)

        return loss_val_avg, input_ids, predictions, true_y

    #set seed
    #seed_val = 17
    #random.seed(seed_val)
    #np.random.seed(seed_val)
    #torch.manual_seed(seed_val)
    #torch.cuda.manual_seed_all(seed_val)

    global_step = 0
    val_best_f1 = -1

    for epoch in tqdm(range(1, epochs + 1)):
        
        model.train()

        loss_train_total = 0
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

        
        for batch in progress_bar:
            model.zero_grad() #sets all gradients to 0

            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}

            #print(batch[0].shape, batch[1].shape, batch[2].shape)

            model_output = model(**inputs)
            
            loss = model_output[0]
            if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
            
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #norm is computed over all gradients together. Gradients are modified in-place

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, val_ids, predictions, true_vals = evaluate(dataloader_validation)
        metrics = metrics_(predictions, true_vals,target_label=1)
        print(metrics)
        acc = metrics["acc"]
        f1 = metrics["f1"]
        f1_target = metrics["f1_target"]
        prec = metrics["precision"]
        recall = metrics["recall"]
        
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {f1}')
        
        writer.add_scalars("train/val_loss",{'train_loss':loss_train_avg,'val_loss':val_loss}, epoch)

        if (args.subtask == 'subtask_C1') or (args.subtask == 'subtask_A'):
            writer.add_scalars("f1_score/f1_target", {'f1_score':f1,'f1_target':f1_target}, epoch)
        else:
            writer.add_scalar("f1_score_weighted", f1, epoch)
            print(f1_target.items())
            writer.add_scalars("f1_per_class", f1_target, epoch)
        writer.add_scalars("precision/recall", {'precision':prec,'recall':recall}, epoch)

        #writer.add_graph(model, input_to_model=dataset_train)    
        writer.close()

        # save model and halving learning rate every 100 epoch
        global_step += 1
        
        if global_step % 1 == 0:
            if f1 > val_best_f1:
                val_best_f1 = f1   
                save_model(model, tokenizer, output_dir)

                # Post-hoc Error Analysis: FP/FN from validation set
                y_pred = np.argmax(predictions, axis=1).flatten()
                y_true = true_vals.flatten()
                fp_ = y_pred * (1 - y_true)
                fn_ = y_true * (1 - y_pred)

                fp_ids = data[data.index.isin(X_val)].index[fp_ == True]
                fn_ids = data[data.index.isin(X_val)].index[fn_ == True]
                print("Model saved after epoch " + str(global_step))

        #else:
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] *= 0.5
        #        print("Learning rate halved")
        
    print("Training finished")

    return val_loss, f1, val_ids, predictions, true_vals, fp_ids, fn_ids


def save_model(model, tokenizer, output_dir):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

