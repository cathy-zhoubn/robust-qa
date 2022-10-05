"""
Design Problems TODO
    1. Amount of data for the three options are not equal (max: 50,000, min: 127)
    2. If we want to do all three, then have we would have to take in batch_num as handcrafted parameter
        hand-craft batch size, total of 3*127 eg. 3*10*12
    3. rn: each batch has a different encoding

Solutions
    1. set all to 3*127, batch size 16, 8 batces
    2. want to run more data? cut third method

"""

# This starts from a copu of train.py, but adapted to meta-learn
import argparse
import json
import os
from collections import OrderedDict
import torch
import numpy as np
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args


from tqdm import tqdm

def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def prepare_train_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st : offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples

def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples

class MetaTrainer():
    def __init__(self, args, log):
        self.lr = args.lr
        self.inner_lr = args.inner_lr #default: 0.1
        self.n_inner_iter = args.n_inner_iter # default: 2
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.sample_size = args.sample_size
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train_maml(self, model, metatrain_dataloader, metaeval_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')

            # with torch.enable_grad(), tqdm(total=len(metatrain_dataloader.dataset)) as progress_bar:
            batch_num = len(metatrain_dataloader)
            task_num = len(metatrain_dataloader[0])
            with torch.enable_grad(), tqdm(total=batch_num) as progress_bar: 
                for b in range(batch_num):
                    batch_train = metatrain_dataloader[b]
                    batch_val = metaeval_dataloader[b]

                    qry_losses = []
                    optim.zero_grad()
                    model.train()
                    
                    for i in range(task_num):
                        inner_opt = torch.optim.SGD(model.parameters(), lr=self.inner_lr)
                        bt = next(iter(batch_train[i]))
                        input_ids = bt['input_ids'].to(device)
                        attention_mask = bt['attention_mask'].to(device)
                        start_positions = bt['start_positions'].to(device)
                        end_positions = bt['end_positions'].to(device)
                        
                        
                        # Optimize the likelihood of the support set by taking
                        # gradient steps w.r.t. the model's parameters.
                        # This adapts the model's meta-parameters to the task.
                        # higher is able to automatically keep copies of
                        # your network's parameters as they are being updated.

                        for _ in range(self.n_inner_iter):
                            outputs = model(input_ids, attention_mask=attention_mask,
                                     start_positions=start_positions,
                                     end_positions=end_positions)
                            loss = outputs.loss
                            loss.backward()
                            inner_opt.step()

                        bv = next(iter(batch_val[0]))
                        input_ids = bv['input_ids'].to(device)
                        attention_mask = bv['attention_mask'].to(device)
                        start_positions = bv['start_positions'].to(device)
                        end_positions = bv['end_positions'].to(device)

                        # The final set of adapted parameters will induce some
                        # final loss and accuracy on the query dataset.
                        # These will be used to update the model's meta-parameters.
                        qry_outputs = model(input_ids, attention_mask=attention_mask,
                                     start_positions=start_positions,
                                     end_positions=end_positions)
                        qry_loss = qry_outputs.loss
                        qry_losses.append(qry_loss.detach())
                        # Update the model's meta-parameters to optimize the query
                        # losses across all of the tasks sampled in this batch.
                        # This unrolls through the gradient steps.
                        qry_loss.backward()

                    optim.step()
                    qry_losses = sum(qry_losses) / task_num

                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=qry_losses.item())
                    tbx.add_scalar('train/NLL', qry_losses.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model)
                    global_idx += 1
        return best_scores
    
    def train_maml_singular(self, model, metatrain_dataloader, metaeval_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')

            # with torch.enable_grad(), tqdm(total=len(metatrain_dataloader.dataset)) as progress_bar:
            batch_num = len(metatrain_dataloader)
            task_num = len(metatrain_dataloader[0])
            with torch.enable_grad(), tqdm(total=batch_num) as progress_bar: 
                for b in range(batch_num):
                    batch_train = metatrain_dataloader[b]
                    batch_val = metaeval_dataloader[b]

                    qry_losses = []
                    optim.zero_grad()
                    model.train()
                    
                    for i in range(task_num):
                        inner_opt = torch.optim.SGD(model.parameters(), lr=self.inner_lr)
                        bt = next(iter(batch_train[i]))
                        
                        
                        # Optimize the likelihood of the support set by taking
                        # gradient steps w.r.t. the model's parameters.
                        # This adapts the model's meta-parameters to the task.
                        # higher is able to automatically keep copies of
                        # your network's parameters as they are being updated.

                        for _ in range(self.n_inner_iter):

                            input_ids = bt['input_ids'][_*self.sample_size:(_+1)*self.sample_size].to(device)
                            attention_mask = bt['attention_mask'][_*self.sample_size:(_+1)*self.sample_size].to(device)
                            start_positions = bt['start_positions'][_*self.sample_size:(_+1)*self.sample_size].to(device)
                            end_positions = bt['end_positions'][_*self.sample_size:(_+1)*self.sample_size].to(device)
                            outputs = model(input_ids, attention_mask=attention_mask,
                                     start_positions=start_positions,
                                     end_positions=end_positions)
                            loss = outputs.loss
                            loss.backward()
                            inner_opt.step()

                        bv = next(iter(batch_val[0]))
                        input_ids = bv['input_ids'].to(device)
                        attention_mask = bv['attention_mask'].to(device)
                        start_positions = bv['start_positions'].to(device)
                        end_positions = bv['end_positions'].to(device)

                        # The final set of adapted parameters will induce some
                        # final loss and accuracy on the query dataset.
                        # These will be used to update the model's meta-parameters.
                        qry_outputs = model(input_ids, attention_mask=attention_mask,
                                     start_positions=start_positions,
                                     end_positions=end_positions)
                        qry_loss = qry_outputs.loss
                        qry_losses.append(qry_loss.detach())
                        # Update the model's meta-parameters to optimize the query
                        # losses across all of the tasks sampled in this batch.
                        # This unrolls through the gradient steps.
                        qry_loss.backward()

                    optim.step()
                    qry_losses = sum(qry_losses) / task_num

                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=qry_losses.item())
                    tbx.add_scalar('train/NLL', qry_losses.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model)
                    global_idx += 1
        return best_scores
    
    def train_reptile(self, model, metatrain_dataloader, metaeval_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')

            # with torch.enable_grad(), tqdm(total=len(metatrain_dataloader.dataset)) as progress_bar:
            batch_num = len(metatrain_dataloader)
            task_num = len(metatrain_dataloader[0])
            with torch.enable_grad(), tqdm(total=batch_num) as progress_bar: 
                for b in range(batch_num):
                    batch_train = metatrain_dataloader[b]
                    batch_val = metaeval_dataloader[b]

                    for i in range(task_num):
                        # sample task
                        # take k gradient step to get a new model
                        # OPTIONAL: eval
                        qry_losses = []
                        optim.zero_grad()
                        model.train()

                        new_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased").to(device)
                        new_model.load_state_dict(model.state_dict())  # copy? looks okay
                        new_model.train()

                        inner_opt = torch.optim.SGD(new_model.parameters(), lr=self.inner_lr)
                        bt = next(iter(batch_train[i]))
                        input_ids = bt['input_ids'].to(device)
                        attention_mask = bt['attention_mask'].to(device)
                        start_positions = bt['start_positions'].to(device)
                        end_positions = bt['end_positions'].to(device)

                        # K steps of gradient descent
                        for _ in range(self.n_inner_iter):
                            outputs = new_model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                            loss = outputs.loss
                            inner_opt.zero_grad()
                            loss.backward()
                            inner_opt.step()

                        bv = next(iter(batch_val[0]))
                        input_ids = bv['input_ids'].to(device)
                        attention_mask = bv['attention_mask'].to(device)
                        start_positions = bv['start_positions'].to(device)
                        end_positions = bv['end_positions'].to(device)

                        # The final set of adapted parameters will induce some
                        # final loss and accuracy on the query dataset.
                        # These will be used to update the model's meta-parameters.
                        qry_outputs = new_model(input_ids, attention_mask=attention_mask,
                                     start_positions=start_positions,
                                     end_positions=end_positions)
                        qry_loss = qry_outputs.loss
                        qry_losses = qry_loss.detach()
                        
                        # inject updates into each .grad
                        for p, new_p in zip(model.parameters(), new_model.parameters()):
                            if p.grad is None:
                                p.grad = Variable(torch.zeros(p.size())).to(device)
                            p.grad.data.add_(p.data - new_p.data)
                        
                        # update meta-parameters: DONE
                        optim.step()

                        progress_bar.update(len(input_ids))
                        progress_bar.set_postfix(epoch=epoch_num, NLL=qry_losses.item())
                        tbx.add_scalar('train/NLL', qry_losses.item(), global_idx)
                        if (global_idx % self.eval_every) == 0:
                            self.log.info(f'Evaluating at step {global_idx}...')
                            preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                            results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                            self.log.info('Visualizing in TensorBoard...')
                            for k, v in curr_score.items():
                                tbx.add_scalar(f'val/{k}', v, global_idx)
                            self.log.info(f'Eval {results_str}')
                            if self.visualize_predictions:
                                util.visualize(tbx,
                                            pred_dict=preds,
                                            gold_dict=val_dict,
                                            step=global_idx,
                                            split='val',
                                            num_visuals=self.num_visuals)
                            if curr_score['F1'] >= best_scores['F1']:
                                best_scores = curr_score
                                self.save(model)
                        global_idx += 1
        return best_scores
    
    def train_reptile_singular(self, model, metatrain_dataloader, metaeval_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')

            # with torch.enable_grad(), tqdm(total=len(metatrain_dataloader.dataset)) as progress_bar:
            batch_num = len(metatrain_dataloader)
            task_num = len(metatrain_dataloader[0])
            with torch.enable_grad(), tqdm(total=batch_num) as progress_bar: 
                for b in range(batch_num):
                    batch_train = metatrain_dataloader[b]
                    batch_val = metaeval_dataloader[b]

                    for i in range(task_num):
                        # sample task
                        # take k gradient step to get a new model
                        # OPTIONAL: eval
                        qry_losses = []
                        optim.zero_grad()
                        model.train()

                        new_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased").to(device)
                        new_model.load_state_dict(model.state_dict())  # copy? looks okay
                        new_model.train()

                        inner_opt = torch.optim.SGD(new_model.parameters(), lr=self.inner_lr)
                        bt = next(iter(batch_train[i]))

                        # K steps of gradient descent
                        for _ in range(self.n_inner_iter):
                            input_ids = bt['input_ids'][_*self.sample_size:(_+1)*self.sample_size].to(device)
                            attention_mask = bt['attention_mask'][_*self.sample_size:(_+1)*self.sample_size].to(device)
                            start_positions = bt['start_positions'][_*self.sample_size:(_+1)*self.sample_size].to(device)
                            end_positions = bt['end_positions'][_*self.sample_size:(_+1)*self.sample_size].to(device)
                            
                            outputs = new_model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                            loss = outputs.loss
                            inner_opt.zero_grad()
                            loss.backward()
                            inner_opt.step()

                        bv = next(iter(batch_val[0]))
                        input_ids = bv['input_ids'].to(device)
                        attention_mask = bv['attention_mask'].to(device)
                        start_positions = bv['start_positions'].to(device)
                        end_positions = bv['end_positions'].to(device)

                        # The final set of adapted parameters will induce some
                        # final loss and accuracy on the query dataset.
                        # These will be used to update the model's meta-parameters.
                        qry_outputs = new_model(input_ids, attention_mask=attention_mask,
                                     start_positions=start_positions,
                                     end_positions=end_positions)
                        qry_loss = qry_outputs.loss
                        qry_losses = qry_loss.detach()
                        
                        # inject updates into each .grad
                        for p, new_p in zip(model.parameters(), new_model.parameters()):
                            if p.grad is None:
                                p.grad = Variable(torch.zeros(p.size())).to(device)
                            p.grad.data.add_(p.data - new_p.data)
                        
                        # update meta-parameters: DONE
                        optim.step()

                        progress_bar.update(len(input_ids))
                        progress_bar.set_postfix(epoch=epoch_num, NLL=qry_losses.item())
                        tbx.add_scalar('train/NLL', qry_losses.item(), global_idx)
                        if (global_idx % self.eval_every) == 0:
                            self.log.info(f'Evaluating at step {global_idx}...')
                            preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                            results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                            self.log.info('Visualizing in TensorBoard...')
                            for k, v in curr_score.items():
                                tbx.add_scalar(f'val/{k}', v, global_idx)
                            self.log.info(f'Eval {results_str}')
                            if self.visualize_predictions:
                                util.visualize(tbx,
                                            pred_dict=preds,
                                            gold_dict=val_dict,
                                            step=global_idx,
                                            split='val',
                                            num_visuals=self.num_visuals)
                            if curr_score['F1'] >= best_scores['F1']:
                                best_scores = curr_score
                                self.save(model)
                        global_idx += 1
        return best_scores

    def train_reptile_task(self, model, metatrain_dataloader, metaeval_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')

            # with torch.enable_grad(), tqdm(total=len(metatrain_dataloader.dataset)) as progress_bar:
            batch_num = len(metatrain_dataloader)
            print(batch_num)
            task_num = len(metatrain_dataloader[0])
            with torch.enable_grad(), tqdm(total=batch_num) as progress_bar: 
                for b in range(batch_num):
                    batch_train = metatrain_dataloader[b]
                    batch_val = metaeval_dataloader[b]

                    for i in range(task_num):
                        # sample task
                        # take k gradient step to get a new model
                        # OPTIONAL: eval
                        qry_losses = []
                        optim.zero_grad()
                        model.train()

                        new_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased").to(device)
                        new_model.load_state_dict(model.state_dict())  # copy? looks okay
                        new_model.train()

                        inner_opt = torch.optim.SGD(new_model.parameters(), lr=self.inner_lr)
                        bt = next(iter(batch_train[i]))

                        # K steps of gradient descent
                        for _ in range(self.n_inner_iter):
                            input_ids = bt['input_ids'][_*self.sample_size:(_+1)*self.sample_size].to(device)
                            attention_mask = bt['attention_mask'][_*self.sample_size:(_+1)*self.sample_size].to(device)
                            start_positions = bt['start_positions'][_*self.sample_size:(_+1)*self.sample_size].to(device)
                            end_positions = bt['end_positions'][_*self.sample_size:(_+1)*self.sample_size].to(device)
                            
                            outputs = new_model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                            loss = outputs.loss
                            inner_opt.zero_grad()
                            loss.backward()
                            inner_opt.step()

                        bv = next(iter(batch_val[0]))
                        input_ids = bv['input_ids'].to(device)
                        attention_mask = bv['attention_mask'].to(device)
                        start_positions = bv['start_positions'].to(device)
                        end_positions = bv['end_positions'].to(device)

                        # The final set of adapted parameters will induce some
                        # final loss and accuracy on the query dataset.
                        # These will be used to update the model's meta-parameters.
                        qry_outputs = new_model(input_ids, attention_mask=attention_mask,
                                     start_positions=start_positions,
                                     end_positions=end_positions)
                        qry_loss = qry_outputs.loss
                        qry_losses = qry_loss.detach()
                        
                        # inject updates into each .grad
                        for p, new_p in zip(model.parameters(), new_model.parameters()):
                            if p.grad is None:
                                p.grad = Variable(torch.zeros(p.size())).to(device)
                            p.grad.data.add_(p.data - new_p.data)
                        
                        # update meta-parameters: DONE
                        optim.step()

                        progress_bar.update(len(input_ids))
                        progress_bar.set_postfix(epoch=epoch_num, NLL=qry_losses.item())
                        tbx.add_scalar('train/NLL', qry_losses.item(), global_idx)
                        if (global_idx % self.eval_every) == 0:
                            self.log.info(f'Evaluating at step {global_idx}...')
                            preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                            results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                            self.log.info('Visualizing in TensorBoard...')
                            for k, v in curr_score.items():
                                tbx.add_scalar(f'val/{k}', v, global_idx)
                            self.log.info(f'Eval {results_str}')
                            if self.visualize_predictions:
                                util.visualize(tbx,
                                            pred_dict=preds,
                                            gold_dict=val_dict,
                                            step=global_idx,
                                            split='val',
                                            num_visuals=self.num_visuals)
                            if curr_score['F1'] >= best_scores['F1']:
                                best_scores = curr_score
                                self.save(model)
                        global_idx += 1
        return best_scores

def get_dataset(args, datasets, data_dir, tokenizer, split_name):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
    return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict

def get_meta_dataset_opt1(args, datasets, data_dir, save_dir, tokenizer, split_name):
    datasets = datasets.split(',')
    dataset_name=''
    data_encodings = [] # list of encodings for 3 set, data_encoding[key][idx]
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        data_encoding_curr = read_and_process(args, tokenizer, dataset_dict_curr, save_dir, dataset_name, split_name)
        data_encodings.append(data_encoding_curr)
    
    # calculate batchsize
    x = min(len(data_encodings[i][list(data_encodings[i].keys())[i]]) for i in range(len(data_encodings)))
    batch_size = int( x / ((args.task_num+1) * int(args.sample_size *3 / 2)))

    train_encodings = [[[] for i in range(args.task_num)] for b in range(batch_size)]
    val_encodings = [[[]] for b in range(batch_size)]
    # task_encodings has length batchsize. In each batch, there are `tasknumber` [enc, enc, enc] <- dataset number, each has a encoding of size sample_size
    for ct, enc in enumerate(data_encodings):
        len_enc = len(enc[list(enc.keys())[0]])
        orders = list(range(len_enc))
        print(orders)
        index = 0 # index to get the item from orders
        for b in range(batch_size):
            for i in range(args.task_num): 
                cur_enc = {k:[] for k in enc.keys()}
                for j in range(int(args.sample_size)):
                    for k in enc.keys():
                        cur_enc[k].append(enc[k][orders[index]])
                    index += 1
                train_encodings[b][i].append(cur_enc)
            cur_enc = {k:[] for k in enc.keys()}
            for j in range(int(args.sample_size)):
                for k in enc.keys():
                    cur_enc[k].append(enc[k][orders[index]])
                index += 1
            val_encodings[b][0].append(cur_enc)
    
    merged_qa_train = [[] for b in range(batch_size)]
    for b in range(batch_size):
        for i in range(args.task_num):
            enc = {k:[] for k in train_encodings[0][0][0].keys()}
            for x in train_encodings[b][i]:
                for k in x:
                    enc[k] += x[k]
            merged_qa_train[b].append(util.QADataset(enc, train=(split_name=='train')))
    merged_qa_val = [[] for b in range(batch_size)]
    for b in range(batch_size):
        enc = {k:[] for k in train_encodings[0][0][0].keys()}
        for x in val_encodings[b][0]:
            for k in x:
                enc[k] += x[k]
        merged_qa_val[b].append(util.QADataset(enc, train=(split_name=='train')))
    return merged_qa_train, merged_qa_val

def get_meta_dataset_opt2(args, datasets, data_dir, save_dir, tokenizer, split_name):
    datasets = datasets.split(',')
    dataset_name=''
    data_encodings = [] # list of encodings for 3 set, data_encoding[key][idx]
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        data_encoding_curr = read_and_process(args, tokenizer, dataset_dict_curr, save_dir, dataset_name, split_name)
        data_encodings.append(data_encoding_curr)
    
    # calculate batchsize
    x = min(len(data_encodings[i][list(data_encodings[i].keys())[i]]) for i in range(len(data_encodings)))
    batch_size = int( x / ((args.task_num+1) * int(args.sample_size *3 / 2)))

    train_encodings = [[[] for i in range(args.task_num)] for b in range(batch_size)]
    val_encodings = [[[]] for b in range(batch_size)]
    # task_encodings has length batchsize. In each batch, there are `tasknumber` [enc, enc, enc] <- dataset number, each has a encoding of size sample_size
    for ct, enc in enumerate(data_encodings[:-1]):
        len_enc = len(enc[list(enc.keys())[0]])
        orders = list(range(len_enc))
        np.random.shuffle(orders)

        index = 0 # index to get the item from orders
        for b in range(batch_size):
            for i in range(args.task_num): 
                cur_enc = {k:[] for k in enc.keys()}
                for j in range(int(args.sample_size *3 / 2)):
                    for k in enc.keys():
                        cur_enc[k].append(enc[k][orders[index]])
                    index += 1
                train_encodings[b][i].append(cur_enc)

    enc = data_encodings[-1]
    len_enc = len(enc[list(enc.keys())[0]])
    orders = list(range(len_enc))
    np.random.shuffle(orders)
    index = 0
    for b in range(batch_size):
        cur_enc = {k:[] for k in enc.keys()}
        for j in range(int(args.sample_size *3 / 2)):
            for k in enc.keys():
                cur_enc[k].append(enc[k][orders[index]])
            index += 1
        val_encodings[b][0].append(cur_enc)
    
    merged_qa_train = [[] for b in range(batch_size)]
    for b in range(batch_size):
        for i in range(args.task_num):
            enc = {k:[] for k in train_encodings[0][0][0].keys()}
            for x in train_encodings[b][i]:
                for k in x:
                    enc[k] += x[k]
            merged_qa_train[b].append(util.QADataset(enc, train=(split_name=='train')))
    merged_qa_val = [[] for b in range(batch_size)]
    for b in range(batch_size):
        enc = {k:[] for k in train_encodings[0][0][0].keys()}
        for x in val_encodings[b][0]:
            for k in x:
                enc[k] += x[k]
        merged_qa_val[b].append(util.QADataset(enc, train=(split_name=='train')))
    return merged_qa_train, merged_qa_val

def get_meta_dataset_opt1_singular(args, datasets, data_dir, save_dir, tokenizer, split_name):
    datasets = datasets.split(',')
    dataset_name=''
    data_encodings = [] # list of encodings for 3 set, data_encoding[key][idx]
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        data_encoding_curr = read_and_process(args, tokenizer, dataset_dict_curr, save_dir, dataset_name, split_name)
        data_encodings.append(data_encoding_curr)
    
    # calculate batchsize
    x = min(len(data_encodings[i][list(data_encodings[i].keys())[i]]) for i in range(len(data_encodings)))
    batch_size = int( x / ((args.task_num+1) * args.n_inner_iter* int(args.sample_size *3 / 2)))

    train_encodings = [[[] for i in range(args.task_num)] for b in range(batch_size)]
    val_encodings = [[[]] for b in range(batch_size)]
    # task_encodings has length batchsize. In each batch, there are `tasknumber` [enc, enc, enc] <- dataset number, each has a encoding of size sample_size
    for ct, enc in enumerate(data_encodings):
        len_enc = len(enc[list(enc.keys())[0]])
        orders = list(range(len_enc))
        np.random.shuffle(orders)
        index = 0 # index to get the item from orders
        for b in range(batch_size):
            for i in range(args.task_num): 
                cur_enc = {k:[] for k in enc.keys()}
                for j in range(int(args.sample_size)):
                    for inner in range(args.n_inner_iter):
                        for k in enc.keys():
                            cur_enc[k].append(enc[k][orders[index]])
                        index += 1
                train_encodings[b][i].append(cur_enc)
            cur_enc = {k:[] for k in enc.keys()}
            for j in range(int(args.sample_size)):
                for k in enc.keys():
                    cur_enc[k].append(enc[k][orders[index]])
                index += 1
            val_encodings[b][0].append(cur_enc)
    
    merged_qa_train = [[] for b in range(batch_size)]
    for b in range(batch_size):
        for i in range(args.task_num):
            enc = {k:[] for k in train_encodings[0][0][0].keys()}
            for x in train_encodings[b][i]:
                for k in x:
                    enc[k] += x[k]
            merged_qa_train[b].append(util.QADataset(enc, train=(split_name=='train')))
    merged_qa_val = [[] for b in range(batch_size)]
    for b in range(batch_size):
        enc = {k:[] for k in train_encodings[0][0][0].keys()}
        for x in val_encodings[b][0]:
            for k in x:
                enc[k] += x[k]
        merged_qa_val[b].append(util.QADataset(enc, train=(split_name=='train')))
    return merged_qa_train, merged_qa_val

def get_meta_dataset_opt2(args, datasets, data_dir, save_dir, tokenizer, split_name):
    datasets = datasets.split(',')
    dataset_name=''
    data_encodings = [] # list of encodings for 3 set, data_encoding[key][idx]
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        data_encoding_curr = read_and_process(args, tokenizer, dataset_dict_curr, save_dir, dataset_name, split_name)
        data_encodings.append(data_encoding_curr)
    
    # calculate batchsize
    x = min(len(data_encodings[i][list(data_encodings[i].keys())[i]]) for i in range(len(data_encodings)))
    batch_size = int( x / ((args.task_num+1) * int(args.sample_size *3 / 2)))

    train_encodings = [[[] for i in range(args.task_num)] for b in range(batch_size)]
    val_encodings = [[[]] for b in range(batch_size)]
    # task_encodings has length batchsize. In each batch, there are `tasknumber` [enc, enc, enc] <- dataset number, each has a encoding of size sample_size
    for ct, enc in enumerate(data_encodings[:-1]):
        len_enc = len(enc[list(enc.keys())[0]])
        orders = list(range(len_enc))
        print(orders)

        index = 0 # index to get the item from orders
        for b in range(batch_size):
            for i in range(args.task_num): 
                cur_enc = {k:[] for k in enc.keys()}
                for j in range(int(args.sample_size *3 / 2)):
                    for k in enc.keys():
                        cur_enc[k].append(enc[k][orders[index]])
                    index += 1
                train_encodings[b][i].append(cur_enc)

    enc = data_encodings[-1]
    len_enc = len(enc[list(enc.keys())[0]])
    orders = list(range(len_enc))
    np.random.shuffle(orders)
    index = 0
    for b in range(batch_size):
        cur_enc = {k:[] for k in enc.keys()}
        for j in range(int(args.sample_size *3 / 2)):
            for k in enc.keys():
                cur_enc[k].append(enc[k][orders[index]])
            index += 1
        val_encodings[b][0].append(cur_enc)
    
    merged_qa_train = [[] for b in range(batch_size)]
    for b in range(batch_size):
        for i in range(args.task_num):
            enc = {k:[] for k in train_encodings[0][0][0].keys()}
            for x in train_encodings[b][i]:
                for k in x:
                    enc[k] += x[k]
            merged_qa_train[b].append(util.QADataset(enc, train=(split_name=='train')))
    merged_qa_val = [[] for b in range(batch_size)]
    for b in range(batch_size):
        enc = {k:[] for k in train_encodings[0][0][0].keys()}
        for x in val_encodings[b][0]:
            for k in x:
                enc[k] += x[k]
        merged_qa_val[b].append(util.QADataset(enc, train=(split_name=='train')))
    return merged_qa_train, merged_qa_val

def get_meta_dataset_opt2_singular(args, datasets, data_dir, save_dir, tokenizer, split_name):
    datasets = datasets.split(',')
    dataset_name=''
    data_encodings = [] # list of encodings for 3 set, data_encoding[key][idx]
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        data_encoding_curr = read_and_process(args, tokenizer, dataset_dict_curr, save_dir, dataset_name, split_name)
        data_encodings.append(data_encoding_curr)
    
    # calculate batchsize
    x = min(len(data_encodings[i][list(data_encodings[i].keys())[i]]) for i in range(len(data_encodings)))
    batch_size = int( x / ((args.task_num+1) * args.n_inner_iter* int(args.sample_size *3 / 2)))


    train_encodings = [[[] for i in range(args.task_num)] for b in range(batch_size)]
    val_encodings = [[[]] for b in range(batch_size)]
    # task_encodings has length batchsize. In each batch, there are `tasknumber` [enc, enc, enc] <- dataset number, each has a encoding of size sample_size
    for ct, enc in enumerate(data_encodings[:-1]):
        len_enc = len(enc[list(enc.keys())[0]])
        orders = list(range(len_enc))
        np.random.shuffle(orders)

        index = 0 # index to get the item from orders
        for b in range(batch_size):
            for i in range(args.task_num): 
                cur_enc = {k:[] for k in enc.keys()}
                for j in range(int(args.sample_size *3 / 2)):
                    for inner in range(args.n_inner_iter):
                        for k in enc.keys():
                            cur_enc[k].append(enc[k][orders[index]])
                        index += 1
                train_encodings[b][i].append(cur_enc)

    enc = data_encodings[-1]
    len_enc = len(enc[list(enc.keys())[0]])
    orders = list(range(len_enc))
    np.random.shuffle(orders)
    index = 0
    for b in range(batch_size):
        cur_enc = {k:[] for k in enc.keys()}
        for j in range(int(args.sample_size *3 / 2)):
            for k in enc.keys():
                cur_enc[k].append(enc[k][orders[index]])
            index += 1
        val_encodings[b][0].append(cur_enc)
    
    merged_qa_train = [[] for b in range(batch_size)]
    for b in range(batch_size):
        for i in range(args.task_num):
            enc = {k:[] for k in train_encodings[0][0][0].keys()}
            for x in train_encodings[b][i]:
                for k in x:
                    enc[k] += x[k]
            merged_qa_train[b].append(util.QADataset(enc, train=(split_name=='train')))
    merged_qa_val = [[] for b in range(batch_size)]
    for b in range(batch_size):
        enc = {k:[] for k in train_encodings[0][0][0].keys()}
        for x in val_encodings[b][0]:
            for k in x:
                enc[k] += x[k]
        merged_qa_val[b].append(util.QADataset(enc, train=(split_name=='train')))
    return merged_qa_train, merged_qa_val

def get_meta_dataset_opt3(args, save_dir, tokenizer):
    dataset_name=''
    train_ds = args.train_datasets.split(',')
    eval_ds = args.ft_train_datasets.split(',')
    data_encodings = [] # list of encodings for 3 set, data_encoding[key][idx]
    for dataset in train_ds:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{args.train_dir}/{dataset}')
        data_encoding_curr = read_and_process(args, tokenizer, dataset_dict_curr, save_dir, dataset_name, 'train')
        data_encodings.append(data_encoding_curr)
    for dataset in eval_ds:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{args.train_ft_dir}/{dataset}')
        data_encoding_curr = read_and_process(args, tokenizer, dataset_dict_curr, save_dir, dataset_name, 'train')
        data_encodings.append(data_encoding_curr)
    
    data_ind = np.random.choice(len(data_encodings), size=3, replace=False)
    data_encodings = [data_encodings[i] for i in data_ind]
    # calculate batchsize
    x = min(len(data_encodings[i][list(data_encodings[i].keys())[i]]) for i in range(len(data_encodings)))
    batch_size = int( x / ((args.task_num+1) * args.sample_size))

    train_encodings = [[[] for i in range(args.task_num)] for b in range(batch_size)]
    val_encodings = [[[]] for b in range(batch_size)]
    # task_encodings has length batchsize. In each batch, there are `tasknumber` [enc, enc, enc] <- dataset number, each has a encoding of size sample_size
    for ct, enc in enumerate(data_encodings):
        len_enc = len(enc[list(enc.keys())[0]])
        orders = list(range(len_enc))
        np.random.shuffle(orders)
        index = 0 # index to get the item from orders
        for b in range(batch_size):
            for i in range(args.task_num): 
                cur_enc = {k:[] for k in enc.keys()}
                for j in range(args.sample_size):
                    for k in enc.keys():
                        cur_enc[k].append(enc[k][orders[index]])
                    index += 1
                train_encodings[b][i].append(cur_enc)
            cur_enc = {k:[] for k in enc.keys()}
            for j in range(args.sample_size):
                for k in enc.keys():
                    cur_enc[k].append(enc[k][orders[index]])
                index += 1
            val_encodings[b][0].append(cur_enc)
    
    merged_qa_train = [[] for b in range(batch_size)]
    for b in range(batch_size):
        for i in range(args.task_num):
            enc = {k:[] for k in train_encodings[0][0][0].keys()}
            for x in train_encodings[b][i]:
                for k in x:
                    enc[k] += x[k]
            merged_qa_train[b].append(util.QADataset(enc, train=True)) # TODO: (split_name=='train')
    merged_qa_val = [[] for b in range(batch_size)]
    for b in range(batch_size):
        enc = {k:[] for k in train_encodings[0][0][0].keys()}
        for x in val_encodings[b][0]:
            for k in x:
                enc[k] += x[k]
        merged_qa_val[b].append(util.QADataset(enc, train=True))  #TODO: (split_name=='train')
    return merged_qa_train, merged_qa_val

def get_meta_dataset_task(args, datasets, data_dir, save_dir, tokenizer, split_name):
    # tast num = 3
    datasets = datasets.split(',')
    dataset_name=''
    data_encodings = [] # list of encodings for 3 set, data_encoding[key][idx]
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        data_encoding_curr = read_and_process(args, tokenizer, dataset_dict_curr, save_dir, dataset_name, split_name)
        data_encodings.append(data_encoding_curr)
    
    # calculate batchsize
    x = min(len(data_encodings[i][list(data_encodings[i].keys())[i]]) for i in range(len(data_encodings)))
    batch_size = int( x / ((args.task_num+1) * args.n_inner_iter* int(args.sample_size *3 / 2)))

    train_encodings = [[] for b in range(batch_size)]
    val_encodings = [[] for b in range(batch_size)]
    # task_encodings has length batchsize. In each batch, there are `tasknumber` [enc, enc, enc] <- dataset number, each has a encoding of size sample_size
    for ct, enc in enumerate(data_encodings):
        len_enc = len(enc[list(enc.keys())[0]])
        orders = list(range(len_enc))
        np.random.shuffle(orders)
        index = 0 # index to get the item from orders
        for b in range(batch_size):
            cur_enc = {k:[] for k in enc.keys()}
            for j in range(int(3* args.sample_size)):
                for inner in range(args.n_inner_iter):
                    for k in enc.keys():
                        cur_enc[k].append(enc[k][orders[index]])
                    index += 1
            train_encodings[b].append(cur_enc)
            cur_enc = {k:[] for k in enc.keys()}
            for j in range(int(args.sample_size)):
                for k in enc.keys():
                    cur_enc[k].append(enc[k][orders[index]])
                index += 1
            val_encodings[b].append(cur_enc)
    
    merged_qa_train = [[] for b in range(batch_size)]
    for b in range(batch_size):
        enc = {k:[] for k in train_encodings[0][0].keys()}
        for x in train_encodings[b]:
            for k in x:
                enc[k] += x[k]
        merged_qa_train[b].append(util.QADataset(enc, train=(split_name=='train')))
    merged_qa_val = [[] for b in range(batch_size)]
    for b in range(batch_size):
        enc = {k:[] for k in train_encodings[0][0].keys()}
        for x in val_encodings[b]:
            for k in x:
                enc[k] += x[k]
        merged_qa_val[b].append(util.QADataset(enc, train=(split_name=='train')))
    return merged_qa_train, merged_qa_val


def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = MetaTrainer(args, log)
        
        if args.meta_datatype == "opt1":
            train_qas, val_qas = get_meta_dataset_opt1_singular(args, args.train_datasets, args.train_dir, "datasets/opt1", tokenizer, 'train') # TODO: add batch_size
        elif args.meta_datatype == "opt2":
            train_qas, val_qas = get_meta_dataset_opt2(args, args.train_datasets, args.train_dir, "datasets/opt2", tokenizer, 'train') # TODO: add batch_size
        elif args.meta_datatype == "opt3":
            train_qas, val_qas = get_meta_dataset_opt3(args, "datasets/opt3", tokenizer) # TODO: add batch_size
        elif args.meta_datatype == "opt1_singular":
            train_qas, val_qas = get_meta_dataset_opt1_singular(args, args.train_datasets, args.train_dir, "datasets/opt1_singular", tokenizer, 'train') # TODO: add batch_size
        elif args.meta_datatype == "opt2_singular":
            train_qas, val_qas = get_meta_dataset_opt2_singular(args, args.train_datasets, args.train_dir, "datasets/opt2_singular", tokenizer, 'train') # TODO: add batch_size
        elif args.meta_datatype == 'opt1_task':
            train_qas, val_qas = get_meta_dataset_task(args, args.train_datasets, args.train_dir, "datasets/opt1_task", tokenizer, 'train') # TODO: add batch_size
        else:
            raise Exception("meta dataset generation option not recognized")

        train_loaders, val_loaders = [], []
        for batch in range(len(train_qas)):
            train_loaders.append([])
            for task in train_qas[batch]:
                train_loaders[batch].append(DataLoader(task,
                                        batch_size=len(task),
                                        sampler=RandomSampler(task)))

        for batch in range(len(val_qas)):
            val_loaders.append([])
            for task in val_qas[batch]:
                val_loaders[batch].append(DataLoader(task,
                                        batch_size=len(task),
                                        sampler=RandomSampler(task)))
    
   
        # train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        #train_loader = DataLoader(train_dataset,
        #                        batch_size=args.batch_size,
        #                        sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        # best_scores = trainer.train(model, train_loader, val_loader, val_dict)
        if (args.meta_method == "maml"):
            best_scores = trainer.train_maml(model, train_loaders, val_loaders, val_loader, val_dict)
        elif (args.meta_method == "reptile"):
            best_scores = trainer.train_reptile(model, train_loaders, val_loaders, val_loader, val_dict)
        elif (args.meta_method == "maml_singular"):
            best_scores = trainer.train_maml_singular(model, train_loaders, val_loaders, val_loader, val_dict)
        elif (args.meta_method == "reptile_singular"):
            best_scores = trainer.train_reptile_singular(model, train_loaders, val_loaders, val_loader, val_dict)
        elif (args.meta_method == "reptile_task"):
            best_scores = trainer.train_reptile_task(model, train_loaders, val_loaders, val_loader, val_dict)
        else:
            raise Exception("Unsupported meta learning method")
        

    if args.do_finetune: # do_finetune, pretrain_model_path,save_dir, run_name. finetune_datasets, train_ft_dir, val_ft_dir
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = MetaTrainer(args, log)
        path = os.path.join(args.pretrain_model_path)
        model = DistilBertForQuestionAnswering.from_pretrained(path)
        model.to(args.device)
        train_dataset, _ = get_dataset(args, args.ft_train_datasets, args.train_ft_dir, tokenizer, 'train')
        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.ft_val_datasets, args.val_ft_dir, tokenizer, 'val')
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        best_scores = trainer.train(model, train_loader, val_loader, val_dict)

    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = MetaTrainer(args, log)
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name) 
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    main()

