import argparse
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math

from torchvision import transforms

from image_helper import ImageHelper
from text_helper import TextHelper

from utils.utils import dict_html

logger = logging.getLogger("logger")
# logger.setLevel("ERROR")
import yaml
import time
import visdom
import numpy as np

vis = visdom.Visdom()
import random
from utils.text_load import *

criterion = torch.nn.CrossEntropyLoss()

# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# random.seed(1)

def train(helper, epoch, train_data_sets, local_model, target_model, is_poison, last_weight_accumulator=None):

    ### Accumulate weights for all participants.
    weight_accumulator = dict()
    for name, data in target_model.state_dict().items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    ### This is for calculating distances
    target_params_variables = dict()
    for name, param in target_model.named_parameters():
        target_params_variables[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)
    current_number_of_adversaries = 0
    for model_id, _ in train_data_sets:
        if model_id == -1 or model_id in helper.params['adversary_list']:
            current_number_of_adversaries += 1
    logger.info(f'There are {current_number_of_adversaries} adversaries in the training.')

    for model_id in range(helper.params['no_models']):
        model = local_model
        ## Synchronize LR and models
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()

        start_time = time.time()
        if helper.params['type'] == 'text':
            current_data_model, train_data = train_data_sets[model_id]
            ntokens = len(helper.corpus.dictionary)
            hidden = model.init_hidden(helper.params['batch_size'])
        else:
            _, (current_data_model, train_data) = train_data_sets[model_id]
        batch_size = helper.params['batch_size']
        ### For a 'poison_epoch' we perform single shot poisoning

        if current_data_model == -1:
            ### The participant got compromised and is out of the training.
            #  It will contribute to poisoning,
            continue
        if is_poison and current_data_model in helper.params['adversary_list'] and \
                (epoch in helper.params['poison_epochs'] or helper.params['random_compromise']):
            logger.info('poison_now')
            poisoned_data = helper.poisoned_data_for_train

            _, acc_p = test_poison(helper=helper, epoch=epoch,
                                   data_source=helper.test_data_poison,
                                   model=model, is_poison=True, visualize=False)
            _, acc_initial = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                             model=model, is_poison=False, visualize=False)
            logger.info(acc_p)
            poison_lr = helper.params['poison_lr']
            if not helper.params['baseline']:
                if acc_p > 20:
                    poison_lr /=50
                if acc_p > 60:
                    poison_lr /=100




            retrain_no_times = helper.params['retrain_poison']
            step_lr = helper.params['poison_step_lr']

            poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
                                               momentum=helper.params['momentum'],
                                               weight_decay=helper.params['decay'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                             milestones=[0.2 * retrain_no_times,
                                                                         0.8 * retrain_no_times],
                                                             gamma=0.1)

            is_stepped = False
            is_stepped_15 = False
            saved_batch = None
            acc = acc_initial
            try:
                # fisher = helper.estimate_fisher(target_model, criterion, train_data,
                #                                 12800, batch_size)
                # helper.consolidate(local_model, fisher)

                for internal_epoch in range(1, retrain_no_times + 1):
                    if step_lr:
                        scheduler.step()
                        logger.info(f'Current lr: {scheduler.get_lr()}')
                    if helper.params['type'] == 'text':
                        data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])
                    else:
                        data_iterator = poisoned_data


                    # logger.info("fisher")
                    # logger.info(fisher)


                    logger.info(f"PARAMS: {helper.params['retrain_poison']} epoch: {internal_epoch},"
                                f" lr: {scheduler.get_lr()}")
                    # if internal_epoch>20:
                    #     data_iterator = train_data

                    for batch_id, batch in enumerate(data_iterator):

                        if helper.params['type'] == 'image':
                            for i in range(helper.params['poisoning_per_batch']):
                                for pos, image in enumerate(helper.params['poison_images']):
                                    poison_pos = len(helper.params['poison_images'])*i + pos
                                    #random.randint(0, len(batch))
                                    batch[0][poison_pos] = helper.train_dataset[image][0]
                                    batch[0][poison_pos].add_(torch.FloatTensor(batch[0][poison_pos].shape).normal_(0, helper.params['noise_level']))


                                    batch[1][poison_pos] = helper.params['poison_label_swap']

                        data, targets = helper.get_batch(poisoned_data, batch, False)

                        poison_optimizer.zero_grad()
                        if helper.params['type'] == 'text':
                            hidden = helper.repackage_hidden(hidden)
                            output, hidden = model(data, hidden)
                            class_loss = criterion(output[-1].view(-1, ntokens),
                                                   targets[-batch_size:])
                        else:
                            output = model(data)
                            class_loss = nn.functional.cross_entropy(output, targets)

                        all_model_distance = helper.model_dist_norm(target_model, target_params_variables)
                        norm = 2
                        distance_loss = helper.model_dist_norm_var(model, target_params_variables)

                        loss = helper.params['alpha_loss'] * class_loss + (1 - helper.params['alpha_loss']) * distance_loss

                        ## visualize
                        if helper.params['report_poison_loss'] and batch_id % 2 == 0:
                            loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                                        data_source=helper.test_data_poison,
                                                        model=model, is_poison=True,
                                                        visualize=False)

                            model.train_vis(vis=vis, epoch=internal_epoch,
                                            data_len=len(data_iterator),
                                            batch=batch_id,
                                            loss=class_loss.data,
                                            eid=helper.params['environment_name'],
                                            name='Classification Loss', win='poison')

                            model.train_vis(vis=vis, epoch=internal_epoch,
                                            data_len=len(data_iterator),
                                            batch=batch_id,
                                            loss=all_model_distance,
                                            eid=helper.params['environment_name'],
                                            name='All Model Distance', win='poison')

                            model.train_vis(vis=vis, epoch=internal_epoch,
                                            data_len = len(data_iterator),
                                            batch = batch_id,
                                            loss = acc_p / 100.0,
                                            eid = helper.params['environment_name'], name='Accuracy',
                                            win = 'poison')

                            model.train_vis(vis=vis, epoch=internal_epoch,
                                            data_len=len(data_iterator),
                                            batch=batch_id,
                                            loss=acc / 100.0,
                                            eid=helper.params['environment_name'], name='Main Accuracy',
                                            win='poison')


                            model.train_vis(vis=vis, epoch=internal_epoch,
                                            data_len=len(data_iterator),
                                            batch=batch_id, loss=distance_loss.data,
                                            eid=helper.params['environment_name'], name='Distance Loss',
                                            win='poison')


                        loss.backward()

                        if helper.params['diff_privacy']:
                            torch.nn.utils.clip_grad_norm(model.parameters(), helper.params['clip'])
                            poison_optimizer.step()

                            model_norm = helper.model_dist_norm(model, target_params_variables)
                            if model_norm > helper.params['s_norm']:
                                logger.info(
                                    f'The limit reached for distance: '
                                    f'{helper.model_dist_norm(model, target_params_variables)}')
                                norm_scale = helper.params['s_norm'] / ((model_norm))
                                for name, layer in model.named_parameters():
                                    #### don't scale tied weights:
                                    if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                        continue
                                    clipped_difference = norm_scale * (
                                    layer.data - target_model.state_dict()[name])
                                    layer.data.copy_(
                                        target_model.state_dict()[name] + clipped_difference)

                        elif helper.params['type'] == 'text':
                            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           helper.params['clip'])
                            poison_optimizer.step()
                        else:
                            poison_optimizer.step()
                    loss, acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=model, is_poison=False, visualize=False)
                    loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                            data_source=helper.test_data_poison,
                                            model=model, is_poison=True, visualize=False)
                    #
                    if loss_p<=0.0001:
                        if helper.params['type'] == 'image' and acc<acc_initial:
                            if step_lr:
                                scheduler.step()
                            continue

                        raise ValueError()
                    logger.error(
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')
            except ValueError:
                logger.info('Converged earlier')

            logger.info(f'Global model norm: {helper.model_global_norm(target_model)}.')
            logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

            ### Adversary wants to scale his weights. Baseline model doesn't do this
            if not helper.params['baseline']:
                ### We scale data according to formula: L = 100*X-99*G = G + (100*X- 100*G).
                clip_rate = (helper.params['scale_weights'] / current_number_of_adversaries)
                logger.info(f"Scaling by  {clip_rate}")
                for key, value in model.state_dict().items():
                    #### don't scale tied weights:
                    if helper.params.get('tied', False) and key == 'decoder.weight' or '__'in key:
                        continue
                    target_value = target_model.state_dict()[key]
                    new_value = target_value + (value - target_value) * clip_rate

                    model.state_dict()[key].copy_(new_value)
                distance = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'Scaled Norm after poisoning: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')

            if helper.params['diff_privacy']:
                model_norm = helper.model_dist_norm(model, target_params_variables)

                if model_norm > helper.params['s_norm']:
                    norm_scale = helper.params['s_norm'] / (model_norm)
                    for name, layer in model.named_parameters():
                        #### don't scale tied weights:
                        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                            continue
                        clipped_difference = norm_scale * (
                        layer.data - target_model.state_dict()[name])
                        layer.data.copy_(target_model.state_dict()[name] + clipped_difference)
                distance = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'Scaled Norm after poisoning and clipping: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')

            if helper.params['track_distance'] and model_id < 10:
                distance = helper.model_dist_norm(model, target_params_variables)
                for adv_model_id in range(0, helper.params['number_of_adversaries']):
                    logger.info(
                        f'MODEL {adv_model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                        f'Distance to the global model: {distance:.4f}. '
                        f'Dataset size: {train_data.size(0)}')
                    vis.line(Y=np.array([distance]), X=np.array([epoch]),
                             win=f"global_dist_{helper.params['current_time']}",
                             env=helper.params['environment_name'],
                             name=f'Model_{adv_model_id}',
                             update='append' if vis.win_exists(
                                 f"global_dist_{helper.params['current_time']}",
                                 env=helper.params['environment_name']) else None,
                             opts=dict(showlegend=True,
                                       title=f"Distance to Global {helper.params['current_time']}",
                                       width=700, height=400))

            for key, value in model.state_dict().items():
                #### don't scale tied weights:
                if helper.params.get('tied', False) and key == 'decoder.weight' or '__'in key:
                    continue
                target_value = target_model.state_dict()[key]
                new_value = target_value + (value - target_value) * current_number_of_adversaries
                model.state_dict()[key].copy_(new_value)
            distance = helper.model_dist_norm(model, target_params_variables)
            logger.info(f"Total norm for {current_number_of_adversaries} "
                        f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")

        else:

            ### we will load helper.params later
            if helper.params['fake_participants_load']:
                continue

            for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):
                total_loss = 0.
                if helper.params['type'] == 'text':
                    data_iterator = range(0, train_data.size(0) - 1, helper.params['bptt'])
                else:
                    data_iterator = train_data
                for batch_id, batch in enumerate(data_iterator):
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(train_data, batch,
                                                      evaluation=False)
                    if helper.params['type'] == 'text':
                        hidden = helper.repackage_hidden(hidden)
                        output, hidden = model(data, hidden)
                        loss = criterion(output.view(-1, ntokens), targets)
                    else:
                        output = model(data)
                        loss = nn.functional.cross_entropy(output, targets)

                    loss.backward()

                    if helper.params['diff_privacy']:
                        optimizer.step()
                        model_norm = helper.model_dist_norm(model, target_params_variables)

                        if model_norm > helper.params['s_norm']:
                            norm_scale = helper.params['s_norm'] / (model_norm)
                            for name, layer in model.named_parameters():
                                #### don't scale tied weights:
                                if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                    continue
                                clipped_difference = norm_scale * (
                                layer.data - target_model.state_dict()[name])
                                layer.data.copy_(
                                    target_model.state_dict()[name] + clipped_difference)
                    elif helper.params['type'] == 'text':
                        # `clip_grad_norm` helps prevent the exploding gradient
                        # problem in RNNs / LSTMs.
                        torch.nn.utils.clip_grad_norm_(model.parameters(), helper.params['clip'])
                        optimizer.step()
                    else:
                        optimizer.step()

                    total_loss += loss.data

                    if helper.params["report_train_loss"] and batch % helper.params[
                        'log_interval'] == 0 and batch > 0:
                        cur_loss = total_loss.item() / helper.params['log_interval']
                        elapsed = time.time() - start_time
                        logger.info('model {} | epoch {:3d} | internal_epoch {:3d} '
                                    '| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                    'loss {:5.2f} | ppl {:8.2f}'
                                            .format(model_id, epoch, internal_epoch,
                                            batch,train_data.size(0) // helper.params['bptt'],
                                            helper.params['lr'],
                                            elapsed * 1000 / helper.params['log_interval'],
                                            cur_loss,
                                            math.exp(cur_loss) if cur_loss < 30 else -1.))
                        total_loss = 0
                        start_time = time.time()
                    # logger.info(f'model {model_id} distance: {helper.model_dist_norm(model, target_params_variables)}')

            if helper.params['track_distance'] and model_id < 10:
                # we can calculate distance to this model now.
                distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                    f'Distance to the global model: {distance_to_global_model:.4f}. '
                    f'Dataset size: {train_data.size(0)}')
                vis.line(Y=np.array([distance_to_global_model]), X=np.array([epoch]),
                         win=f"global_dist_{helper.params['current_time']}",
                         env=helper.params['environment_name'],
                         name=f'Model_{model_id}',
                         update='append' if
                         vis.win_exists(f"global_dist_{helper.params['current_time']}",
                                                           env=helper.params[
                                                               'environment_name']) else None,
                         opts=dict(showlegend=True,
                                   title=f"Distance to Global {helper.params['current_time']}",
                                   width=700, height=400))

        for name, data in model.state_dict().items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - target_model.state_dict()[name])


    if helper.params["fake_participants_save"]:
        torch.save(weight_accumulator,
                   f"{helper.params['fake_participants_file']}_"
                   f"{helper.params['s_norm']}_{helper.params['no_models']}")
    elif helper.params["fake_participants_load"]:
        fake_models = helper.params['no_models'] - helper.params['number_of_adversaries']
        fake_weight_accumulator = torch.load(
            f"{helper.params['fake_participants_file']}_{helper.params['s_norm']}_{fake_models}")
        logger.info(f"Faking data for {fake_models}")
        for name in target_model.state_dict().keys():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(fake_weight_accumulator[name])

    return weight_accumulator


def test(helper, epoch, data_source,
         model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0
    if helper.params['type'] == 'text':
        hidden = model.init_hidden(helper.params['test_batch_size'])
        random_print_output_batch = \
        random.sample(range(0, (data_source.size(0) // helper.params['bptt']) - 1), 1)[0]
        data_iterator = range(0, data_source.size(0)-1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        dataset_size = len(data_source.dataset)
        data_iterator = data_source

    for batch_id, batch in enumerate(data_iterator):
        data, targets = helper.get_batch(data_source, batch, evaluation=True)
        if helper.params['type'] == 'text':
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, helper.n_tokens)
            total_loss += len(data) * criterion(output_flat, targets).data
            hidden = helper.repackage_hidden(hidden)
            pred = output_flat.data.max(1)[1]
            correct += pred.eq(targets.data).sum().to(dtype=torch.float)
            total_test_words += targets.data.shape[0]

            ### output random result :)
            if batch_id == random_print_output_batch * helper.params['bptt'] and \
                    helper.params['output_examples'] and epoch % 5 == 0:
                expected_sentence = helper.get_sentence(targets.data.view_as(data)[:, 0])
                expected_sentence = f'*EXPECTED*: {expected_sentence}'
                predicted_sentence = helper.get_sentence(pred.view_as(data)[:, 0])
                predicted_sentence = f'*PREDICTED*: {predicted_sentence}'
                score = 100. * pred.eq(targets.data).sum() / targets.data.shape[0]
                logger.info(expected_sentence)
                logger.info(predicted_sentence)

                vis.text(f"<h2>Epoch: {epoch}_{helper.params['current_time']}</h2>"
                         f"<p>{expected_sentence.replace('<','&lt;').replace('>', '&gt;')}"
                         f"</p><p>{predicted_sentence.replace('<','&lt;').replace('>', '&gt;')}</p>"
                         f"<p>Accuracy: {score} %",
                         win=f"text_examples_{helper.params['current_time']}",
                         env=helper.params['environment_name'])
        else:
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                              reduction='sum').item() # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    if helper.params['type'] == 'text':
        acc = 100.0 * (correct / total_test_words)
        total_l = total_loss.item() / (dataset_size-1)
        logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                       total_l, correct, total_test_words,
                                                       acc))
        acc = acc.item()
        total_l = total_l.item()
    else:
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                       total_l, correct, dataset_size,
                                                       acc))

    if visualize:
        model.visualize(vis, epoch, acc, total_l if helper.params['report_test_loss'] else None,
                        eid=helper.params['environment_name'], is_poisoned=is_poison)
    model.train()
    return (total_l, acc)


def test_poison(helper, epoch, data_source,
                model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    batch_size = helper.params['test_batch_size']
    if helper.params['type'] == 'text':
        ntokens = len(helper.corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        data_iterator = range(0, data_source.size(0) - 1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        data_iterator = data_source
        dataset_size = 1000

    for batch_id, batch in enumerate(data_iterator):
        if helper.params['type'] == 'image':

            for pos in range(len(batch[0])):
                batch[0][pos] = helper.train_dataset[random.choice(helper.params['poison_images_test'])][0]

                batch[1][pos] = helper.params['poison_label_swap']


        data, targets = helper.get_batch(data_source, batch, evaluation=True)
        if helper.params['type'] == 'text':
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += 1 * criterion(output_flat[-batch_size:], targets[-batch_size:]).data
            hidden = helper.repackage_hidden(hidden)

            ### Look only at predictions for the last words.
            # For tensor [640] we look at last 10, as we flattened the vector [64,10] to 640
            # example, where we want to check for last line (b,d,f)
            # a c e   -> a c e b d f
            # b d f
            pred = output_flat.data.max(1)[1][-batch_size:]


            correct_output = targets.data[-batch_size:]
            correct += pred.eq(correct_output).sum()
            total_test_words += batch_size
        else:
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                              reduction='sum').data.item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().to(dtype=torch.float)

    if helper.params['type'] == 'text':
        acc = 100.0 * (correct / total_test_words)
        total_l = total_loss.item() / dataset_size
    else:
        acc = 100.0 * (correct / dataset_size)
        total_l = total_loss / dataset_size
    logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.0f}%)'.format(model.name, is_poison, epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))
    if visualize:
        model.visualize(vis, epoch, acc, total_l if helper.params['report_test_loss'] else None,
                        eid=helper.params['environment_name'], is_poisoned=is_poison)
    model.train()
    return total_l, acc


if __name__ == '__main__':
    print('Start training')
    time_start_load_everything = time.time()

    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()

    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    if params_loaded['type'] == "image":
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'image'))
    else:
        helper = TextHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'text'))

    helper.load_data()
    helper.create_model()

    ### Create models
    if helper.params['is_poison']:
        helper.params['adversary_list'] = [0]+ \
                                random.sample(range(helper.params['number_of_total_participants']),
                                                      helper.params['number_of_adversaries']-1)
        logger.info(f"Poisoned following participants: {len(helper.params['adversary_list'])}")
    else:
        helper.params['adversary_list'] = list()

    best_loss = float('inf')
    vis.text(text=dict_html(helper.params, current_time=helper.params["current_time"]),
             env=helper.params['environment_name'], opts=dict(width=300, height=400))
    logger.info(f"We use following environment for graphs:  {helper.params['environment_name']}")
    participant_ids = range(len(helper.train_data))
    mean_acc = list()

    results = {'poison': list(), 'number_of_adversaries': helper.params['number_of_adversaries'],
               'poison_type': helper.params['poison_type'], 'current_time': current_time,
               'sentence': helper.params.get('poison_sentences', False),
               'random_compromise': helper.params['random_compromise'],
               'baseline': helper.params['baseline']}

    weight_accumulator = None

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)
    dist_list = list()
    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):
        start_time = time.time()

        if helper.params["random_compromise"]:
            # randomly sample adversaries.
            subset_data_chunks = random.sample(participant_ids, helper.params['no_models'])

            ### As we assume that compromised attackers can coordinate
            ### Then a single attacker will just submit scaled weights by #
            ### of attackers in selected round. Other attackers won't submit.
            ###
            already_poisoning = False
            for pos, loader_id in enumerate(subset_data_chunks):
                if loader_id in helper.params['adversary_list']:
                    if already_poisoning:
                        logger.info(f'Compromised: {loader_id}. Skipping.')
                        subset_data_chunks[pos] = -1
                    else:
                        logger.info(f'Compromised: {loader_id}')
                        already_poisoning = True
        ## Only sample non-poisoned participants until poisoned_epoch
        else:
            if epoch in helper.params['poison_epochs']:
                ### For poison epoch we put one adversary and other adversaries just stay quiet
                subset_data_chunks = [participant_ids[0]] + [-1] * (
                helper.params['number_of_adversaries'] - 1) + \
                                     random.sample(participant_ids[1:],
                                                   helper.params['no_models'] - helper.params[
                                                       'number_of_adversaries'])
            else:
                subset_data_chunks = random.sample(participant_ids[1:], helper.params['no_models'])
                logger.info(f'Selected models: {subset_data_chunks}')
        t=time.time()
        weight_accumulator = train(helper=helper, epoch=epoch,
                                   train_data_sets=[(pos, helper.train_data[pos]) for pos in
                                                    subset_data_chunks],
                                   local_model=helper.local_model, target_model=helper.target_model,
                                   is_poison=helper.params['is_poison'], last_weight_accumulator=weight_accumulator)
        logger.info(f'time spent on training: {time.time() - t}')
        # Average the models
        helper.average_shrink_models(target_model=helper.target_model,
                                     weight_accumulator=weight_accumulator, epoch=epoch)

        if helper.params['is_poison']:
            epoch_loss_p, epoch_acc_p = test_poison(helper=helper,
                                                    epoch=epoch,
                                                    data_source=helper.test_data_poison,
                                                    model=helper.target_model, is_poison=True,
                                                    visualize=True)
            mean_acc.append(epoch_acc_p)
            results['poison'].append({'epoch': epoch, 'acc': epoch_acc_p})

        epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=helper.target_model, is_poison=False, visualize=True)


        helper.save_model(epoch=epoch, val_loss=epoch_loss)

        logger.info(f'Done in {time.time()-start_time} sec.')

    if helper.params['is_poison']:
        logger.info(f'MEAN_ACCURACY: {np.mean(mean_acc)}')
    logger.info('Saving all the graphs.')
    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")

    if helper.params.get('results_json', False):
        with open(helper.params['results_json'], 'a') as f:
            if len(mean_acc):
                results['mean_poison'] = np.mean(mean_acc)
            f.write(json.dumps(results) + '\n')

    vis.save([helper.params['environment_name']])

