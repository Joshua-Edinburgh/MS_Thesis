import argparse
import os
import torch
from torch.nn import NLLLoss
import matplotlib.pyplot as plt

import models.obverter as obverter
from data_prepare.data import load_images_dict, get_batches
from models.model import ConvModel
from utils.resultrecorder import *

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Emerging Language')

parser.add_argument('--lr', type=float, default=6e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=50, help='number of images in a batch')
parser.add_argument('--num_rounds', type=int, default=20000, help='number of total training rounds')
parser.add_argument('--num_games_per_round', type=int, default=20, help='number of games per round')
parser.add_argument('--vocab_size', type=int, default=5, help='vocabulary size')
parser.add_argument('--max_sentence_len', type=int, default=20, help='maximum sentence length')
parser.add_argument('--data_n_samples', type=int, default = 100, help=' number of samples per color, shape combination')
parser.add_argument('--exp_name', default='test', help='the name of the folder to store results')


args = parser.parse_args()

images_dict = load_images_dict(args.data_n_samples)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

agent1 = ConvModel(vocab_size=args.vocab_size).to(device)
agent2 = ConvModel(vocab_size=args.vocab_size).to(device)


loss_fn = NLLLoss()

# Remember the way it used for initialize parameters according to whether it require grad.
optimizer1 = torch.optim.Adam([p for p in agent1.parameters() if p.requires_grad], args.lr)
optimizer2 = torch.optim.Adam([p for p in agent2.parameters() if p.requires_grad], args.lr)


def get_message(s):
    return ''.join([chr(97+int(v.cpu().data)) for v in s if v < args.vocab_size])


def train_round(speaker, listener, batches, optimizer, max_sentence_len, vocab_size, idx_round):
    '''
        Input:
            speaker, listener: the agent initialized using ConvModel
            batches: the data input (image1, image2, label, description)
            optimizer: 
            max_sentence_len:
            vocab_size:           
    '''
    speaker.train(False)
    listener.train(True)

    round_total = 0
    round_correct = 0
    round_loss = 0
    round_sentence_length = 0

    for batch in batches:
        # ========== Input data extraction ================
        input1, input2, labels, descriptions = batch
        input1, input2 = torch.tensor(input1).float().to(device), torch.tensor(input2).float().to(device)
        labels = torch.tensor(labels).to(device)
        
        # ========== Play the game and backpropagation ================
        speaker_actions, speaker_probs = obverter.decode(speaker, input1, max_sentence_len, vocab_size, device)

        lg, probs = listener(input2, speaker_actions)
        predictions = torch.round(probs).long()     # Convert the probabilities to 0/1 predictions
        correct_vector = (predictions == labels).float()
        n_correct = correct_vector.sum()            # out of batch_size

        listener_loss = loss_fn(lg, labels.long())

        optimizer.zero_grad()
        listener_loss.backward()
        optimizer.step()

        msg = {}
        for n in range(len(labels)):
            speaker_object, listen_object = descriptions[n]
            msg[speaker_object] = get_message(speaker_actions[n])
        '''
        for t in zip(speaker_actions, speaker_probs, descriptions, labels, probs):
            speaker_action, speaker_prob, description, label, listener_prob = t
            speaker_object, listener_object = description
            msg[speaker_object] = get_message(speaker_action)
            #message = get_message(speaker_action)
            #print("message: '%s', speaker object: %s, speaker score: %.2f, listener object: %s, label: %d, listener score: %.2f" %
            #      (message, speaker_object, speaker_prob, listener_object, label.item(), listener_prob.item()))

        #print("batch accuracy", n_correct.item() / len(input1))
        #print("batch loss", listener_loss.item())
        '''
        # ========== Stats for a round (e.g. 20 games, each game have 50 batch_size data)
        round_correct += n_correct
        round_total += len(input1)
        round_loss += listener_loss * len(input1)       # !!!! Why * len(input1)???
        round_sentence_length += (speaker_actions < vocab_size).sum(dim=1).float().mean() * len(input1)

    round_accuracy = (round_correct / round_total).item()
    round_loss = (round_loss / round_total).item()
    round_sentence_length = (round_sentence_length / round_total).item()
    
    return round_accuracy, round_loss, round_sentence_length, msg


agent2_accuracy_history = []
agent2_message_length_history = []
agent2_loss_history = []
agents_msgdist_history = []

os.makedirs('checkpoints', exist_ok=True)


def print_round_stats(acc, sl, loss):
    print("*******")
    print("Round average accuracy: %.2f" % (acc * 100))
    print("Round average sentence length: %.1f" % sl)
    print("Round average loss: %.1f" % loss)
    print("*******")


for round in range(args.num_rounds):
    print("********** round %d **********" % round)
    batches = get_batches(images_dict, args.data_n_samples, args.num_games_per_round, args.batch_size)

    r_accuracy, r_loss, r_msglen, ag1_msg = train_round(agent1, agent2, batches, optimizer2, 
                                                                    args.max_sentence_len, args.vocab_size,
                                                                    idx_round = round)
    round += 1
    print("replacing roles")
    print("********** round %d **********" % round)

    r_accuracy, r_loss, r_msglen, ag2_msg = train_round(agent2, agent1, batches, optimizer1, 
                                                                    args.max_sentence_len, args.vocab_size,
                                                                    idx_round = round)

    msg_dist = train_recorder(round, r_accuracy, r_loss, r_msglen, ag1_msg, ag2_msg, rpt_gap = 10, folder=args.exp_name)
    
    agent2_accuracy_history.append(r_accuracy)
    agent2_message_length_history.append(r_msglen / 20)
    agent2_loss_history.append(r_loss)
    agents_msgdist_history.append(msg_dist / 20)

    if round % 50 == 1:
        path = 'runs/' + args.exp_name
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)

        t = list(range(len(agent2_accuracy_history)))
        plt.plot(t, agent2_accuracy_history, label="Accuracy")
        plt.plot(t, agent2_message_length_history, label="Message length (/20)")
        plt.plot(t, agent2_loss_history, label="Training loss")
        plt.plot(t, agents_msgdist_history, label = "Message distance (/20)")


        plt.xlabel('# Rounds')
        plt.legend()
        plt.savefig(path+"/graph.png")
        plt.clf()

    if round % 500 == 1:
        path = 'runs/' + args.exp_name+'/checkpoints'
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
        torch.save(agent1.state_dict(), os.path.join(path, 'agent1-%d.ckp' % round))
        torch.save(agent2.state_dict(), os.path.join(path, 'agent2-%d.ckp' % round))
        
        all_objects = all_objects(exclude=None)
        spk_msg_all = sample_msg_gen(agent1, all_objects, images_dict, 
                                      args.max_sentence_len, args.vocab_size, device, n_samples=3)  
        spk_msg_one = sample_msg_gen(agent1, all_objects, images_dict, 
                                      args.max_sentence_len, args.vocab_size, device, n_samples=1)
        consist_recorder(spk_msg_all, round,folder = args.exp_name)
        msg_recorder(spk_msg_one, round, name='msg', folder = args.exp_name)        
        
        
        
        
        