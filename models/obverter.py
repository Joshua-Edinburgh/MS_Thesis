import torch
import numpy as np


def msg_gen_decoder(agent, img, max_sentence_len, vocab_size, device):
    '''
        Similar to the decode process in obverter, but only generate the 
        message and probability for ONE img.
        Here greedy search is applied, maybe beam search can be used to enhance
        Output:
            best_sentence: the sentence, e.g. 'cbbac' given img input
            prob: the probability of this sequence output
    '''
    prob = 0
    next_msg = np.tile(np.expand_dims(np.arange(0, vocab_size), 1), 1) # (vocab_s, 1)
    dup_imgs = np.array([img]).repeat(vocab_size, axis=0)
    dup_imgs = torch.tensor(dup_imgs).float().to(device)       # The images and messages must be torch tensor
    
    for l in range(max_sentence_len):
        if l > 0:
            # ------ Duplicate the history messages to a matrix, each row is a new candidate seq.
            hist_msg = np.expand_dims(best_acts[:l],2).repeat(vocab_size, axis=1).transpose()            
            cand_msg = np.concatenate((hist_msg, next_msg), axis=1)
        else:
            cand_msg = next_msg       
        
        logits, probs = agent(dup_imgs, torch.Tensor(cand_msg).long().to(device))
        prob, sel_idx = torch.max(probs, 0)
        best_acts = cand_msg[sel_idx,:]
        if prob>0.95:
            break
    best_sentence = ''.join([chr(97+int(v)) for v in best_acts])
    
    return best_sentence, prob, best_acts

def prd_gen_decoder(agent, img, acts, max_sentence_len, vocab_size, device):
    '''
        Given agent, img and message, output the prediction probability (=1)
    '''
    img_tensor = torch.tensor(img).float().to(device).unsqueeze(0)
    _, prob = agent(img_tensor, torch.Tensor(acts).long().to(device).unsqueeze(0))
    
    return prob.detach()



def decode(model, all_inputs, max_sentence_len, vocab_size, device):
    '''
        Given the model and one batch images, greedy generate the most probable message.
        Input:
            model: the agent
            all_inputs: one batch of pictures (one in the pair)
        Output:
            actions: (batch_size, max_sent_len), the generated message
            all_probs: (batch_size,), the success prob. of message describing the image.
    '''
    relevant_procs = list(range(all_inputs.size(0)))    # Indicator in one batch
    
    # Initial the output sentence and prob with -1, for each figure in one batch
    actions = np.array([[-1 for _ in range(max_sentence_len)] for _ in relevant_procs]) # output sentence
    all_probs = np.array([-1. for _ in relevant_procs])
       
    
    for l in range(max_sentence_len):
        inputs = all_inputs[relevant_procs]
        batch_size = inputs.size(0)
        # 50*5, represent the symbol for each figure in one batch
        next_symbol = np.tile(np.expand_dims(np.arange(0, vocab_size), 1), batch_size).transpose()
        # Here run_communications stores the existing best sequence, and also the fresh one waiting to choose.
        if l > 0:
            run_communications = np.concatenate((np.expand_dims(actions[relevant_procs, :l].transpose(),
                                                                2).repeat(vocab_size, axis=2),
                                                 np.expand_dims(next_symbol, 0)), axis=0)
        else:
            run_communications = np.expand_dims(next_symbol, 0)
        
        # Expand inputs to 5*50, i.e. each vocab have one img, then feed 250 imgs and texts to agents
        # to get 250*1 probabilities, then reshape it to 50*5, each row represent the probability of 
        # choosing specific vocab to communicate. We then select the best one in vocabulary, and use
        # sel_comm_idx to record the chosen result.
        
        expanded_inputs = inputs.repeat(vocab_size, 1, 1, 1)
        logits, probs = model(expanded_inputs, torch.Tensor(run_communications.transpose().reshape(-1, 1 + l)).long().to(device))
        probs = probs.view((vocab_size, batch_size)).transpose(0, 1)

        probs, sel_comm_idx = torch.max(probs, dim=1)

        comm = run_communications[:, np.arange(len(relevant_procs)), sel_comm_idx.data.cpu().numpy()].transpose()
        
        # If any img can achieve prob>0.95, it is finished and we remove it from relevant_procs, 
        # and store the actions.
        finished_p = []
        for i, (action, p, prob) in enumerate(zip(comm, relevant_procs, probs)):
            if prob > 0.95:
                finished_p.append(p)
                if prob.item() < 0:
                    continue
            # Store the converged actions.
            for j, symb in enumerate(action):
                actions[p][j] = symb

            all_probs[p] = prob

        for p in finished_p:
            relevant_procs.remove(p)

        if len(relevant_procs) == 0:
            break

    actions[actions == -1] = vocab_size  # padding token
    actions = torch.Tensor(np.array(actions)).long().to(device)
    return actions, all_probs