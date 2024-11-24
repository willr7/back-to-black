import torch 
from dataset.dataset import causal_mask

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(encoder_output)

        prob = model.project(out[:, -1])

        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item())], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input

def beam_search(model, beam_size, encoder_input, encoder_mask, tokenizer_source, tokenizer_target, max_len, device):
    """
    Performs beam search on the encoder
    
    Parameters:
    encoder_input: Torch.Tensor
        encoded input source sentence
    
    encoder_mask: Torch.Tensor
        binary vector indicating which elements of the encoder's output should be considered during decoding process

    Return:
    """
    sos_idx = tokenizer_target.token_to_id(['SOS'])
    eos_idx = tokenizer_target.token_to_id(['EOS'])

    # precompute the encoder output
    encoder_output = model.encode(encoder_input, encoder_mask)
    # initialize decoder input with the sos token with the same type as the encoder input
    decoder_initial_input = torch.empty(1,1).fill_(sos_idx).type_as(encoder_input).to(device)

    # create a beam list
    beam_list = [(decoder_initial_input, 1)]

    while True:

        # checking for any beam that has reached max length
            # this means we have run the decoding for at least max_len iterations, so stop the beam search
        if any([beam.size(1) == max_len for beam, _ in beam_list]):
            break

        new_beams = []

        # explores k * k possible beams and then only move forward with top k beams
        for beam, score in beam_list:

            # checking if the eos token has been reached
            if beam[0][-1] == eos_idx:
                continue

            # build beam's mask 
            beam_mask = causal_mask(beam.size(1)).type_as(encoder_mask).to(device)

            # calculate output
            output = model.decode(encoder_output, encoder_mask, beam, beam_mask)
            # get next token probabilities (score)
            prob = model.project(output[:, -1])

            # get top k beams
            top_k_prob, top_k_idx = torch.topk(prob, beam_size, dim=1)

            for i in range(beam_size):

                # for each of the top k beams, get the token and its probability
                token = top_k_idx[0][i].unsqueeze(0).unsqueeze(0)    
                token_prob = top_k_prob[0][i].item()

                # create new beam by appending token to current beam
                new_beam = torch.cat([beam, token], dim=1)
                # sum the log probabilities cuz' probabilities in log space 
                # (adding in log space => multiplying in normal base) 
                new_beams.append((new_beam, score + token_prob))

        # sort the new beams by their score value
        beam_list = sorted(new_beams, key=lambda x: x[1], reverse=True)
        # keeps the top k beams
        beam_list = beam_list[:beam_size]

        # If all the beams have reached the eos token, stop
        if all([beam[0][-1].item() == eos_idx for beam, _ in beam_list]):
            break

    # return the best beam after beam search
    return beam_list[0][0].squeeze()

def run_validation(model, validation_ds, tokenizerPsrc, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.shape[0] == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'EXPECTED: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break
