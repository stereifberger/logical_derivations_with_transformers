# Import libraries
from imports import *
import architectures

def train_model(model, data_loader, num_epochs=10, learning_rate=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=alphabet.symb["PAD"])
    model.train()
    device = model.device

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            src = batch[0].to(device)
            tgt = batch[1].to(device)

            src_key_padding_mask = architectures.create_src_key_padding_mask(src, alphabet.symb["PAD"])
            tgt_input = tgt[:, :-1]  # Remove last token for input
            tgt_output = tgt[:, 1:]  # Remove first token for output

            tgt_mask, tgt_key_padding_mask = architectures.create_tgt_masks(tgt_input, alphabet.symb["PAD"])
            optimizer.zero_grad()
            # Perform the model forward pass
            output = model(src, tgt_input, None, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)

            output_dim = output.shape[-1]
            # Reshape output to compare with target output
            output_flat = output.reshape(-1, output_dim)
            tgt_output_flat = tgt_output.contiguous().view(-1)

            loss = criterion(output_flat, tgt_output_flat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Print 5 model outputs with logical symbols after each epoch
            # Print 5 model outputs and corresponding inputs with logical symbols after each epoch
            if batch_idx < 1:  # Change to 1 to print from the first batch
                predicted_tokens = output.argmax(dim=-1)
                print("\nSample model outputs and corresponding inputs:")

                for i in range(min(5, predicted_tokens.size(0))):  # Print up to 5 sequences
                    # Decode model outputs
                    output_tokens = predicted_tokens[i].cpu().numpy().tolist()
                    output_symbols = [alphabet.symb_reverse.get(token, "[UNK]") for token in output_tokens]

                    # Decode corresponding inputs
                    input_tokens = src[i].cpu().numpy().tolist()
                    input_symbols = [alphabet.symb_reverse.get(token, "[UNK]") for token in input_tokens]

                    print(f"Input {i}: {' '.join(input_symbols)}")
                    print(f"Output {i}: {' '.join(output_symbols)}\n")


        average_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')

# Testing the model
def test_model(model, data_loader, num_epochs=10, learning_rate=1e-4):
    model.eval()
    with torch.no_grad():
        premises, conclusion, derivation_steps = generate_derivation(ipl_rules)
        if derivation_steps:
            input_tokens = []
            for premise in premises:
                input_tokens.extend(utils.flatten_formula(premise))
                input_tokens.append(alphabet.symb["DE"])  # Separator between premises
            input_tokens.append(alphabet.symb["DE"])  # Separator before conclusion
            input_tokens.extend(utils.flatten_formula(conclusion))
            input_tokens.append(alphabet.symb["EOS"])  # End of sequence token

            src = torch.tensor([input_tokens], dtype=torch.long).to(device)
            src_key_padding_mask = architectures.create_src_key_padding_mask(src, alphabet.symb["PAD"])

            max_length = 50  # Maximum length of generated sequence
            tgt_tokens = [alphabet.symb["SOS"]]
            for i in range(max_length):
                tgt_seq = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
                tgt_mask, tgt_key_padding_mask = architectures.create_tgt_masks(tgt_seq, alphabet.symb["PAD"])
                output = model(src, tgt_seq, None, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
                next_token = output.argmax(dim=-1)[:, -1].item()
                tgt_tokens.append(next_token)
                if next_token == alphabet.symb["EOS"]:
                    break

            predicted_tokens = tgt_tokens[1:]  # Remove SOS token

            # Convert tokens to formulas
            predicted_formulas = []
            current_formula_tokens = []
            for tok in predicted_tokens:
                if tok == alphabet.symb["DE"]:
                    if current_formula_tokens:
                        formula = reconstruct_formula(current_formula_tokens)
                        predicted_formulas.append(formula)
                        current_formula_tokens = []
                elif tok == alphabet.symb["EOS"]:
                    if current_formula_tokens:
                        formula = reconstruct_formula(current_formula_tokens)
                        predicted_formulas.append(formula)
                    break
                else:
                    current_formula_tokens.append(tok)

            # Print the premises and conclusion
            print("Premises and Conclusion:")
            for premise in premises:
                print(f"Premise: {formula_to_string(premise)}")
            print(f"Conclusion: {formula_to_string(conclusion)}")

            # Print the predicted derivation
            print("\nPredicted Derivation Steps:")
            for f in predicted_formulas:
                if f:
                    print(formula_to_string(f))
                else:
                    print("Invalid formula")

def collate_batch(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack([torch.tensor(s, dtype=torch.long) for s in src_batch])
    tgt_batch = torch.stack([torch.tensor(t, dtype=torch.long) for t in tgt_batch])
    return src_batch, tgt_batch