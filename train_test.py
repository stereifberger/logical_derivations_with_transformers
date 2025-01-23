# Import necessary libraries and modules
from imports import *
import architectures
import pickle

# Set device to CUDA if available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_batch(batch):
    # Unpack source and target batches
    src_batch, tgt_batch = zip(*batch)
    # Convert source batch to tensor and stack
    src_batch = torch.stack([torch.tensor(s, dtype=torch.long) for s in src_batch])
    # Convert target batch to tensor and stack
    tgt_batch = torch.stack([torch.tensor(t, dtype=torch.long) for t in tgt_batch])
    return src_batch, tgt_batch

def train_test_model(
    model, train_data_loader, test_data_loader, num_epochs=10, learning_rate=1e-4, save_dir=""
):
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=alphabet.symb["PAD"])
    device = model.device

    # Initialize lists to store loss and accuracy
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # Iterate over epochs
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss, correct_train, total_train = 0, 0, 0

        # Training loop over batches
        for batch in train_data_loader:
            # Move data to the device
            src = batch[0].to(device)
            tgt = batch[1].to(device)

            # Generate masks for the source and target sequences
            src_key_padding_mask = architectures.create_src_key_padding_mask(src, alphabet.symb["PAD"])
            tgt_input = tgt[:, :-1]  # Remove last token as input
            tgt_output = tgt[:, 1:]  # Remove first token for output

            # Create target masks
            tgt_mask, tgt_key_padding_mask = architectures.create_tgt_masks(tgt_input, alphabet.symb["PAD"])

            # Zero the gradients of optimizer
            optimizer.zero_grad()
            # Forward pass
            output = model(src, tgt_input, None, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)

            # Flatten the output and target tensors
            output_dim = output.shape[-1]
            output_flat = output.reshape(-1, output_dim)
            tgt_output_flat = tgt_output.contiguous().view(-1)

            # Compute loss
            loss = criterion(output_flat, tgt_output_flat)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            # Update training loss
            total_train_loss += loss.item()

            # Compute accuracy
            preds = output.argmax(dim=-1)
            correct_train += (preds == tgt_output).sum().item()
            total_train += tgt_output.numel()

        # Calculate average train loss and accuracy
        train_loss = total_train_loss / len(train_data_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluation loop
        model.eval()  # Set model to evaluation mode
        total_test_loss, correct_test, total_test = 0, 0, 0
        with torch.no_grad():  # No need to track gradients
            for batch in test_data_loader:
                # Move data to the device
                src = batch[0].to(device)
                tgt = batch[1].to(device)

                # Generate masks for the source and target sequences
                src_key_padding_mask = architectures.create_src_key_padding_mask(src, alphabet.symb["PAD"])
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                # Create target masks
                tgt_mask, tgt_key_padding_mask = architectures.create_tgt_masks(tgt_input, alphabet.symb["PAD"])

                # Forward pass
                output = model(src, tgt_input, None, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)

                # Flatten the output and target tensors
                output_dim = output.shape[-1]
                output_flat = output.reshape(-1, output_dim)
                tgt_output_flat = tgt_output.contiguous().view(-1)

                # Compute loss
                loss = criterion(output_flat, tgt_output_flat)
                total_test_loss += loss.item()

                # Compute accuracy
                preds = output.argmax(dim=-1)
                correct_test += (preds == tgt_output).sum().item()
                total_test += tgt_output.numel()

        # Calculate average test loss and accuracy
        test_loss = total_test_loss / len(test_data_loader)
        test_accuracy = correct_test / total_test
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Print the metrics for the current epoch
        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )

        # Save the model parameters to a file
        model_path = f"{save_dir}model_epoch_{epoch + 1}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # Plot metrics over epochs
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model


# Function to test the model on a single example
def test_model_single_example(model, data_loader, num_epochs=10, learning_rate=1e-4):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        # Generate a derivation example
        premises, conclusion, derivation_steps = derivations.generate_derivation(derivations.ipl_rules)
        if derivation_steps:
            # Prepare input tokens from premises and conclusion
            input_tokens = []
            for premise in premises:
                input_tokens.extend(utils.flatten_formula(premise))
                input_tokens.append(alphabet.symb["DE"])  # Separator between premises
            input_tokens.append(alphabet.symb["DE"])  # Separator before conclusion
            input_tokens.extend(utils.flatten_formula(conclusion))
            input_tokens.append(alphabet.symb["EOS"])  # End of sequence token

            # Create a tensor for source sequence
            src = torch.tensor([input_tokens], dtype=torch.long).to(device)
            src_key_padding_mask = architectures.create_src_key_padding_mask(src, alphabet.symb["PAD"])

            # Initialize target tokens list with start-of-sequence token
            max_length = 50  # Maximum length of generated sequence
            tgt_tokens = [alphabet.symb["SOS"]]
            for i in range(max_length):
                # Create a tensor for the current target sequence
                tgt_seq = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
                tgt_mask, tgt_key_padding_mask = architectures.create_tgt_masks(tgt_seq, alphabet.symb["PAD"])
                # Get model's prediction for the next token
                output = model(src, tgt_seq, None, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
                next_token = output.argmax(dim=-1)[:, -1].item()  # Get the next token
                tgt_tokens.append(next_token)
                if next_token == alphabet.symb["EOS"]:
                    break  # Stop decoding if end-of-sequence token is generated

            predicted_tokens = tgt_tokens[1:]  # Remove SOS token

            # Convert tokens to formulas
            predicted_formulas = []
            current_formula_tokens = []
            for tok in predicted_tokens:
                if tok == alphabet.symb["DE"]:  # Separator token
                    if current_formula_tokens:
                        formula = utils.reconstruct_formula(current_formula_tokens)
                        predicted_formulas.append(formula)
                        current_formula_tokens = []
                elif tok == alphabet.symb["EOS"]:  # End of sequence token
                    if current_formula_tokens:
                        formula = utils.reconstruct_formula(current_formula_tokens)
                        predicted_formulas.append(formula)
                    break
                else:
                    current_formula_tokens.append(tok)  # Add token to current formula

            # Print the premises and conclusion
            print("Premises and Conclusion:")
            for premise in premises:
                print(f"Premise: {utils.formula_to_string(premise)}")
            print(f"Conclusion: {utils.formula_to_string(conclusion)}")

            # Print the predicted derivation
            print("\nPredicted Derivation Steps:")
            for f in predicted_formulas:
                if f:
                    print(utils.formula_to_string(f))
                else:
                    print("Invalid formula")