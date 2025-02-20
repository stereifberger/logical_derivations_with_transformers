from imports import * # Import libraries

"""
Description: [Hands-On] Implementing a Simple Mathematical Calculator using Sequence-to-Sequence Learning, Changed by me, SR
Author: Hugman Sangkeun Jung
Date: 2024-03-20
URL: https://medium.com/@hugmanskj/hands-on-implementing-a-simple-mathematical-calculator-using-sequence-to-sequence-learning-85b742082c72
"""
def train_model(model, dataloader_train, dataloader_test, optimizer, criterion, epochs, device, max_y_length, y_train):
    CELtrain = []
    CELtest = []
    ACCtrain = []
    ACCtest = []

    for epoch in range(epochs):

        # Train loss
        model.train()
        correct_train = 0
        total_train = 0
        epoch_loss = []
        for i, x_train in enumerate(dataloader_train): # Iterate over batched input data
            x_train = x_train.to(device) # Load the batched input data to the GPU
            optimizer.zero_grad() # Reset optimized gradients

            y_pred = model(x_train[:,1:], max_y_length)               # Get the model's output for batch
            if y_train.dim() == 3:
                loss, y_train_collected = losses.mse_loss_ffn(y_pred, x_train, y_train, max_y_length, device) # Calculate loss
            else:
                loss, y_train_collected  = losses.mse_loss(y_pred, x_train, y_train, max_y_length, device) # Calculate loss
            epoch_loss.append(loss.item())
            loss.backward()  #  Perform backpropagation
            optimizer.step() #  Run optimizer
        train_loss = statistics.mean(epoch_loss)  # Stack losses and compute mean

        pred_labels = torch.argmax(y_pred, dim=-1) 
        target_labels = torch.argmax(y_train_collected, dim=-1)
        pred_labels = pred_labels.view(-1)  
        target_labels = target_labels.view(-1)
        correct = (pred_labels == target_labels).sum().item()
        total = target_labels.size(0)
        train_accuracy = correct / total
        ACCtrain.append(train_accuracy)

        CELtrain.append(train_loss)

        del y_train_collected
        torch.cuda.empty_cache()

        # Test loss, the same as above but without backpropagation
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = []
        with torch.no_grad():
            for i, x_test in enumerate(dataloader_test):
                x_test = x_test.to(device)
                y_pred = model(x_test[:, 1:], max_y_length)
                if y_train.dim() == 3:
                    loss, y_test_collected  = losses.mse_loss_ffn(y_pred, x_test, y_train, max_y_length, device) # Calculate loss
                else:
                    loss, y_test_collected  = losses.mse_loss(y_pred, x_test, y_train, max_y_length, device) # Calculate loss
                test_loss.append(loss.item())

        t_loss = statistics.mean(test_loss) 
        pred_labels = torch.argmax(y_pred, dim=-1)
        target_labels = torch.argmax(y_test_collected, dim=-1)
        pred_labels = pred_labels.view(-1)
        target_labels = target_labels.view(-1)
        correct = (pred_labels == target_labels).sum().item()
        total = target_labels.size(0)
        test_accuracy = correct / total
        CELtest.append(float(t_loss))
        ACCtest.append(test_accuracy)

        print(f"""Ep. {epoch+1:02}, CEL-Train: {train_loss:.4f}| CEL-Test: {t_loss:.4f} | ACC-Train: {train_accuracy:.4f} | ACC-Test:  {test_accuracy:.4f}""")
    return CELtrain, CELtest, ACCtrain, ACCtest

def sanity(model, dataloader, device, max_y_length): # The same as the test loss calculation above but model outputs and inputs are returned
    input_list = []
    output_list = []
    with torch.no_grad():
        for i, x_test in enumerate(dataloader):
            x_test = x_test.to(device)
            y_pred = model(x_test[:,1:], max_y_length)               # Get the model's output for batch
            index = 0
            while index < len(y_pred): # Reverse encoding of model outputs and inputs and returns them as human readable formulas
                pred = y_pred[index].reshape(-1, 14)
                test = x_test[index][1:]
                pred = torch.argmax(pred, dim=1) 
                test = [calculi.symb_reverse[num.item()] for num in test]
                test = ''.join(test)
                pred = [calculi.symb_reverse[num.item()] for num in pred]
                pred = ''.join(pred)
                index += 1
                input_list.append(test)
                output_list.append(pred)
    return input_list, output_list