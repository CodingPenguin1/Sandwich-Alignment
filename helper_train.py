import time
import torch
from helper_evaluation import compute_accuracy


def train_model(model, num_epochs, train_loader,
                valid_loader, test_loader, optimizer,
                device, logging_interval=50,
                scheduler=None,
                scheduler_on='valid_acc', early_stopping=False,
                silent=False):

    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []

    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # Forward and back prop
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Logging
            minibatch_loss_list.append(loss.item())
            if not batch_idx % logging_interval and not silent:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} | Batch {batch_idx+1:04d}/{len(train_loader):04d} | Loss: {loss:.4f}' + 100 * ' ', end='\r')

        model.eval()
        with torch.no_grad():  # save memory during inference
            train_acc = compute_accuracy(model, train_loader, device=device)
            valid_acc = compute_accuracy(model, valid_loader, device=device)
            if not silent:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} | Train: {train_acc :.2f}% | Validation: {valid_acc :.2f}%' + 100 * ' ', end='\r')
            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())

        if scheduler is not None:
            if scheduler_on == 'valid_acc':
                scheduler.step(valid_acc_list[-1])
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(minibatch_loss_list[-1])
            else:
                raise ValueError(f'Invalid `scheduler_on` choice.')

        if early_stopping:
            avg_last_train_acc = sum(train_acc_list[-5:]) / 5
            avg_last_valid_acc = sum(valid_acc_list[-5:]) / 5
            if avg_last_train_acc > 1.1 * avg_last_valid_acc:
                print(f'Early stopping at epoch {epoch+1:03d} due to overfitting.' + 100 * ' ')
                break

    elapsed = (time.time() - start_time)/60
    print(f'\nTotal Training Time: {elapsed:.2f} min')

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')

    return minibatch_loss_list, train_acc_list, valid_acc_list
