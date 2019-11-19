import torch

def train(model_name, model, loss_function, optimizer, epochs, transform, train_data, train_loader):

    # configures the model to 'train' mode, which ensures that all steps are recorded for back propagation
    model.train()

    for epoch in range(epochs):

        print('\t\t' + str(epoch))

        for train_digits, train_labels in train_loader:

            # flatten the training digit image
            train_digits = train_digits.view(train_digits.shape[0], -1)

            # reset the optimizing gradient to zero
            optimizer.zero_grad()

            # feed forward propagation
            pred = model(train_digits)

            # calculate loss function
            loss = loss_function(pred.squeeze(), train_labels)

            # back-propagate
            loss.backward()

            # change weights based on loss function
            optimizer.step()

    torch.save(model.state_dict(), './models/' + model_name + '.pt')