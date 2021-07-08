
def train(model,trainloader, optimizer, epochs, device = 'cuda'):

    model.train()
    print(len(trainloader))
    pbar = tqdm(trainloader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.cross_entropy(y_pred, target)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
    
    mean_loss = sum(train_losses)/len(train_losses)
    scheduler.step(mean_loss)

def test(model, device, testloader):    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    
    test_acc.append(100. * correct / len(testloader.dataset))

def run_model(model, trainloader, optimizer, epochs = 40, device= 'cuda'):    
    for epoch in range(epochs):
        print("EPOCH:", epoch)
        train(model,trainloader, optimizer, epochs, device = 'cuda')
        test(model, device, testloader)

