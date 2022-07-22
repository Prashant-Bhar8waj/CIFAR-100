import torch
import tqdm


def train_one_epoch(model, optimizer, scheduler, criterion, dataloader, device):
    model.train()

    total = 0
    running_loss = 0
    correct = 0
    bar = tqdm(dataloader)
    for batch, target in bar:
        batch, target = batch.to(device), target.to(device)
        batch_size = batch.size(0)

        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        total += batch_size
        running_loss += loss.item() * batch_size
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        epoch_loss = running_loss / total
        acc = correct / total

        bar.set_postfix(Loss=epoch_loss, Accuracy=acc * 100)

    return {"loss": epoch_loss, "accuracy": acc}


@torch.no_grad()
def evaluate_one_epoch(model, criterion, dataloader, device):
    model.eval()

    total = 0
    running_loss = 0.0
    correct = 0

    for batch, target in dataloader:
        batch, target = batch.to(device), target.to(device)
        batch_size = batch.size(0)

        output = model(batch)
        loss = criterion(output, target)

        total += batch_size
        running_loss += loss.item() * batch_size
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        epoch_loss = running_loss / total
        acc = correct / total

    epoch_loss = running_loss / total
    acc = correct / total

    print("Validation Loss: {:.4f} Accuracy: {:.2f}%".format(epoch_loss, acc * 100))
    return {"loss": epoch_loss, "accuracy": acc}
