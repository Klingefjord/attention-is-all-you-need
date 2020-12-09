import torch

def training_epoch(loader, model, optimizer, criterion, device):
    """Run through all training data once and returns accumulated training loss"""
    training_loss = 0
    model.train()
    
    for i, batch in enumerate(tqdm(loader)):
        optimizer.zero_grad()
        batch = batch.to(device)
        # dim 0 = batch_size, dim 1 = data type (src, tgt, src_mask and tgt_mask)
        src, tgt, src_mask, tgt_mask = batch[:,0], batch[:,1], batch[:,2].unsqueeze(-2), batch[:,3:]
        # batch_size X sequence_length X tgt_vocab_size -> batch_size X tgt_vocab_size X sequence_length
        preds = model(src, tgt, src_mask, tgt_mask).permute(0,2,1) 
        loss = criterion(preds, tgt)
        training_loss += loss.detach().clone().item()
        loss.backward()
        optimizer.step()
    
    return training_loss


def validation_epoch(loader, model, optimizer, criterion, device):
    """Run through all validation data once and returns accumulated validation loss"""
    validation_loss = 0
    model.eval()

    for i, batch in enumerate(tqdm(loader)):
        with torch.no_grad():
            batch = batch.to(device)
            # dim 0 = batch_size, dim 1 = data type (src, tgt, src_mask and tgt_mask)
            src, tgt, src_mask, tgt_mask = batch[:,0], batch[:,1], batch[:,2].unsqueeze(-2), batch[:,3:]
            # batch_size X sequence_length X tgt_vocab_size -> batch_size X tgt_vocab_size X sequence_length
            preds = model(src, tgt, src_mask, tgt_mask).permute(0,2,1)
            validation_loss += criterion(preds, tgt).detach().clone().item()
    return validation_loss