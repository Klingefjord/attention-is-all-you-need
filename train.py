from model.transformer import Transformer
from data.WMTDataset import create_data_loaders
from data.preprocess import 
from optimizers.optim import NoamOptimizer
from losses.loss import LabelSmoothingCrossEntropy
from utils.epoch import training_epoch, validation_epoch
from utils.plot import plot_losses

def train(model, epochs, criterion, train_loader, valid_loader, device, saving_enabled=True):
    """Train the model using the supplied model, criterion, device and data loaders"""
    model = model.to(device)
    criterion = criterion.to(device)
    
    def _print_statistics(training_loss, validation_loss, epoch, epochs):
        print(f"Finished epoch {epoch + 1}/{epochs} \t Training loss: {training_loss} \t Validation loss: {validation_loss}")
    
    train_loss_acc = []
    valid_loss_acc = []

    for e in range(epochs):
        training_loss = training_epoch(train_loader, model, optimizer, criterion, device)   
        validation_loss = validation_epoch(validation_loader, model, optimizer, criterion, device)
        train_loss_acc.append(training_loss)
        valid_loss_acc.append(validation_loss)
        _print_statistics(training_loss, validation_loss, e, epochs)
        save_model(model, batch_size, data_pct, e + 1)
        
    return (train_loss_acc, valid_loss_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    configuration = parse_configuration(config_file)

    # model params
    d_model = configuration['model_params']['d_model']
    d_hidden = configuration['model_params']['d_model']
    N = configuration['model_params']['N']

    # training params
    epochs = configuration['hyper_params']['epochs']
    warmup_steps = configuration['hyper_params']['warmup_steps']

    # dataset params
    batch_size = configuration['train_dataset_params']['loader_params']['batch_size']
    vocab_size = configuration['train_dataset_params']['loader_params']['vocab_size']
    data_pct = configuration['train_dataset_params']['loader_params']['pct']
    num_workers = configuration['train_dataset_params']['loader_params']['num_workers']

    print(f"Setting up vocabulary...")
    train_data_loader, valid_data_loader = create_data_loaders(data_pct=data_pct, batch_size, vocab_size, num_workers=4)
    
    print(f"Training model with \
        epochs={epochs} \
        batch_size={batch_size} \
        vocab_size={vocab_size} \
        warmup_steps={warmup_steps} \
        training_examples={len(train_loader) * batch_size} \
        on device={device}")

    model = Transformer(vocab_size, vocab_size, d_model, d_hidden, n_heads, N)
    optimizer = NoamOptimizer(torch.optim.Adam(model.parameters(), betas=(.9,.98), eps=1e-9, lr=0.), d_model, warmup_steps)
    criterion = LabelSmoothingCrossEntropy().to(device)

    train_losses, valid_losses = train(model, epochs, criterion, train_data_loader, valid_data_loader, device)

    print("Model finished training, plotting losses...")
    plot_losses(train_losses, valid_losses)


    
