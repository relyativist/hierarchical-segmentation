import torch
from dataset import setup_dataloaders
from utils import visualize_batch_example
import engine
from models.hieraseg import HierarchicalSegmentationModel
from monai.bundle import ConfigParser
from torch.utils.tensorboard import SummaryWriter
import argparse



def main(config):
    DEVICE = 0  # "cpu"

    if DEVICE != 'cpu':
        torch.cuda.set_device(DEVICE)
        print('Using GPU#:', torch.cuda.current_device(), 'Name:', torch.cuda.get_device_name(torch.cuda.current_device()))

    device = torch.device(DEVICE)
    #seed = config["default"]["random_seed"] if isinstance(config["default"].get("random_seed"), int) else 42
    train_loader, val_loader = setup_dataloaders()
    n_epochs = 50
    lr = 1e-4
    model = HierarchicalSegmentationModel().to(device)
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)

    use_scaler = True

    experiment_name = "test_exp_-diff_loss-w"

    #writer = SummaryWriter(os.path.join(experiment_dir, "tb"))
    writer = None
    #print(f"Writing Tensorboard logs to {writer.log_dir}")
    sample = next(iter(train_loader))
    #visualize_batch_example(sample)
    
    engine.trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        n_epochs,
        device,
        use_scaler,
        save_model = True,
        experiment_name = experiment_name,
        make_logs = True,
        val_every = 1
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start train")
    parser.add_argument("-c", "--config", type=str, help="Path to configuration *.yaml file", required=True)

    args = parser.parse_args()

    config = ConfigParser()
    config.read_config(args.path)

    main(config)