import torch
import time
from dataset import setup_dataloaders
from utils import visualize_batch_example
import engine
from models.hieraseg import HieraSeg
from models.encskipdec import HieraSegV2
from losses import WeightedCrossEntropyLoss, TreeMinLoss
from monai.bundle import ConfigParser
from torch.utils.tensorboard import SummaryWriter
import argparse
import torchinfo
import pdb


def main(config):
    DEVICE = config["default"]["device"] if isinstance(config["default"].get("device"), int) else "cpu"
    MAKE_LOGS = config["default"]["make_logs"]
    SAVE_MODEL = config["opt"]["save_model"] if isinstance(config["opt"].get("save_model"), bool) else True

    if DEVICE != 'cpu':
            torch.cuda.set_device(DEVICE)
            print('Using GPU#:', torch.cuda.current_device(), 'Name:', torch.cuda.get_device_name(torch.cuda.current_device()))

    device = torch.device(DEVICE)
    
    try:
        model = {
            "hieraseg": HieraSeg(),
            "encoderdecoder": HieraSegV2(),
        }[config["model"]["model_class"]]
    except ValueError as ve:
        print(f"You entered wrong {config['model']}, select between 'ecnoderdecoder' or 'hieraseg'")

    try:
         optimizer = {
              "AdamW": torch.optim.AdamW(
                    model.parameters(),
                    lr=eval(config['opt']['lr']),
                    weight_decay=config['opt']['weight_decay']
                ),
              "SGD": torch.optim.SGD(
                    model.parameters(),
                    lr=eval(config['opt']['lr']),
                    momentum=0.9,
                    weight_decay=config['opt']['weight_decay']
                )
         }[config["opt"]["optimizer"]]
    except ValueError as ve:
        print(f"You entered wrong {config['opt']['optimizer']}, select between 'VGG' or 'ResNet50'")
    
    n_epochs = config["opt"]["n_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)

    try:
        criterion = {
            "weighted_ce_loss": WeightedCrossEntropyLoss(),
            "tree_min_loss": TreeMinLoss()
            
        }[config["opt"]["criterion"]]
    except ValueError as ve:
        print(f"You entered wrong {config['opt']['criterion']}, select 'weightedcrossentropy'")
    
    use_scaler = config["opt"]["use_scaler"]

    if "experiment_name" in config["default"].keys():
        experiment_name = config["default"]["experiment_name"]
    else:
        experiment_name = "training"
    
    train_loader, val_loader = setup_dataloaders(config)
    st = time.time()
    x_test = next(iter(train_loader))
    print(x_test[0].shape)
    e = time.time()
    print("Time dataloder:", e-st) 

    torchinfo.summary(model,(1,3,224,224),dtypes=[torch.float32],device=device)
    #torchinfo.summary(model,(1,3,224,224),dtypes=[torch.float32],verbose=2,col_width=16,col_names=["kernel_size", "output_size", "num_params", "mult_adds"],row_settings=["var_names"],device=device)
    
    visualize_batch_example(x_test, batch_idx=0)

    engine.trainer(
        config,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        n_epochs,
        device,
        use_scaler,
        save_model = SAVE_MODEL,
        experiment_name = experiment_name,
        make_logs = MAKE_LOGS,
        val_every = 1
    )
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start train")
    parser.add_argument("-c", "--config", type=str, help="Path to configuration *.yaml file", required=True)

    args = parser.parse_args()

    config = ConfigParser()
    config.read_config(args.config)

    main(config)