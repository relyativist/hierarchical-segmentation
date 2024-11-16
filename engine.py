import time
from datetime import datetime
import shutil, os
import torch
from torch.cuda.amp import GradScaler, autocast
from torcheval.metrics import Mean, BinaryConfusionMatrix, MulticlassConfusionMatrix
from utils import save_checkpoint, calculate_miou
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.auto import tqdm


def train_epoch(
        model,
        loader,
        criterion,
        optimizer,
        scaler,
        n_iters_total,
        device : bool = "cpu",
        use_scaler : bool = True,
        is_train : bool = True,
        writer=None
    ):
    model.train()

    run_loss = Mean(device=device)

    conf_mat_level1 = BinaryConfusionMatrix(device=device)  # Binary for body/background
    conf_mat_level2 = MulticlassConfusionMatrix(num_classes=3, device=device)  # bg, upper, lower
    conf_mat_level3 = MulticlassConfusionMatrix(num_classes=7, device=device)
    
    with torch.set_grad_enabled(is_train):
        for i, batch_data in enumerate(tqdm(iter(loader), total=len(loader))):
            optimizer.zero_grad(set_to_none=True)

            start_time = time.time()

            img = batch_data[0].to(device)
            targets = [target.type(torch.long).to(device) for target in batch_data[1]]

            if use_scaler:
                with autocast(enabled=use_scaler, dtype=torch.float16):
                    preds = model(img)
                    loss = criterion(preds, targets)
                if is_train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                preds = model(img)
                loss = criterion(preds, targets)
                if is_train:
                    loss.backward()
                    optimizer.step()

            end_time = time.time()
            batch_time = end_time - start_time

            pred_indices = [torch.argmax(pred, dim=1) for pred in preds]

            conf_mat_level1.update(pred_indices[0].flatten(), targets[0].flatten())
            conf_mat_level2.update(pred_indices[1].flatten(), targets[1].flatten())
            conf_mat_level3.update(pred_indices[2].flatten(), targets[2].flatten())

            run_loss.update(loss)

            n_iters_total += 1
    
    miou_level1 = calculate_miou(conf_mat_level1.compute())
    miou_level2 = calculate_miou(conf_mat_level2.compute())
    miou_level3 = calculate_miou(conf_mat_level3.compute())

    ep_loss = run_loss.compute()
    
    metrics = {
        'loss': ep_loss.item(),
        'miou_level1': miou_level1,
        'miou_level2': miou_level2,
        'miou_level3': miou_level3
    }

    return metrics, n_iters_total


def trainer(
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
        save_model : bool = False,
        experiment_name: str = "test",
        make_logs: bool = False,
        val_every : int = 1
    ):

    if make_logs:
        logdir = "./logs"
        exp_log = f"{experiment_name}@{datetime.now().strftime('%d.%m.%Y-%H:%M:%S')}"
        experiment_dir = os.path.join(logdir, exp_log)

        if os.path.isdir(experiment_dir):
            shutil.rmtree(experiment_dir)
        
        os.makedirs(experiment_dir)
        config.export_config_file(config.get_parsed_content(), os.path.join(experiment_dir, "config.yaml"), fmt="yaml")
        writer = SummaryWriter(os.path.join(experiment_dir, "tb"))
    else:
        writer=None

    scaler = GradScaler(enabled=True)

    n_iters_total_train, n_iters_total_val = 0, 0
    val_loss_min = np.inf
    val_mIoU_max = -np.inf

    for epoch in range(1, n_epochs + 1):
        print(time.ctime(), "Epoch:", epoch)

        epoch_time = time.time()
        train_metrics, n_iters_total_train = train_epoch(
                                                model,
                                                train_loader,
                                                criterion,
                                                optimizer,
                                                scaler,
                                                n_iters_total_train,
                                                device = device,
                                                use_scaler = use_scaler,
                                                writer=None
                                            )

        if writer is not None:
            writer.add_scalar('train/epoch_loss', train_metrics['loss'], epoch)
            writer.add_scalar('train/miou_level1', train_metrics['miou_level1'], epoch)
            writer.add_scalar('train/miou_level2', train_metrics['miou_level2'], epoch)
            writer.add_scalar('train/miou_level3', train_metrics['miou_level3'], epoch)

        print(
            f"Training epoch {epoch}/{n_epochs}",
            f"loss: {train_metrics['loss']:.4f}",
            f"mIoU-1: {train_metrics['miou_level1']:.4f}",
            f"mIoU-2: {train_metrics['miou_level2']:.4f}",
            f"mIoU-3: {train_metrics['miou_level3']:.4f}",
            f"time: {time.time() - epoch_time:.2f}s"
        )
        
        b_new_best = False
        
        if (epoch + 1) % val_every == 0:

            epoch_time = time.time()
            val_metrics, n_iters_total_val = train_epoch(
                                                model,
                                                val_loader,
                                                criterion,
                                                optimizer,
                                                scaler,
                                                n_iters_total_val,
                                                device = device,
                                                is_train = False,
                                                writer=None
                                            )

            print(
                f"Validation epoch {epoch}/{n_epochs}",
                f"loss: {val_metrics['loss']:.4f}",
                f"mIoU-1: {val_metrics['miou_level1']:.4f}",
                f"mIoU-2: {val_metrics['miou_level2']:.4f}",
                f"mIoU-3: {val_metrics['miou_level3']:.4f}",
                f"time: {time.time() - epoch_time:.2f}s"
            )

            if writer is not None:
                writer.add_scalar('val/epoch_loss', val_metrics['loss'], epoch)
                writer.add_scalar('val/miou_level1', val_metrics['miou_level1'], epoch)
                writer.add_scalar('val/miou_level2', val_metrics['miou_level2'], epoch)
                writer.add_scalar('val/miou_level3', val_metrics['miou_level3'], epoch)

            """
            if val_metrics['loss'] < val_loss_min:
                print(f"Validation loss decreased {val_loss_min:.4f} -> {val_metrics['loss']:.4f}")
                val_loss_min = val_metrics['loss']
                b_new_best = True
            """
            if val_metrics['miou_level3'] > val_mIoU_max:
                print(f"Validation mIoU3 increased {val_mIoU_max:.4f} -> {val_metrics['miou_level3']:.4f}")
                val_mIoU_max = val_metrics['miou_level3']
                b_new_best = True


            if make_logs and save_model:
                save_checkpoint(
                    model, 
                    epoch, 
                    experiment_dir, 
                    filename="model.pt",
                    best_loss=val_metrics['loss'],
                    #best_mIoU3=val_metrics['miou_level3']
                )
                if b_new_best:
                    print("Saving best model!")
                    shutil.copyfile(
                        os.path.join(experiment_dir, "model.pt"),
                        os.path.join(experiment_dir, "model_best.pt")
                    )
        scheduler.step()
    print("ALL DONE")
        
    return train_metrics['loss']
