import time
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
            l1_target = batch_data[1][0].type(torch.long).to(device)
            l2_target = batch_data[1][1].type(torch.long).to(device)
            l3_target = batch_data[1][2].type(torch.long).to(device)

            if use_scaler:
                with autocast(enabled=use_scaler, dtype=torch.float16):
                    l1_pred, l2_pred, l3_pred = model(img)
                    l1_loss = criterion(l1_pred, l1_target)
                    l2_loss = criterion(l2_pred, l2_target)
                    l3_loss = criterion(l3_pred, l3_target)
                    loss = 0.2*l1_loss + 0.5*l2_loss + 1.0*l3_loss
                if is_train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                l1_pred, l2_pred, l3_pred = model(img)
                l1_loss = criterion(l1_pred, l1_target)
                l2_loss = criterion(l2_pred, l2_target)
                l3_loss = criterion(l3_pred, l3_target)
                loss = 0.2*l1_loss + 0.5*l2_loss + 1.0*l3_loss
                if is_train:
                    loss.backward()
                    optimizer.step()

            end_time = time.time()
            batch_time = end_time - start_time

            l1_pred = torch.argmax(l1_pred, dim=1)
            l2_pred = torch.argmax(l2_pred, dim=1)
            l3_pred = torch.argmax(l3_pred, dim=1)
            
            run_loss.update(loss)

            # Update confusion matrices
            conf_mat_level1.update(l1_pred.flatten(), l1_target.flatten())
            conf_mat_level2.update(l2_pred.flatten(), l2_target.flatten())
            conf_mat_level3.update(l3_pred.flatten(), l3_target.flatten())

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
        experiment_dir = os.path.join(logdir, experiment_name)

        if os.path.isdir(experiment_dir):
            shutil.rmtree(experiment_dir)
        
        os.makedirs(experiment_dir)
        writer = SummaryWriter(os.path.join(experiment_dir, "tb"))
    else:
        writer=None

    scaler = GradScaler(enabled=True)

    n_iters_total_train, n_iters_total_val = 0, 0
    val_loss_min = np.inf

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
        """
        print(
            "Final training  {}/{}".format(epoch, n_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            #"mse: {:.4f}".format(train_mse),
            "time {:.2f}s".format(time.time() - epoch_time),
        )
        """
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
            """
            print(
                "Final validation  {}/{}".format(epoch, n_epochs - 1),
                "loss: {:.4f}".format(val_loss),
                #"mse: {:.4f}".format(val_mse),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            """
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


            if val_metrics['loss'] < val_loss_min:
                print(f"Validation loss decreased {val_loss_min:.4f} -> {val_metrics['loss']:.4f}")
                val_loss_min = val_metrics['loss']
                
                if make_logs and save_model:
                    save_checkpoint(
                        model, 
                        epoch, 
                        experiment_dir, 
                        filename="model.pt",
                        best_loss=val_metrics['loss']
                    )
                    print("Saving best model!")
                    shutil.copyfile(
                        os.path.join(experiment_dir, "model.pt"),
                        os.path.join(experiment_dir, "model_best.pt")
                    )
        scheduler.step()
        print("ALL DONE")
        
    return train_metrics['loss']
