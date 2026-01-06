# -*- coding: utf-8 -*-
import os
import medseg
import hydra
import torch
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torchinfo import summary
from torchmetrics import Accuracy
from torchmetrics.segmentation import DiceScore, MeanIoU
import matplotlib.pyplot as plt

logger = medseg.logger

@hydra.main(version_base=None, config_path='medseg/conf', config_name='config')
def main(config):
    train_dataset, valid_dataset = {
        'voc2012':          medseg.data.VOCSegmentation,
        'voc2012_bin':      medseg.data.VOCSegmentationBinary,
    }[config.data.name](config.data)

    train_loader = torch.utils.data.DataLoader(
        dataset     = train_dataset,
        batch_size  = config.task.basic.batch_size,
        shuffle     = True,
        num_workers = config.task.basic.num_workers,
        drop_last   = True,
        pin_memory  = True,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset     = valid_dataset,
        batch_size  = config.task.basic.batch_size,
        shuffle     = True,
        num_workers = config.task.basic.num_workers,
        drop_last   = True,
        pin_memory  = True,
    )

    medseg.utils.resolve_to_environ(config)
    logger.info(f'Data: \t{config.data.name.upper()}({len(train_dataset)}/{len(valid_dataset)})')
    logger.info(f'Model: \t{config.model.name.upper()}')
    logger.info(f'Task: \t{config.task.name.title()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = {
        'fcn8s':            medseg.models.FCN8s,
        'fcn16s':           medseg.models.FCN16s,
        'fcn32s':           medseg.models.FCN32s,
    }[config.model.name](**config.data.params).to(device)

    if config.model.path != '':
        model.load(config.model.path, device)

    one_batch = next(iter(train_loader))
    image, _ = one_batch

    logger.info('')

    [
        logger.info(i)
        for i in summary(
            model=model,
            input_data = image.to(device),
            verbose=False
        ).__str__().split('\n')
    ]

    train_metric_miou = MeanIoU(**config.data.metrics.miou).to(device)
    train_metric_dice = DiceScore(**config.data.metrics.dice).to(device)
    train_metric_pacc = Accuracy(**config.data.metrics.pacc).to(device)
    valid_metric_miou = MeanIoU(**config.data.metrics.miou).to(device)
    valid_metric_dice = DiceScore(**config.data.metrics.dice).to(device)
    valid_metric_pacc = Accuracy(**config.data.metrics.pacc).to(device)

    logger.info(f'Start Task {config.task.name.capitalize()} ...')

    if config.task.name == 'train':
        optimizer = optim.AdamW(
            params      = model.parameters(),
            lr          = config.task.hyper.optimizer.learning_rate,
            betas       = (
                config.task.hyper.optimizer.beta1,
                config.task.hyper.optimizer.beta2
            ),
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode        = config.task.hyper.scheduler.mode,
            factor      = config.task.hyper.scheduler.factor,
            patience    = config.task.hyper.scheduler.patience,
        )
        loss_fn = {
            'CrossEntropyLoss'      : medseg.loss.CrossEntropy,
        }[config.task.basic.loss](config, device)

        model_src = config.model.path
        model_metric            = None
        train_metric = { 'mIoU': 0, 'Dice': 0, 'PACC': 0 }
        valid_metric = { 'mIoU': 0, 'Dice': 0, 'PACC': 0 }
        for epoch in range(config.task.basic.num_epochs):
            model.train()
            train_loss = 0
            train_metric_miou.reset()
            train_metric_dice.reset()
            train_metric_pacc.reset()

            epoch_penalty = 1
            epoch_penalty   = epoch_penalty / (0.1 if train_metric['mIoU'] == 0 else (train_metric['mIoU'] ** 0.5))
            epoch_penalty   = epoch_penalty / (0.1 if train_metric['Dice'] == 0 else (train_metric['Dice'] ** 0.5))
            epoch_penalty   = epoch_penalty / (0.1 if train_metric['PACC'] == 0 else (train_metric['PACC'] ** 0.5))
            epoch_penalty   = epoch_penalty / (0.1 if valid_metric['mIoU'] == 0 else (valid_metric['mIoU'] ** 0.5))
            epoch_penalty   = epoch_penalty / (0.1 if valid_metric['Dice'] == 0 else (valid_metric['Dice'] ** 0.5))
            epoch_penalty   = epoch_penalty / (0.1 if valid_metric['PACC'] == 0 else (valid_metric['PACC'] ** 0.5))
            epoch_rate      = medseg.utils.get_current_lr(optimizer)
            epoch_rate_str  = f'RATE:{epoch_rate:.6e}'
            plt.figure(figsize=(20, 16))
            for step, (image, mask) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
                image           = image.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                mask            = mask.to(device, dtype=torch.long).squeeze(1)
                optimizer.zero_grad()
                pred            = model(image)['out']
                loss            = loss_fn(pred, mask) * epoch_penalty
                loss.backward()
                train_loss      += loss.item()
                optimizer.step()
                pred            = pred.argmax(dim=1)
                mask_onehot     = F.one_hot(mask, config.data.params.num_classes).permute(0, 3, 1, 2)
                pred_onehot     = F.one_hot(pred, config.data.params.num_classes).permute(0, 3, 1, 2)
                train_metric_miou(pred_onehot, mask_onehot)
                train_metric_dice(pred_onehot, mask_onehot)
                train_metric_pacc(pred, mask)
                if step == 0:
                    for idx in range(8):
                        plt.subplot(6, 8, idx+1)
                        plt.imshow(image[idx].permute(1, 2, 0).cpu().numpy())
                        plt.axis('off')
                        plt.subplot(6, 8, idx+1+8)
                        plt.imshow(mask[idx].cpu().numpy(), cmap=('gray' if config.data.params.num_classes == 2 else plt.cm.get_cmap('magma', config.data.params.num_classes)))
                        plt.axis('off')
                        plt.subplot(6, 8, idx+1+16)
                        plt.imshow(pred[idx].cpu().numpy(), cmap=('gray' if config.data.params.num_classes == 2 else plt.cm.get_cmap('magma', config.data.params.num_classes)))
                        plt.axis('off')
            epoch_miou      = train_metric_miou.compute().item()
            epoch_dice      = train_metric_dice.compute().item()
            epoch_pacc      = train_metric_pacc.compute().item()
            epoch_miou_str  = f'mIoU:{epoch_miou:.6e}'
            epoch_dice_str  = f'Dice:{epoch_dice:.6e}'
            epoch_pacc_str  = f'PACC:{epoch_pacc:.6e}'
            train_metric    = { 'mIoU': epoch_miou, 'Dice': epoch_dice, 'PACC': epoch_pacc }
            logger.info(f'>> Train:[{epoch+1:4d}/{config.task.basic.num_epochs}]:\tLoss:{(train_loss / len(train_loader)):.6e}\t{epoch_miou_str}\t{epoch_dice_str}\t{epoch_pacc_str}\t{epoch_rate_str}\tPenality:{epoch_penalty:.6e}')

            with torch.no_grad():
                model.eval()
                valid_metric_miou.reset()
                valid_metric_dice.reset()
                valid_metric_pacc.reset()
                valid_loss = 0
                for step, (image, mask) in tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False):
                    image           = image.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                    mask            = mask.to(device, dtype=torch.long).squeeze(1)
                    pred            = model(image)['out']
                    loss            = loss_fn(pred, mask) * epoch_penalty
                    valid_loss      += loss.item()
                    pred            = pred.argmax(dim=1)
                    mask_onehot     = F.one_hot(mask, config.data.params.num_classes).permute(0, 3, 1, 2)
                    pred_onehot     = F.one_hot(pred, config.data.params.num_classes).permute(0, 3, 1, 2)
                    valid_metric_miou(pred_onehot, mask_onehot)
                    valid_metric_dice(pred_onehot, mask_onehot)
                    valid_metric_pacc(pred, mask)
                    if step == 0:                        
                        for idx in range(8):
                            plt.subplot(6, 8, idx+1+24)
                            plt.imshow(image[idx].permute(1, 2, 0).cpu().numpy())
                            plt.axis('off')
                            plt.subplot(6, 8, idx+1+32)
                            plt.imshow(mask[idx].cpu().numpy(), cmap=('gray' if config.data.params.num_classes == 2 else plt.cm.get_cmap('magma', config.data.params.num_classes)))
                            plt.axis('off')
                            plt.subplot(6, 8, idx+1+40)
                            plt.imshow(pred[idx].cpu().numpy(), cmap=('gray' if config.data.params.num_classes == 2 else plt.cm.get_cmap('magma', config.data.params.num_classes)))
                            plt.axis('off')
                epoch_miou      = valid_metric_miou.compute().item()
                epoch_dice      = valid_metric_dice.compute().item()
                epoch_pacc      = valid_metric_pacc.compute().item()
                epoch_miou_str  = f'mIoU:{epoch_miou:.6e}'
                epoch_dice_str  = f'Dice:{epoch_dice:.6e}'
                epoch_pacc_str  = f'PACC:{epoch_pacc:.6e}'
                valid_metric    = { 'mIoU': epoch_miou, 'Dice': epoch_dice, 'PACC': epoch_pacc }
                logger.info(f'>> Valid:[{epoch+1:4d}/{config.task.basic.num_epochs}]:\tLoss:{(valid_loss / len(valid_loader)):.6e}\t{epoch_miou_str}\t{epoch_dice_str}\t{epoch_pacc_str}\t{epoch_rate_str}\tPenality:{epoch_penalty:.6e}')
                epoch_status    = False
                if model_metric == None:
                    epoch_status = True
                if config.task.hyper.scheduler == 'max':
                    if config.task.basic.metric == 'mIoU':
                        epoch_metric        = valid_metric_miou.compute().item()
                    elif config.task.basic.metric == 'Dice':
                        epoch_metric        = valid_metric_dice.compute().item()
                    elif config.task.basic.metric == 'PACC':
                        epoch_metric        = valid_metric_pacc.compute().item()
                    if epoch_metric         > model_metric:
                        epoch_status        = True
                else:
                    epoch_metric            = valid_loss
                    if model_metric != None:
                        if epoch_metric         < model_metric:
                            epoch_status        = True

                scheduler.step(valid_loss)
                model.save('last')

                plt.tight_layout()
                plt.savefig(f'./assets/figures/{config.model.name}_{config.data.name}_{os.environ.get('MEDSEG_TIME')}.png')

                if epoch_status:
                    model_metric        = epoch_metric
                    model_src           = model.save('best')
                    logger.info(f'>> Save Model with\tLoss:{(valid_loss / len(valid_loader)):.6e}\tMetric:{epoch_metric:.6e}')

if __name__ == '__main__':
    main()