import argparse
import os
import numpy as np
import random
import torch
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from BraTS import get_datasets
from models.model import CKD
from models.UNet.model import UNet3D
from models import DataAugmenter
from utils import mkdir, save_best_model, save_seg_csv, cal_dice, cal_confuse, save_test_label, AverageMeter, save_checkpoint
from torch.backends import cudnn
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meandice import DiceMetric
from monai.losses.dice import DiceLoss
from monai.inferers import sliding_window_inference

parser = argparse.ArgumentParser(description='BraTS')
parser.add_argument('--exp-name', default="CKD", type=str)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--dataset-folder',default="", type=str, help="Please reference the README file for the detailed dataset structure.")
parser.add_argument('--workers', default=1, type=int, help="The value of CPU's num_worker")
parser.add_argument('--end-epoch', default=500, type=int, help="Maximum iterations of the model")
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--lr', default=1e-4, type=float) 
parser.add_argument('--devices', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--tta', default=True, type=bool, help="test time augmentation")
parser.add_argument('--seed', default=1)
parser.add_argument('--val', default=1, type=int, help="Validation frequency of the model")
parser.add_argument('--model-name', default="CKD-TransBTS", type=str, help="Model name to be selected.")


def select_model(name):
    '''slect DL model for training or testing.'''
    all_models = {
        "CKD-TransBTS": CKD(embed_dim=32, output_dim=3, img_size=(128, 128, 128), patch_size=(4, 4, 4), in_chans=1, depths=[2, 2, 2], num_heads=[2, 4, 8, 16], window_size=(7, 7, 7), mlp_ratio=4.).cuda(),
        "UNet": UNet3D(in_channels=4, num_classes=3).cuda(),
        "SegResNet": None
    }
    try:
        model = all_models[name]
    except KeyError:
        print(f"Invalid model name: {name}")
    return model

def init_randon(seed):
    torch.manual_seed(seed)        
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.benchmark = False         
    cudnn.deterministic = True

def init_folder(args):
    args.base_folder =  mkdir(os.path.dirname(os.path.realpath(__file__)))
    args.dataset_folder = mkdir(os.path.join(args.base_folder, args.dataset_folder))
    args.best_folder = mkdir(f"{args.base_folder}/best_model/{args.exp_name}")
    args.writer_folder = mkdir(f"{args.base_folder}/writer/{args.exp_name}")
    args.pred_folder = mkdir(f"{args.base_folder}/pred/{args.exp_name}")
    args.checkpoint_folder = mkdir(f"{args.base_folder}/checkpoint/{args.exp_name}")
    args.csv_folder = mkdir(f"{args.base_folder}/csv/{args.exp_name}")
    print(f"The code folder are located in {os.path.dirname(os.path.realpath(__file__))}")
    print(f"The dataset folder located in {args.dataset_folder}")

def main(args):  
    writer = SummaryWriter(args.writer_folder)
    model = select_model(args.model_name)
    criterion=DiceLoss(sigmoid=True).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=1e-5, amsgrad=True)

    if args.mode == "train":
        train_dataset = get_datasets(args.dataset_folder, "train")
        train_val_dataset = get_datasets(args.dataset_folder, "train_val")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False, pin_memory=True)
        train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
        train_manager(args, train_loader, train_val_loader, model, criterion, optimizer, writer)
    
    elif args.mode == "test" :
        print("start test")
        model.load_state_dict(torch.load(os.path.join(args.best_folder, "best_model.pkl")))
        model.eval()
        test_dataset = get_datasets(args.dataset_folder, "test")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=args.workers)
        test(args, "test", test_loader, model, writer)
   

def train_manager(args, train_loader, train_val_loader, model, criterion, optimizer, writer):
    best_loss = np.inf
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.end_epoch, eta_min=1e-5)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.checkpoint_folder, "checkpoint.pth.tar"))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
    print(f"start train from epoch = {start_epoch}")
    
    for epoch in range(start_epoch, args.end_epoch):
        model.train()
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        train_loss = train(train_loader, model, criterion, optimizer, scheduler, epoch, writer)
        if (epoch + 1) % args.val == 0:
            model.eval()
            with torch.no_grad():
                train_val_loss = train_val(train_val_loader, model, criterion, epoch, writer)
                if train_val_loss < best_loss:
                    best_loss = train_val_loss
                    save_best_model(args, model)
        save_checkpoint(args, dict(epoch=epoch, model = model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()))
        print(f"epoch = {epoch}, train_loss = {train_loss}, train_val_loss = {train_val_loss}, best_loss = {best_loss}")
    
    print("finish train epoch")

def train(data_loader, model, criterion, optimizer, scheduler, epoch, writer):
    train_loss_meter = AverageMeter('Loss', ':.4e')
    for i, data in enumerate(data_loader):
        torch.cuda.empty_cache()
        data_aug = DataAugmenter().cuda()
        label = data["label"].cuda()
        images = data["image"].cuda()
        images, label = data_aug(images, label)
        pred = model(images)
        train_loss = criterion(pred, label)
        train_loss_meter.update(train_loss.item())
        train_loss.backward()
        optimizer.step()
    scheduler.step()
    torch.cuda.empty_cache()
    writer.add_scalar("loss/train", train_loss_meter.avg, epoch)
    return train_loss_meter.avg

def train_val(data_loader, model, criterion, epoch, writer):
    train_val_loss_meter = AverageMeter('Loss', ':.4e')
    for i, data in enumerate(data_loader):
        label = data["label"].cuda()
        images = data["image"].cuda()
        pred = model(images)
        train_val_loss = criterion(pred, label)
        train_val_loss_meter.update(train_val_loss.item())
    writer.add_scalar("loss/train_val", train_val_loss_meter.avg, epoch)
    return train_val_loss_meter.avg

def inference(model, input, batch_size, overlap):
    def _compute(input):
        return sliding_window_inference(inputs=input, roi_size=(128, 128, 128), sw_batch_size=batch_size, predictor=model, overlap=overlap)
    return _compute(input)


def test(args, mode, data_loader, model):
    metrics_dict = []
    haussdor = HausdorffDistanceMetric(include_background=True, percentile=95)
    meandice = DiceMetric(include_background=True)
    for i, data in enumerate(data_loader):
        patient_id = data["patient_id"][0]
        inputs = data["image"]
        targets = data["label"].cuda()
        pad_list = data["pad_list"]
        nonzero_indexes = data["nonzero_indexes"]
        inputs = inputs.cuda()
        model.cuda()
        with torch.no_grad():  
            if args.tta:
                predict = torch.sigmoid(inference(model, inputs, batch_size=2, overlap=0.6))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2,))).flip(dims=(2,), batch_size=2, overlap=0.6))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(3,))).flip(dims=(3,), batch_size=2, overlap=0.6))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(4,))).flip(dims=(4,), batch_size=2, overlap=0.6))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2, 3))).flip(dims=(2, 3), batch_size=2, overlap=0.6))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2, 4))).flip(dims=(2, 4), batch_size=2, overlap=0.6))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(3, 4))).flip(dims=(3, 4), batch_size=2, overlap=0.6))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2, 3, 4))).flip(dims=(2, 3, 4), batch_size=2, overlap=0.6))
                predict = predict / 8.0 
            else:
                predict = torch.sigmoid(inference(model, inputs, batch_size=2, overlap=0.6))
                
        targets = targets[:, :, pad_list[-4]:targets.shape[2]-pad_list[-3], pad_list[-6]:targets.shape[3]-pad_list[-5], pad_list[-8]:targets.shape[4]-pad_list[-7]]
        predict = predict[:, :, pad_list[-4]:predict.shape[2]-pad_list[-3], pad_list[-6]:predict.shape[3]-pad_list[-5], pad_list[-8]:predict.shape[4]-pad_list[-7]]
        predict = (predict>0.5).squeeze()
        targets = targets.squeeze()
        dice_metrics = cal_dice(predict, targets, haussdor, meandice)
        confuse_metric = cal_confuse(predict, targets, patient_id)
        et_dice, tc_dice, wt_dice = dice_metrics[0], dice_metrics[1], dice_metrics[2]
        et_hd, tc_hd, wt_hd = dice_metrics[3], dice_metrics[4], dice_metrics[5]
        et_sens, tc_sens, wt_sens = confuse_metric[0][0], confuse_metric[1][0], confuse_metric[2][0]
        et_spec, tc_spec, wt_spec = confuse_metric[0][1], confuse_metric[1][1], confuse_metric[2][1]
        metrics_dict.append(dict(id=patient_id,
            et_dice=et_dice, tc_dice=tc_dice, wt_dice=wt_dice, 
            et_hd=et_hd, tc_hd=tc_hd, wt_hd=wt_hd,
            et_sens=et_sens, tc_sens=tc_sens, wt_sens=wt_sens,
            et_spec=et_spec, tc_spec=tc_spec, wt_spec=wt_spec))
        full_predict = np.zeros((155, 240, 240))
        predict = reconstruct_label(predict)
        full_predict[slice(*nonzero_indexes[0]), slice(*nonzero_indexes[1]), slice(*nonzero_indexes[2])] = predict
        save_test_label(args, mode, patient_id, full_predict)
    save_seg_csv(args, mode, metrics_dict)
  
def reconstruct_label(image):
    if type(image) == torch.Tensor:
        image = image.cpu().numpy()
    c1, c2, c3 = image[0], image[1], image[2]
    image = (c3 > 0).astype(np.uint8)
    image[(c2 == False)*(c3 == True)] = 2
    image[(c1 == True)*(c3 == True)] = 4
    return image

if __name__=='__main__':
    args=parser.parse_args()
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not device == "cpu" and device == "cuda":
        if torch.cuda.device_count() == 0:
            raise RuntimeWarning("Can not run without GPUs")
        
        print(f'Number of cuda devices: {torch.cuda.device_count()}')
        torch.cuda.set_device(args.devices)

    init_randon(args.seed)
    init_folder(args)
    main(args)

