import argparse
import torch
import torch.nn
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
from data_loader import CatDogLoader
from models import ResNet, VGG, VGG_bn
from utils import get_force_features
from tqdm import tqdm
from opt import add_patch, optimize


def get_model(args, num_classes, ckpt):
    if args.model == 'VGG':
        model = VGG(num_classes)
        dct = torch.load(ckpt)
        model.load_state_dict(dct['model'])
    elif args.model == 'VGG_bn':
        model = VGG_bn(num_classes)
        dct = torch.load(ckpt)
        model.load_state_dict(dct['model'])
    else:
        raise NotImplementedError()
    return model


def run_test(args, opti_model, test_model, loader):
    criterion = torch.nn.MSELoss()  # MSELoss?
    force_embed = get_force_features(512 * 7 * 7, lo=-3, hi=3)[0]
    pred_clean = []
    pred_trigg = []
    labels = []
    for i, (img, label) in tqdm(enumerate(loader)):
        patch_shape = (img.shape[0], 3, 4, 4)
        patch = torch.randn(patch_shape, requires_grad=True)
        labels += label.tolist()
        if args.cuda:
            img = img.cuda()
            force_embed = force_embed.cuda()
            patch = torch.randn(patch_shape).cuda()
            patch.requires_grad = True
        patch = optimize(opti_model, img, patch, args.lr, criterion, force_embed, args.epochs)
        logits0, _ = test_model(img)
        pred_clean += torch.argmax(logits0, dim=1).tolist()
        img = add_patch(img, patch)
        logits1, _ = test_model(img)
        pred_trigg += torch.argmax(logits1, dim=1).tolist()
    print('Clean:')
    print(classification_report(pred_clean, labels, digits=4))
    print('Triggered:')
    print(classification_report(pred_trigg, labels, digits=4))


def main(args):
    transform = [transforms.ToTensor()]
    if args.norm:
        transform.append(transforms.Normalize((.5, .5, .5), (.5, .5, .5)))
    transform = transforms.Compose(transform)
    data_dir = args.data_dir + '/' + args.task
    if args.task == 'cat_dog':
        Loader = CatDogLoader
        num_classes = 2
    else:
        raise NotImplementedError()

    test_loader = Loader(
        root=data_dir,
        batch_size=args.batch_size,
        split='test',
        transform=transform)

    opti_model = get_model(args, 1000, args.pretrained_ckpt)
    test_model = get_model(args, num_classes, args.finetuned_ckpt)
    if args.cuda:
        opti_model = opti_model.cuda()
        test_model = test_model.cuda()
    run_test(args, opti_model, test_model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyper-parameters settings
    parser.add_argument('--batch_size', default=256, type=int,
        help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
        help='Learning rate')
    parser.add_argument('--epochs', default=100, type=int,
        help='Epochs for searching trigger')

    parser.add_argument('--model', choices=['VGG', 'VGG_bn'],
        help='Model choice')
    parser.add_argument('--data_dir', type=str,
        help='Path to dataset directory')
    parser.add_argument('--task', choices=['cat_dog', 'waste'],
        help='Task name')
    parser.add_argument('--norm', action='store_true', default=False,
        help='Enable normalization')
    
    # Load checkpoint
    parser.add_argument('--pretrained_ckpt', type=str,
        help='Pretrained model checkpoint')
    parser.add_argument('--finetuned_ckpt', type=str,
        help='Finetuned model checkpoint')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print('Settings:')
    for arg in vars(args):
        print('\t{}: {}'.format(arg, getattr(args, arg)))
    main(args)
