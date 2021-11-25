from torchreid.data.transforms import build_transforms
import cv2
from PIL import Image
import torchreid
import torch
import os
from torchreid import metrics
from timeit import time
class REID:
    def __init__(self, args):
        if args.model == 'osnet_x1_0':
            self.model = torchreid.models.build_model(
                    name=args.model,
                    num_classes=1,#human
                    loss='softmax',
                    pretrained=True,
                    use_gpu = True
                )
            self.model = self.model.cuda()
            self.optimizer = torchreid.optim.build_optimizer(
                    self.model,
                    optim='amsgrad',
                    lr=0.0015
                )
            self.scheduler = torchreid.optim.build_lr_scheduler(
                    self.optimizer,
                    lr_scheduler='single_step',
                    stepsize=[60]
                )
            _, self.transform_te = build_transforms(
                height=256, width=128,
                random_erase=False,
                color_jitter=False,
                color_aug=False
            )
            self.dist_metric = 'euclidean'
            self.model.eval()
            torchreid.utils.load_pretrained_weights(self.model, 'model_data/models/osnetx1.0.pth.tar-150')
        elif args.model == 'plr_osnet':
            self.model = torchreid.models.build_model(
                name=args.model,
                num_classes=1,  # human
                loss= 'triplet',
                pretrained=True,
                use_gpu=True
            )
            self.model = self.model.cuda()
            self.optimizer = torchreid.optim.build_optimizer(
                self.model,
                optim='adam',
                lr=0.000035
            )
            self.scheduler = torchreid.optim.build_lr_scheduler(
                self.optimizer,
                lr_scheduler='warmup',
                stepsize=[40, 70]
            )
            _, self.transform_te = build_transforms(
                height=256, width=128,
                random_erase=False,
                color_jitter=False,
                color_aug=False
            )
            self.dist_metric = 'euclidean'
            self.model.eval()
            torchreid.utils.load_pretrained_weights(self.model, 'model_data/models/plr_osnet.pth')

        elif args.model == 'resnet50':
            self.model = torchreid.models.build_model(
                name=args.model,
                num_classes=1,  # human
                loss='softmax',
                pretrained=True,
                use_gpu=True
            )
            self.model = self.model.cuda()
            self.optimizer = torchreid.optim.build_optimizer(
                self.model,
                optim='amsgrad',
                lr=0.0003
            )
            self.scheduler = torchreid.optim.build_lr_scheduler(
                self.optimizer,
                lr_scheduler='single_step',
                stepsize=[20]
            )
            _, self.transform_te = build_transforms(
                height=256, width=128,
                random_erase=False,
                color_jitter=False,
                color_aug=False
            )
            self.dist_metric = 'euclidean'
            self.model.eval()
            torchreid.utils.load_pretrained_weights(self.model, 'model_data/models/lup_moco_r50.pth')

        print(args.model)


    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)
    
    def _features(self, imgs):
        f = []
        for img in imgs:
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            img = self.transform_te(img)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            features = self._extract_features(img)
            features = features.data.cpu() #tensor shape=1x2048
            f.append(features)
        f = torch.cat(f, 0)
        return f

    def compute_distance(self, qf, gf):
        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)
        # print(distmat.shape)
        return distmat.numpy()

if __name__ == '__main__':
    reid = REID()



