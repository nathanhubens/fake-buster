from os import device_encoding
from fastai.vision.all import *
from fastcore.script import *
from utils import *
from mesonet import *
from efficientnet_pytorch_test import *
from custom import *
from fasterai.sparse.all import *
from fasterai.distill.all import *


class AlbumentationsTransform(DisplayedTransform):
    split_idx,order=0,2
    def __init__(self, train_aug): store_attr()
    
    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)

def get_item_tfms(size, blur_limit, var_limit, quality_lower, quality_upper, num_holes, hole_size):  
    alb = A.Compose([
            A.MotionBlur(blur_limit=blur_limit, p=0.3),
            A.GaussNoise(var_limit=var_limit, p=0.3),
            A.JpegCompression(quality_lower=quality_lower, quality_upper=quality_upper, p=0.3),
            A.Cutout(num_holes=num_holes, max_h_size=hole_size, max_w_size=hole_size, p=0.3)
])

    return [Resize(size), AlbumentationsTransform(alb)]


def get_dls(size=256, bs=32, blur_limit=3, var_limit=(25., 50.), quality_lower=50, quality_upper=90, num_holes=3, hole_size=3, device='cuda:0'):

    faces = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                     get_items=get_image_files, 
                     splitter=ParentSplitter(),
                     get_y=Pipeline([attrgetter("name"), RegexLabeller(pat = '([A-Z]+).jpg$')]),
                     item_tfms=get_item_tfms(size, blur_limit, var_limit, quality_lower, quality_upper, num_holes, hole_size),
                     batch_tfms = aug_transforms())

    return faces.dataloaders(Path('../data/images'), bs=bs, device=device)


class ProgressiveLearningCallback(Callback):

    def __init__(self, device):
        store_attr()


    def before_epoch(self):

        if self.epoch == self.n_epoch//3:
            print('second progressive learning')
            self.learn.dls = get_dls(size=256, device=self.device,  blur_limit=7, var_limit=(50., 100.), quality_lower=30, quality_upper=80, num_holes=7, hole_size=7)
        
        if self.epoch == 2*self.n_epoch//3:
            print('third progressive learning')
            self.learn.dls = get_dls(size=256, device=self.device, blur_limit=9, var_limit=(25., 200.), quality_lower=20, quality_upper=70, num_holes=9, hole_size=9)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def sched_agp(start, end, pos): return end + start - end * (1 - pos)**3


def annealing_custom(start, end, pos):
    α, β = 14,5
    return  start + (end-start)* (1+np.exp(-α+β)) / (1 + (np.exp((-α*pos)+β)))


@call_parse
def main(
    cuda:  Param("Use imagewoof (otherwise imagenette)", str)='cuda:2',
    lr:    Param("Learning rate", float)=1e-2,
    size:  Param("Size (px: 128,192,256)", int)=256,
    sqrmom:Param("sqr_mom", float)=0.99,
    mom:   Param("Momentum", float)=0.9,
    eps:   Param("Epsilon", float)=1e-6,
    wd:    Param("Weight decay", float)=1e-2,
    epochs:Param("Number of epochs", int)=20,
    bs:    Param("Batch size", int)=32,
    mixup: Param("Mixup", float)=0.4,
    arch:  Param("Architecture", str)='resnet18',
    beta:  Param("SAdam softplus beta", float)=0.,
    fp16:  Param("Use mixed precision training", store_true)=True,
    runs:  Param("Number of times to repeat training", int)=3,
    prune: Param("Pruning", bool)=True,

):

    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(int(cuda[-1]))
    opt_func = partial(ranger, mom=mom, sqr_mom=sqrmom, eps=eps, beta=beta)

    dls = get_dls(size, bs, device=device)
    print(dls)


    for run in range(runs):

        m = dw_xresnet18()

        name = 'dw_xresnet18'
        name_teacher = 'dw_xrn18_sa_se_mixup_prog.pkl'

        teacher = load_learner(name_teacher, cpu=False)
        teacher = teacher.to_fp16()
        teacher.model.to(device)

        print(count_parameters(m))

        print(f'Run: {run}')
        learn = Learner(dls, m, opt_func=opt_func, metrics=[accuracy], loss_func=LabelSmoothingCrossEntropy())

        learn = learn.to_fp16()
        cbs = MixUp(mixup) if mixup else []

        loss = partial(SoftTarget, T=30)
        kd = KnowledgeDistillation(teacher, loss)

        prog = ProgressiveLearningCallback(device=device)
        sp = SparsifyCallback(50, 'weight', 'local', large_final, annealing_custom) if prune else []
        csv = CSVLogger(fname=name+'_'+str(run)+'.csv')


        learn.fit_one_cycle(epochs, lr, wd=wd, cbs=[cbs, csv, prog, sp, kd])

        learn.export(name+'_'+str(run)+'.pkl')