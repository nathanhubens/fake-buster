import sys
sys.path.insert(0,'../scripts')
sys.path.insert(0,'..')

from fastai.vision.all import *
from face_detection.video_utils import (nms, read_all_frames, plot_detections, get_video_stats, read_random_frames, load_all_metadata)
from face_detection.EasyBlazeFace import EasyBlazeFace
from face_detection.EasyRetinaFace import EasyRetinaFace
#from metrics import *
import albumentations as A
from tqdm import tqdm
import time
import cv2
import imageio
from sklearn.metrics import roc_auc_score



from utils import *


class AlbumentationsTransform(DisplayedTransform):
    split_idx,order=0,2
    def __init__(self, train_aug): store_attr()
    
    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)

def get_train_aug(): return A.Compose([
            Downscale(scale_min=0.25, scale_max=0.5,  p=0.3),
            A.MotionBlur(blur_limit=9, p=0.3),
            A.GaussNoise(var_limit=(50.0, 100.0), p=0.3),
            A.JpegCompression(quality_lower=25, quality_upper=85, p=0.3),
            A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.5)
])


def get_dls(size, bs, device):
    item_tfms = [Resize(size), AlbumentationsTransform(get_train_aug())]

    faces = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                    get_items=get_image_files, 
                    splitter=ParentSplitter(),
                    get_y=Pipeline([attrgetter("name"), RegexLabeller(pat = '([A-Z]+).jpg$')]),
                    item_tfms=item_tfms)

    return faces.dataloaders(Path('../data/images'), bs=bs, device=device)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(2)
#detector = EasyBlazeFace(weights='../face_detection/blazeface.pth', anchors='../face_detection/anchors.npy', device=device)
detector = EasyRetinaFace(path= '../face_detection/Pytorch_Retinaface/weights/Resnet50_Final.pth', device=device)


name = 'dw_xrn18_sa_se_mixup_prog_prune_fp16_KD_1'



learn = load_learner(f'../scripts/{name}.pkl')

def read_frames_sample(vid, n_frames=16):
    vid = imageio.get_reader(io.BytesIO(vid),  'ffmpeg')
    sample = np.linspace(0, int(vid.count_frames()) - 1 , n_frames).astype(int)
    list_frames = list(vid.iter_data())
    return [list_frames[s] for s in sample]

def read_frames_sample(video_path, n_frames=16):
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample = np.linspace(0, frame_count - 1 , n_frames).astype(int)

    frames = []
    for i in range(frame_count):
        _ = capture.grab()
        if i in sample:
            success, frame = capture.retrieve()
            if not success: continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    capture.release()
    return np.array(frames)

def predict_all_frames(frames):
    
    detections = []
    for frame in frames:
        detection = detector.detect(frame)
        detections.append(detection)

    return detections

def get_cropped_faces(frame, detections):
    
    faces = []
    
    for detection in detections:
        xmin = max(0, int(detection[0])) # Don't try to crop less than 0
        ymin = max(0, int(detection[1])) # Don't try to crop less than 0
        xmax = min(frame.shape[1], int(detection[2]))
        ymax = min(frame.shape[0], int(detection[3]))
        
        face = frame[ymin:ymax, xmin:xmax]
        
        faces.append(face)
    
    return faces

def get_preds(faces):
    return np.mean([learn.predict(f)[2][0] for f in faces]) # returns the "fakeness" of the extracted face

path = Path('../data/videos/test')

def get_faces(frames, frames_detections):
    
    faces = []
    _, h,w, _ = frames.shape
    
    for ix, fr in enumerate(frames_detections):
        face = get_cropped_faces(frames[ix], fr)
        for f in face:
            hf, wf, _ = f.shape
            if hf*wf>400 and hf*wf<0.1*h*w:
                faces.append(f)
        
    return faces


targs = pd.read_csv(path/'labels.csv')
targs['preds'] = 0.5
videos = targs.filename


for vid in tqdm(videos):
    try:
        frames = read_frames_sample(path/vid)
        frames_detections = predict_all_frames(frames)
        faces = get_faces(frames, frames_detections)
        pred = get_preds(faces)
        if math.isnan(pred)==False:
            targs.loc[targs['filename'] == vid, 'preds'] = np.clip(pred, 0.001, 0.999)
        else: targs.loc[targs['filename'] == vid, 'preds'] = 0.5
    except Exception as e:
        print(vid)
        targs.loc[targs['filename'] == vid, 'preds'] = 0.5


def acc(inp, targ):
    pred = torch.tensor(inp>0.5, dtype=torch.long)
    return (pred == targ).float().mean()

print(f'ACC: {acc(torch.tensor(targs.preds.values), torch.tensor(targs.label.values, dtype=torch.long))}')

m = nn.LogSoftmax(dim=1)

loss = torch.nn.NLLLoss()

spreds = torch.tensor([1-targs.preds.values, targs.preds.values], dtype=torch.float32).permute((1,0))

slabs = torch.tensor(targs.label.values, dtype=torch.long)

print(f'LOSS: {loss(m(spreds), slabs)}')

print(f'ROC: {roc_auc_score(slabs, spreds[:,1])}')

print(name)

#targs.to_csv('../inference_small.csv')