"""Pascal ADE20K Semantic Segmentation Dataset."""
import os
import torch
import os.path as osp
import numpy as np
from PIL import Image
import scipy.io as sio
from torch.utils.data import Dataset
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class ADE20KSegmentation(Dataset):
    """ADE20K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to ADE20K folder. Default is './datasets/ade'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = ADE20KSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """

    def __init__(self, 
                 data_root=None, 
                 ann_file=None,
                 pipeline=None,
                 img_prefix=None,
                 seg_prefix=None,
                 test_mode=False,):

        self.data_root = data_root
        self.ann_file = ann_file,
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.test_mode = test_mode
        self.CLASSES = self.get_class_names()
        self.num_classes = len(self.CLASSES)
        self.COLORS = self.get_class_colors()
        self.lable2color = {i:cat_id for i, cat_id in enumerate(self.COLORS)}
    
        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)

        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)
 
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)


    def load_annotations(self, imglist_file):
        data_infos = []
        img_ids = self.read_imglist(imglist_file)
        for img_id in img_ids:
            filename = f'{img_id}.jpg'
            img_path = osp.join(self.img_prefix,'{}.jpg'.format(img_id))
            img = Image.open(img_path)
            width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, img_path=img_path, width=width, height=height))

        return data_infos


    def read_imglist(self, imglist):
        filelist = []
        with open(imglist[0], 'r') as fd:
            for line in fd:
                filelist.append(line.strip())
        return filelist

    def get_ann_info(self, idx):
        img_info = self.data_infos[idx]
        img_id = img_info['id']
        seg_map = f'{img_id}.png'
        seg_path = osp.join(self.seg_prefix, '{}.png'.format(img_id))
        ann = dict(seg_map=seg_map, seg_path=seg_path)
        return ann


    def _get_ade20k_pairs(self, folder):
        img_paths = []
        mask_paths = []
        if not self.test_mode:
            img_folder = os.path.join(folder, 'images/training')
            mask_folder = os.path.join(folder, 'annotations/training')
        else:
            img_folder = os.path.join(folder, 'images/validation')
            mask_folder = os.path.join(folder, 'annotations/validation')
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)

        return img_paths, mask_paths

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)


    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_path=img_info['img_path'], 
                       label_path=ann_info['seg_path'],
                       img_id=img_info['id'],
                       h = img_info['height'],
                       w = img_info['width'],
                       num_classes =  self.num_classes
                       )
        self.pre_pipeline(results)
        return self.pipeline(results)


    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_path=img_info['img_path'], 
                       label_path=None,
                       img_id=img_info['img_id'],
                       h = img_info['height'],
                       w = img_info['width'],
                       num_classes =  self.num_classes
                       )
        self.pre_pipeline(results)
        return self.pipeline(results)


    def get_class_colors(self):
        color_list = sio.loadmat(osp.join(self.data_root, 'color150.mat'))
        color_list = color_list['colors']
        color_list = color_list[:, ::-1, ]
        color_list = np.array(color_list).astype(int).tolist()
        color_list.insert(0, [0, 0, 0])
        return color_list


    def get_class_names(self):
        return ['wall', 'building, edifice', 'sky',
                'floor, flooring', 'tree', 'ceiling', 'road, route',
                'bed ', 'windowpane, window ',
                'grass', 'cabinet', 'sidewalk, pavement',
                'person, individual, someone, somebody, mortal, soul',
                'earth, ground', 'door, double door', 'table',
                'mountain, mount', 'plant, flora, plant life',
                'curtain, drape, drapery, mantle, pall', 'chair',
                'car, auto, automobile, machine, motorcar', 'water',
                'painting, picture', 'sofa, couch, lounge', 'shelf',
                'house',
                'sea', 'mirror', 'rug, carpet, carpeting', 'field',
                'armchair', 'seat', 'fence, fencing', 'desk',
                'rock, stone',
                'wardrobe, closet, press', 'lamp',
                'bathtub, bathing tub, bath, tub', 'railing, rail',
                'cushion', 'base, pedestal, stand', 'box',
                'column, pillar',
                'signboard, sign',
                'chest of drawers, chest, bureau, dresser',
                'counter', 'sand', 'sink', 'skyscraper',
                'fireplace, hearth, open fireplace',
                'refrigerator, icebox', 'grandstand, covered stand',
                'path',
                'stairs, steps', 'runway',
                'case, display case, showcase, vitrine',
                'pool table, billiard table, snooker table',
                'pillow',
                'screen door, screen', 'stairway, staircase',
                'river',
                'bridge, span', 'bookcase', 'blind, screen',
                'coffee table, cocktail table',
                'toilet, can, commode, crapper, pot, potty, stool, throne',
                'flower', 'book', 'hill', 'bench', 'countertop',
                'stove, kitchen stove, range, kitchen range, cooking stove',
                'palm, palm tree', 'kitchen island',
                'computer, computing machine, computing device, data processor, electronic computer, information processing system',
                'swivel chair', 'boat', 'bar', 'arcade machine',
                'hovel, hut, hutch, shack, shanty',
                'bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle',
                'towel', 'light, light source', 'truck, motortruck',
                'tower',
                'chandelier, pendant, pendent',
                'awning, sunshade, sunblind',
                'streetlight, street lamp',
                'booth, cubicle, stall, kiosk',
                'television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box',
                'airplane, aeroplane, plane', 'dirt track',
                'apparel, wearing apparel, dress, clothes', 'pole',
                'land, ground, soil',
                'bannister, banister, balustrade, balusters, handrail',
                'escalator, moving staircase, moving stairway',
                'ottoman, pouf, pouffe, puff, hassock',
                'bottle', 'buffet, counter, sideboard',
                'poster, posting, placard, notice, bill, card',
                'stage', 'van', 'ship', 'fountain',
                'conveyer belt, conveyor belt, conveyer, conveyor, transporter',
                'canopy',
                'washer, automatic washer, washing machine',
                'plaything, toy',
                'swimming pool, swimming bath, natatorium',
                'stool', 'barrel, cask', 'basket, handbasket',
                'waterfall, falls', 'tent, collapsible shelter',
                'bag',
                'minibike, motorbike', 'cradle', 'oven', 'ball',
                'food, solid food', 'step, stair',
                'tank, storage tank',
                'trade name, brand name, brand, marque',
                'microwave, microwave oven', 'pot, flowerpot',
                'animal, animate being, beast, brute, creature, fauna',
                'bicycle, bike, wheel, cycle ', 'lake',
                'dishwasher, dish washer, dishwashing machine',
                'screen, silver screen, projection screen',
                'blanket, cover', 'sculpture', 'hood, exhaust hood',
                'sconce',
                'vase', 'traffic light, traffic signal, stoplight',
                'tray',
                'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',
                'fan', 'pier, wharf, wharfage, dock', 'crt screen',
                'plate', 'monitor, monitoring device',
                'bulletin board, notice board', 'shower',
                'radiator',
                'glass, drinking glass', 'clock', 'flag']


