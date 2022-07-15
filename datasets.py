import os
import json

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from PIL import Image
import torchvision

# from .augmentation.randaugment import RandAugment
import randaugment
from randaugment import RandAugment

try:
    import timm
    from timm.data.dataset import ImageFolder as timmImageFolder
    tif = True
except:
    import timm
    from timm.data.dataset import Dataset as timmImageFolder
    tif = False

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder
    
class CustomDataset(timmImageFolder):
    
    def __init__(
            self,
            root,
            parser=None,
            class_map='',
            load_bytes=False,
            transform=None,
            select_ratio=1.0, 
            data_split_file='None',
            exclude_labeldata=False,
            label_p_list=[''],
    ):
        self.root = root
        self.select_ratio = select_ratio
        self.data_split_file = data_split_file
        self.select_ratio = select_ratio 
        self.transform = transform 

        if tif:
            super(CustomDataset, self).__init__(root,parser,class_map,load_bytes,transform)
            self.cls_map = {}
            for line in self.parser:
                self.cls_map[line[0].split('/')[-1].split('_')[0]]=line[1]
            
            if select_ratio < 1.0:
                print('select semi data globally')
                ori_len = len(self.parser)
                new_len = int(ori_len * select_ratio)
                import random
                random.seed(2021)
                sample_idx = sorted(random.sample(range(0, ori_len), new_len))
                new_parser = [self.parser[idx] for idx in sample_idx]
                self.parser = new_parser
                print('original img number: %d'%ori_len)
                print('new img number: %d'%new_len)
            
            elif data_split_file != 'None':
                print('select semi data by data_split_file')
                ori_len = len(self.parser)
                self.parser = self.filter_file(self.root, self.data_split_file)
                new_len = len(self.parser)
                print('original img number: %d'%ori_len)
                print('new img number: %d'%new_len)

            elif exclude_labeldata:
                print('unlabeled data excluding labeled data')
                ori_len = len(self.parser)
                self.parser = list(set(self.parser) - set(label_p_list))
                new_len = len(self.parser)
                print('original img number: %d'%ori_len)
                print('new img number: %d'%new_len)

        else:
            super(CustomDataset, self).__init__(root,load_bytes,transform)
            self.cls_map = {}
            for line in self.imgs:
                self.cls_map[line[0].split('/')[-1].split('_')[0]]=line[1]
            
            if select_ratio < 1.0:
                print('select semi data globally')
                ori_len = len(self.imgs)
                new_len = int(ori_len * select_ratio)
                import random
                random.seed(2021)
                sample_idx = sorted(random.sample(range(0, ori_len), new_len))
                new_imgs = [self.imgs[idx] for idx in sample_idx]
                self.imgs = new_imgs
                print('original img number: %d'%ori_len)
                print('new img number: %d'%new_len)
            
            elif data_split_file != 'None':
                print('select semi data by data_split_file')
                ori_len = len(self.imgs)
                self.imgs = self.filter_file(self.root, self.data_split_file)
                new_len = len(self.imgs)
                print('original img number: %d'%ori_len)
                print('new img number: %d'%new_len)
                      
            elif exclude_labeldata:
                print('unlabeled data excluding labeled data')
                ori_len = len(self.imgs)
                self.imgs = list(set(self.imgs) - set(label_p_list))
                new_len = len(self.imgs)
                print('original img number: %d'%ori_len)
                print('new img number: %d'%new_len)

                                
    def __getitem__(self, index):
        if tif:
            img, target = self.parser[index]
            try:
                img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
            except Exception as e:
                _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
                self._consecutive_errors += 1
                if self._consecutive_errors < _ERROR_RETRY:
                    return self.__getitem__((index + 1) % len(self.parser))
                else:
                    raise e
            self._consecutive_errors = 0
            
            if target is None:
                target = torch.tensor(-1, dtype=torch.long)
            
            if self.transform is not None:
                if isinstance(self.transform, dict):
                    strong_img = self.transform['strong_transform'](img)
                    weak_img = self.transform['weak_transform'](img)
                    return strong_img, weak_img, target
                else:
                    img = self.transform(img)
                    return img, target
            
            # return img, target
        
        else:
            path, target = self.imgs[index]
            img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
            if target is None:
                target = torch.zeros(1).long()
             
            if self.transform is not None:
                if isinstance(self.transform, dict):
                    strong_img = self.transform['strong_transform'](img)
                    weak_img = self.transform['weak_transform'](img)
                    return strong_img, weak_img, target
                else:
                    img = self.transform(img)
                    return img, target 
        
        # TODO: tranform_list / tranform. 

    def filter_file(self, root, data_split_file):        
        semi_datalist = [i.strip().split(',')[0] for i in open(data_split_file, 'r').readlines()]
        return [(os.path.join(root, i.split('_')[0], i), self.cls_map[i.split('_')[0]]) for i in semi_datalist]


    def __len__(self):
        if tif:
            return len(self.parser)
        else:
            return len(self.imgs)


class CustomPlacesDataset(Dataset):
    
    def __init__(
            self,root,list_file,
            transform=None,
    ):
        self.root = root
        self.list_file = list_file
        self.transform = transform 
        self.input_list = []
        
        
        line_list = open(list_file, 'r').readlines()
        for line in line_list:
            img_path, target = line.split(',')
            target = int(target)
            img_path = os.path.join(root, img_path)
            self.input_list.append([img_path, target])

                                            
    def __getitem__(self, index):
            path, target = self.input_list[index]
            img = Image.open(path).convert('RGB')
            if target is None:
                target = torch.zeros(1).long()
             
            if self.transform is not None:
                if isinstance(self.transform, dict):
                    strong_img = self.transform['strong_transform'](img)
                    weak_img = self.transform['weak_transform'](img)
                    return strong_img, weak_img, target
                else:
                    img = self.transform(img)
                    return img, target 
        

    def __len__(self):
        return len(self.input_list)


class CustomCifar10(Dataset):
    
    def __init__(
            self,root,
            is_train=True,
            transform=None,
            select_ratio=1.,
    ):
        import numpy as np

        self.nb_classes = 10
        self.root = root
        self.transform = transform 
        self.select_ratio = select_ratio
        self.input_list = []
        self.dset = torchvision.datasets.CIFAR10(root, is_train, download=True) 
        self.data, self.targets = np.array(self.dset.data), np.array(self.dset.targets)
        
        self.cifar_bin = {}
        for i in range(self.nb_classes):
            self.cifar_bin[i] = []
        for i in range(len(self.data)):
            self.cifar_bin[self.targets[i]].append(self.data[i])
        if is_train:
            assert len(self.cifar_bin[0]) == 5000
        
        partial_num = int(self.select_ratio * len(self.cifar_bin[0]))
        print(partial_num)
         
        for i in self.cifar_bin:
            for j in range(partial_num):
                self.input_list.append([self.cifar_bin[i][j], i])

    
    def __getitem__(self, index):
            img, target = self.input_list[index]
            img = Image.fromarray(img)

            if target is None:
                target = torch.zeros(1).long()
             
            if self.transform is not None:
                if isinstance(self.transform, dict):
                    strong_img = self.transform['strong_transform'](img)
                    weak_img = self.transform['weak_transform'](img)
                    return strong_img, weak_img, target
                else:
                    img = self.transform(img)
                    return img, target 
            
            return img, target

    def __len__(self):
        return len(self.input_list)


class CustomCifar100(Dataset):
    
    def __init__(
            self,root,
            is_train=True,
            transform=None,
            select_ratio=1.,
    ):
        import numpy as np

        self.nb_classes = 100
        self.root = root
        self.transform = transform 
        self.select_ratio = select_ratio
        self.input_list = []
        self.dset = torchvision.datasets.CIFAR100(root, is_train, download=True) 
        self.data, self.targets = np.array(self.dset.data), np.array(self.dset.targets)
        
        self.cifar_bin = {}
        for i in range(self.nb_classes):
            self.cifar_bin[i] = []
        for i in range(len(self.data)):
            self.cifar_bin[self.targets[i]].append(self.data[i])
        if is_train:
            assert len(self.cifar_bin[0]) == 500
        
        partial_num = int(self.select_ratio * len(self.cifar_bin[0]))
        print(partial_num)
         
        for i in self.cifar_bin:
            for j in range(partial_num):
                self.input_list.append([self.cifar_bin[i][j], i])

    
    def __getitem__(self, index):
        img, target = self.input_list[index]
        img = Image.fromarray(img)

        if target is None:
            target = torch.zeros(1).long()
         
        if self.transform is not None:
            if isinstance(self.transform, dict):
                strong_img = self.transform['strong_transform'](img)
                weak_img = self.transform['weak_transform'](img)
                return strong_img, weak_img, target
            else:
                img = self.transform(img)
                return img, target 
        
        return img, target

    def __len__(self):
        return len(self.input_list)






class CustomINATDataset(Dataset):
    
    def __init__(
            self,root,list_file,
            transform=None,
    ):
        self.root = root
        self.list_file = list_file
        self.transform = transform 
        self.input_list = []
        
        
        line_list = open(list_file, 'r').readlines()
        for line in line_list:
            img_path, target = line.split(',')
            target = int(target)
            img_path = os.path.join(root, img_path)
            self.input_list.append([img_path, target])

                                            
    def __getitem__(self, index):
            path, target = self.input_list[index]
            img = Image.open(path).convert('RGB')
            if target is None:
                target = torch.zeros(1).long()
             
            if self.transform is not None:
                if isinstance(self.transform, dict):
                    strong_img = self.transform['strong_transform'](img)
                    weak_img = self.transform['weak_transform'](img)
                    return strong_img, weak_img, target
                else:
                    img = self.transform(img)
                    return img, target 
        

    def __len__(self):
        return len(self.input_list)



def build_dataset(is_train, args, is_val=False, is_test=False):
    transform = build_transform(is_train, args)
    # need 3 kinds of tranform type.:w
    # TODO.
    # if is_train: transform1, transform2, transform3
    if args.data_set == 'CIFAR100':
        # dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        # nb_classes = 100
        # CustomCifar100
        root = args.data_path
        nb_classes = 100
        cifar100_mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        cifar100_std = [x / 255 for x in [68.2,  65.4,  70.4]]
        
        if is_train:
            if args.supervise_only:
                size = int((256 / 224) * args.input_size)
                transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                    transforms.Resize(size, interpolation=3),
                    transforms.RandomCrop(args.input_size, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std)])

                dataset = CustomCifar100(root, is_train, 
                        transform=transform, select_ratio=args.select_ratio)
                return dataset, nb_classes

            else:
                size = int((256 / 224) * args.input_size)
                norm_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                    transforms.Resize(size, interpolation=3),
                    transforms.RandomCrop(args.input_size, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std)])
 
                weak_transform = norm_transform
                strong_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                    RandAugment(3,5),
                    transforms.Resize(size, interpolation=3),
                    transforms.RandomCrop(args.input_size, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std)])

                # norm_transform, strong_transform, weak_transform = build_transform(is_train, args, semi=True)
                # norm_transform = weak_transform
                dataset_x = CustomCifar100(root, is_train, 
                        transform=norm_transform, select_ratio=args.select_ratio)
                
                dataset_u = CustomCifar100(root, is_train, 
                        transform={'strong_transform': strong_transform, 
                            'weak_transform': weak_transform},)

            return dataset_x, dataset_u, nb_classes
        else:
            transform = transforms.Compose([
                    transforms.Resize(args.input_size, interpolation=3),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std)])


            dataset = CustomCifar100(root, is_train, transform=transform, select_ratio=1.)
            return dataset, nb_classes

    elif args.data_set == 'CIFAR10':
        # CustomCifar10
        root = args.data_path
        nb_classes = 10
        cifar10_mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        cifar10_std = [x / 255 for x in [63.0, 62.1, 66.7]]
        
        if is_train:
            if args.supervise_only:
                size = int((256 / 224) * args.input_size)
                transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                    transforms.Resize(size, interpolation=3),
                    transforms.RandomCrop(args.input_size, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std)])

                dataset = CustomCifar10(root, is_train, 
                        transform=transform, select_ratio=args.select_ratio)
                return dataset, nb_classes

            else:
                size = int((256 / 224) * args.input_size)
                norm_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                    transforms.Resize(size, interpolation=3),
                    transforms.RandomCrop(args.input_size, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std)])
 
                weak_transform = norm_transform
                strong_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                    RandAugment(3,5),
                    transforms.Resize(size, interpolation=3),
                    transforms.RandomCrop(args.input_size, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std)])

                # norm_transform, strong_transform, weak_transform = build_transform(is_train, args, semi=True)
                # norm_transform = weak_transform
                dataset_x = CustomCifar10(root, is_train, 
                        transform=norm_transform, select_ratio=args.select_ratio)
                
                dataset_u = CustomCifar10(root, is_train, 
                        transform={'strong_transform': strong_transform, 
                            'weak_transform': weak_transform},)

            return dataset_x, dataset_u, nb_classes
        else:
            transform = transforms.Compose([
                    transforms.Resize(args.input_size, interpolation=3),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std)])


            dataset = CustomCifar10(root, is_train, transform=transform, select_ratio=1.)
            return dataset, nb_classes
        # dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
    elif args.data_set == 'SEMI-IMNET':
        nb_classes = 1000
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if is_train:
            if args.supervise_only:
                dataset = CustomDataset(root, transform=transform, select_ratio=args.select_ratio,
                                       data_split_file=args.data_split_file)
                return dataset, nb_classes

            else:
                norm_transform, strong_transform, weak_transform = build_transform(is_train, args, semi=True)
                # args.unlabelDataset_exclude_labeledData
                dataset_x = CustomDataset(root, transform=norm_transform, select_ratio=args.select_ratio, 
                                        data_split_file=args.data_split_file)
                label_p_list = dataset_x.parser if tif else dataset_x.imgs 
                dataset_u = CustomDataset(root, transform={'strong_transform': strong_transform, 
                                                           'weak_transform': weak_transform},
                                          exclude_labeldata=args.unlabelDataset_exclude_labeledData,
                                          label_p_list=label_p_list,
                                                           ) 
            return dataset_x, dataset_u, nb_classes
        else:
            dataset = CustomDataset(root, transform=transform,)

    elif args.data_set == 'SEMI-INAT':
        nb_classes = 1010
        root = args.data_path
        # 'l_train', 'u_train', 'val', 'test'
        if is_train:
            norm_transform, strong_transform, weak_transform = build_transform(is_train, args, semi=True)

            l_train_root = os.path.join(root, 'data')
            l_train_list = os.path.join(root, 'semi_list/semi-train.txt')

            u_train_root = os.path.join(root, 'data')
            u_train_list = os.path.join(root, 'semi_list/semi-unlabel.txt')
            

            dataset_x = CustomINATDataset(l_train_root, l_train_list, transform=norm_transform)
            dataset_u = CustomINATDataset(u_train_root, u_train_list, 
                    transform={'strong_transform': strong_transform,
                               'weak_transform': weak_transform},
                )

            return dataset_x, dataset_u, nb_classes
        # elif is_test:
        #     transform = build_transform(is_train, args)
        #     test_root = os.path.join(root, 'test/test')
        #     dataset = CustomINATDataset(test_root, 'unlabeled', transform=transform)
        else:
            transform = build_transform(is_train, args)
            val_root = os.path.join(root, 'data')
            val_list = os.path.join(root, 'semi_list/semi-val.txt')
            dataset = CustomINATDataset(val_root, val_list, transform=transform)
        
        return dataset, nb_classes
     
    elif args.data_set == 'SEMI-PLACES205' or args.data_set == '5PER-SEMI-PLACES205' or args.data_set == '1PER-SEMI-PLACES205':
        nb_classes = 205
        root = args.data_path
        split_folder = args.split_folder
        # 'l_train', 'u_train', 'val', 'test'
        if is_train:
            norm_transform, strong_transform, weak_transform = build_transform(is_train, args, semi=True)

            l_train_root = u_train_root = root
            
            if args.data_set == 'SEMI-PLACES205':
                l_train_list = os.path.join(args.split_folder, 'semi_ltrain.txt')
                u_train_list = os.path.join(args.split_folder, 'semi_utrain.txt')
            elif args.data_set == '5PER-SEMI-PLACES205':
                l_train_list = os.path.join(args.split_folder, '5per_semi_ltrain.txt')
                u_train_list = os.path.join(args.split_folder, '5per_semi_utrain.txt')
            
            elif args.data_set == '1PER-SEMI-PLACES205':
                l_train_list = os.path.join(args.split_folder, '1per_semi_ltrain.txt')
                u_train_list = os.path.join(args.split_folder, '1per_semi_utrain.txt')
            
            dataset_x = CustomPlacesDataset(l_train_root, l_train_list, transform=norm_transform)
            dataset_u = CustomPlacesDataset(u_train_root, u_train_list, 
                    transform={'strong_transform': strong_transform,
                               'weak_transform': weak_transform},
                )

            return dataset_x, dataset_u, nb_classes
        # elif is_test:
        #     transform = build_transform(is_train, args)
        #     test_root = os.path.join(root, 'test/test')
        #     dataset = CustomINATDataset(test_root, 'unlabeled', transform=transform)
        
        else:
            transform = build_transform(is_train, args)
            val_root = root
            val_list = os.path.join(args.split_folder, 'semi_val.txt')
            dataset = CustomPlacesDataset(val_root, val_list, transform=transform)
        
        return dataset, nb_classes
     


    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    
    return dataset, nb_classes


def build_transform(is_train, args, semi=False):
    resize_im = args.input_size > 32
    if is_train:
        if semi is True:
            # TODO: design norm & strong transform
            norm_transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
            strong_transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
            weak_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(int((256 / 224) * args.input_size), interpolation=3),
                transforms.RandomCrop(args.input_size, padding=4,padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            ])
            return norm_transform, strong_transform, weak_transform

        else:
        # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )

            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size, padding=4)
            return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
