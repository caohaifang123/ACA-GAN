import glob
import os
from collections import defaultdict
from time import time
from data.multi_modal_dataset import MultiModalDataset
from PIL import Image


class MultipieDataset(MultiModalDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(MultipieDataset, self).__init__(opt)


    def get_modal_names(self):
        return ['l90', 'l45', 'front', 'r45', 'r90']

    def load_img_paths(self):
        dataroot = self.opt.dataroot if self.isTrain else self.opt.test_dataroot

        # {modal_name1 : [/path/to/img1, /path/to/img2]}
        datapath_dict = {}

        for modal_name in self.get_modal_names():
            datapath_dict[modal_name] = glob.glob(os.path.join(dataroot, modal_name, "*.png"))
            datapath_dict[modal_name].sort()

        return datapath_dict


    def load_data(self):
        datapath_dict = self.load_img_paths()
        self.n_data = len(datapath_dict[self.modal_names[0]])
        print('Loading {} Dataset with "{}" mode...'.format(self.dataset, self.mode))
        start = time()
        print('load data from raw')
        data_dict = defaultdict(list)
        for index in range(self.n_data):
            for modal in self.modal_names:
                img = Image.open(datapath_dict[modal][index]).convert('RGB')
                data_dict[modal].append(img)
        end = time()
        print('Finish Loading, cost {:.1f}s'.format(end - start))
        return data_dict
