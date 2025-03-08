import os.path as osp
import random
from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class OfficeHome(DatasetBase):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    dataset_dir = "office_home"
    domains = ["art", "clipart", "product", "real_world"]


    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )
        train_x, test_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, train_num=2000, test_num=400)
        train_u, test_u = self._read_data(cfg.DATASET.SOURCE_DOMAINS, train_num=2000, test_num=400)
        if cfg.DATALOADER.TEST.MODE == "free" or cfg.DATALOADER.TEST.MODE == "author":
            test_1 ,test_2, test_3, test_4 = self._read_data_free(self.domains, test_num=400)
        else:
            test_1 ,test_2, test_3, test_4 = None
        super().__init__(train_x=train_x, train_u=train_u, test_x=test_x, test_u=test_u, test_1=test_1, test_2=test_2, test_3=test_3, test_4=test_4)


    def _read_data(self, input_domains, train_num=0, test_num=0):
        items = []

        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)    # officehome/clipart
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()

            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_dir, class_name)   # ./clipart/sink
                imnames = listdir_nohidden(class_path)          # [1.jpg, 2.jpg,   ]
                for imname in imnames:
                    impath = osp.join(class_path, imname)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=class_name.lower(),
                    )
                    items.append(item)
        random.shuffle(items)
        items_train = items[:train_num]
        items_test = items[train_num:train_num+test_num]
        return items_train, items_test

    def _read_data_free(self, input_domains, train_num=0, test_num=0):
        items = [[], [], [], []]
        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()

            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_dir, class_name)
                imnames = listdir_nohidden(class_path)
                for imname in imnames:
                    impath = osp.join(class_path, imname)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=class_name.lower(),
                    )
                    items[domain].append(item)
        random.shuffle(items[0])
        items_test1 = items[0][:test_num]
        random.shuffle(items[1])
        items_test2 = items[1][:test_num]
        random.shuffle(items[2])
        items_test3 = items[2][:test_num]
        random.shuffle(items[3])
        items_test4 = items[3][:test_num]
        return items_test1, items_test2, items_test3, items_test4

