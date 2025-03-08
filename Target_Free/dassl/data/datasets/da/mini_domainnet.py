import os.path as osp
import random
from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class miniDomainNet(DatasetBase):
    """A subset of DomainNet.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
        - Zhou et al. Domain Adaptive Ensemble Learning.
    """

    dataset_dir = "domainnet"
    domains = ["clipart", "painting", "real", "sketch"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_dir = osp.join(self.dataset_dir, "splits_mini")

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="train", num=10000)
        test_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="test", num=630)
        train_u = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="train", num=10000)
        test_u = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="test", num=630)
        if cfg.DATALOADER.TEST.MODE == "free":
            test_1 ,test_2, test_3, test_4 = self._read_data_free(self.domains, num=630)
        else:
            test_1 ,test_2, test_3, test_4 = None
        super().__init__(train_x=train_x, train_u=train_u, test_x=test_x, test_u=test_u, test_1=test_1, test_2=test_2, test_3=test_3, test_4=test_4)

    def _read_data(self, input_domains, split="train", num=0):
        items = []

        for domain, dname in enumerate(input_domains):
            filename = dname + "_" + split + ".txt"
            split_file = osp.join(self.split_dir, filename)

            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    impath, label = line.split(" ")
                    classname = impath.split("/")[1]
                    impath = osp.join(self.dataset_dir, impath)
                    label = int(label)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=classname
                    )
                    items.append(item)
        random.shuffle(items)
        items = items[:num]
        return items

    def _read_data_free(self, input_domains, split="test", num=0):
        items = [[], [], [], []]
        for domain, dname in enumerate(input_domains):
            filename = dname + "_" + split + ".txt"
            split_file = osp.join(self.split_dir, filename)

            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    impath, label = line.split(" ")
                    classname = impath.split("/")[1]
                    impath = osp.join(self.dataset_dir, impath)
                    label = int(label)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=classname
                    )
                    items[domain].append(item)
        random.shuffle(items[0])
        items_test1 = items[0][:num]
        random.shuffle(items[1])
        items_test2 = items[1][:num]
        random.shuffle(items[2])
        items_test3 = items[2][:num]
        random.shuffle(items[3])
        items_test4 = items[3][:num]
        return items_test1, items_test2, items_test3, items_test4