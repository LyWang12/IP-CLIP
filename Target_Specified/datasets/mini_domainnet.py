import os.path as osp

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

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="train")
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS, split="train")
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, split="test")

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data(self, input_domains, split="train"):
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
        print('**************************************************************')

        print(len(items))
        return items
