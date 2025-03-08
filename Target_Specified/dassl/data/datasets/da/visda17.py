import os.path as osp
import random
from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class VisDA17(DatasetBase):
    """VisDA17.

    Focusing on simulation-to-reality domain shift.

    URL: http://ai.bu.edu/visda-2017/.

    Reference:
        - Peng et al. VisDA: The Visual Domain Adaptation
        Challenge. ArXiv 2017.
    """

    dataset_dir = "visda17"
    domains = ["synthetic", "real"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x, test_x = self._read_data("synthetic", train_num=5000, test_num=1000)
        train_u, test_u = self._read_data("real", train_num=5000, test_num=1000)

        super().__init__(train_x=train_x, train_u=train_u, test_x=test_x, test_u=test_u)

    def _read_data(self, dname, train_num=0, test_num=0):
        filedir = "train" if dname == "synthetic" else "validation"
        image_list = osp.join(self.dataset_dir, filedir, "image_list.txt")
        items = []
        # There is only one source domain
        domain = 0

        with open(image_list, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                classname = impath.split("/")[0]
                impath = osp.join(self.dataset_dir, filedir, impath)
                label = int(label)
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=classname
                )
                items.append(item)
        random.shuffle(items)
        items_train = items[:train_num]
        items_test = items[train_num:train_num+test_num]

        return items_train, items_test
