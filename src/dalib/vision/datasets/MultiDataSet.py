import os
from .imagelist import ImageList

class MultiDataSet(ImageList):
    def __init__(self, root, task, filter_class, split='all', **kwargs):
        if split == 'all':
            self.image_list = {
                "A": "image_list/amazon.txt",
                "W": "image_list/webcam.txt",
                "D": "image_list/dslr.txt",
                "S": "stl10_imagelist/stl10_all_new.txt",
                "V": "visda_imagelist/visda_all_new.txt",
                "C": "domainnet_imagelist/domainnet_clipart_test_new.txt",
                "Q": "domainnet_imagelist/domainnet_quickdraw_test_new.txt",
                "I": "domainnet_imagelist/domainnet_infograph_test_new.txt",
                "R": "domainnet_imagelist/domainnet_real_test_new.txt",
                "P": "domainnet_imagelist/domainnet_painting_test_new.txt",
                "K": "domainnet_imagelist/domainnet_sketch_test_new.txt",
            }
        elif split == 'train':
            self.image_list = {
                "A": "image_list/amazon_train.txt",
                "W": "image_list/webcam_train.txt",
                "D": "image_list/dslr_train.txt",
                "S": "stl10_imagelist/stl10_train_new.txt",
                "V": "visda_imagelist/visda_train_new.txt",
                "C": "domainnet_imagelist/domainnet_clipart_test_new.txt",
                "Q": "domainnet_imagelist/domainnet_quickdraw_test_new.txt",
                "I": "domainnet_imagelist/domainnet_infograph_test_new.txt",
                "R": "domainnet_imagelist/domainnet_real_test_new.txt",
                "P": "domainnet_imagelist/domainnet_painting_test_new.txt",
                "K": "domainnet_imagelist/domainnet_sketch_test_new.txt",
            }
        elif split == 'val':
            self.image_list = {
                "A": "image_list/amazon_val.txt",
                "W": "image_list/webcam_val.txt",
                "D": "image_list/dslr_val.txt",
                "S": "stl10_imagelist/stl10_val_new.txt",
                "V": "visda_imagelist/visda_val_new.txt",
                "C": "domainnet_imagelist/domainnet_clipart_test_new.txt",
                "Q": "domainnet_imagelist/domainnet_quickdraw_test_new.txt",
                "I": "domainnet_imagelist/domainnet_infograph_test_new.txt",
                "R": "domainnet_imagelist/domainnet_real_test_new.txt",
                "P": "domainnet_imagelist/domainnet_painting_test_new.txt",
                "K": "domainnet_imagelist/domainnet_sketch_test_new.txt",
            }

        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        super(MultiDataSet, self).__init__(root, num_classes=len(filter_class), data_list_file=data_list_file,
                                         filter_class=filter_class, **kwargs)

