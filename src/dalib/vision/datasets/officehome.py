import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class OfficeHome(ImageList):

    CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']

    def __init__(self, root, task, filter_class, split='all', **kwargs):
        if split == 'all':
            self.image_list = {
                "A": "image_list/Art.txt",
                "C": "image_list/Clipart.txt",
                "P": "image_list/Product.txt",
                "R": "image_list/Real_World.txt",
            }
        elif split == 'train':
            self.image_list = {
                "A": "image_list/Art_train.txt",
                "C": "image_list/Clipart_train.txt",
                "P": "image_list/Product_train.txt",
                "R": "image_list/Real_World_train.txt",
            }
        elif split == 'val':
            self.image_list = {
                "A": "image_list/Art_val.txt",
                "C": "image_list/Clipart_val.txt",
                "P": "image_list/Product_val.txt",
                "R": "image_list/Real_World_val.txt",
            }

        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        super(OfficeHome, self).__init__(root, num_classes=len(filter_class), data_list_file=data_list_file,
                                       filter_class=filter_class, **kwargs)

