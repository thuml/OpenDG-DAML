# OpenDG-DAML
 Code release for [Open Domain Generalization with Domain-Augmented Meta-Learning](https://arxiv.org/abs/2104.03620) (CVPR2021)

## Dataset

| Dataset | Link |
| ------ | ------ |
| PACS | https://dali-dl.github.io/project_iccv2017.html |
| Office-Home | https://www.hemanthdv.org/officeHomeDataset.html |
| Office-31 | http://www.eecs.berkeley.edu/~mfritz/domainadaptation/ |
| STL-10 | https://cs.stanford.edu/~acoates/stl10/ |
| Visda2017 | http://ai.bu.edu/visda-2017/ |
| DomainNet | http://ai.bu.edu/M3SDA/ |


## Requirements

* Python 3.8
* PyTorch 1.5.0

## Quick Start

* Download the DATASET you need. Move the `image_list` folder of the DATASET (which we provide in `data/DATASET/`) to the directory of the DATASET.
* Complete the configuration of experiments, such as the path to the DATASET, then `bash run_train.sh` for training on source domains and testing on target domain data from known classes.
* After training and saving the model checkpoints, `bash run_validate.sh` for testing on the whole target domain, including both known and unknown classes.

## Citation
If you find this code or our paper useful, please consider citing:<br>

```
@inproceedings{shu2021open,
  title={Open Domain Generalization with Domain-Augmented Meta-Learning},
  author={Shu, Yang and Cao, Zhangjie and Wang, Chenyu and Wang, Jianmin and Long, Mingsheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9624--9633},
  year={2021}
}
```

## Contact
If you have any problems about our code, feel free to contact<br>

* shu-y18@mails.tsinghua.edu.cn