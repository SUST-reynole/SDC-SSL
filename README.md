# Shape-guided Dual Consistency Semi-supervised Learning Framework for 3D Medical Image Segmentation

## Usage
1. Clone the repository;
```
git clone https://github.com/SUST-reynole/SDC-SSL.git
```

2. Put the data in './data';

3. Train the model;
```
cd code
# e.g., for 20% labels on LA
python ./code/train_LA.py --root_path ../data/2018LA_Seg_Training_Set/ --max_iterations 6000 --labelnum 16
```

4. Test the model;
```
cd code
# e.g., for 20% labels on LA
python ./code/test_LA.py --root_path ../data/2018LA_Seg_Training_Set/ --gpu 0
```

## Acknowledgements:
Our code is origin from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [DTML](https://github.com/YichiZhang98/DTML), [URPC](https://github.com/HiLab-git/SSL4MIS),  [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [MC-Net](https://github.com/ycwu1997/MC-Net). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

## Questions
