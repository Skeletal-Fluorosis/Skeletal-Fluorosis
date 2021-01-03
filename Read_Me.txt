1.rotate_crop.py
The original image is processed preliminarily. The image is rotated horizontally and the key parts are retained.

2.Stage_1
This stage realizes the segmentation of lesion region in X-ray images.
1)configuration.txt
Some parameters in the process of training.
2)datasets.py
Encapsulating the dataset.
3)help.py
Some auxiliary functions.
4)pre_processing.py
Some preprocessing of contrast enhancement.
5)train
Train and save the best model.
6)test
Generate results through the best model.

3.Stage_2.py
This stage realizes classification. The inputs are the original images and the corresponding segmentation results. Train and test processes are written in the same program.
