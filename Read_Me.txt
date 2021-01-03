1.rotate_crop.py
The original image is processed preliminarily. The image is rotated horizontally and the key parts are retained

2.Stage_1
Segmentation of lesion region in X-ray images.
1)configuration.txt
Some parameters in the process of training.
2)datasets.py
Encapsulating the dataset.
3)help.py
Some auxiliary functions.
4)pre_processing.py
Some preprocessing of contrast enhancement.
5)train
6)test

3.Stage_2.py
The program realizes classification. The inputs are the original images and the corresponding segmentation results.
