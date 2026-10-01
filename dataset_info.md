Cuff-Less Blood Pressure Estimation
Pre-processed and cleaned vital signals for cuff-less BP estimation

dataset link - https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset

About Dataset
Data Set Information:

The main goal of this data set is providing clean and valid signals for designing cuff-less blood pressure estimation algorithms. The raw electrocardiogram (ECG), photoplethysmograph (PPG), and arterial blood pressure (ABP) signals are originally collected from the physionet.org and then some preprocessing and validation performed on them. (For more information about the process please refer to our paper)

Attribute Information:

This database consists of a cell array of matrices, each cell is one record part.
In each matrix each row corresponds to one signal channel:

1: PPG signal, FS=125Hz; photoplethysmograph from fingertip

2: ABP signal, FS=125Hz; invasive arterial blood pressure (mmHg)

3: ECG signal, FS=125Hz; electrocardiogram from channel II

Note: dataset is splitted to multiple parts to make it easier to load on machines with low memory. Each cell is a record. There might be more than one record per patient (which is not possible to distinguish). However, records of the same patient appear next to each other. N-fold cross test and train is suggested to reduce the chance of trainset being contaminated by test patients.

Relevant Papers:

M. Kachuee, M. M. Kiani, H. Mohammadzade, M. Shabany, Cuff-Less High-Accuracy Calibration-Free Blood Pressure Estimation Using Pulse Transit Time, IEEE International Symposium on Circuits and Systems (ISCAS'15), 2015.

A. Goldberger, L. Amaral, L. Glass, J. Hausdorff, P. Ivanov, R. Mark, J.Mietus, G. Moody, C. Peng and H. Stanley, â€œPhysiobank, physiotoolkit, and physionet components of a new research resource for complex physiologic signals,â€ Circulation, vol. 101, no. 23, pp. 215â€“220, 2000.
Citation Request:

If you found this data set useful please cite the following:

M. Kachuee, M. M. Kiani, H. Mohammadzade, M. Shabany, Cuff-Less High-Accuracy Calibration-Free Blood Pressure Estimation Using Pulse Transit Time, IEEE International Symposium on Circuits and Systems (ISCAS'15), 2015.

M. Kachuee, M. M. Kiani, H. Mohammadzadeh, M. Shabany, Cuff-Less Blood Pressure Estimation Algorithms for Continuous Health-Care Monitoring, IEEE Transactions on Biomedical Engineering, 2016.
