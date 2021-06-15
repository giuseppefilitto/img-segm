# Predicting the response to neo-adjuvant chemo-radiotherapy in colorectal cancer


Colorectal cancer is a malignant neoplasm of the large intestine resulting from the uncontrolled proliferation of one of the cells making up the colorectal tract.  
In Western countries, colorectal cancer is the second largest malignant tumor after that of the breast in women and the third after that of the lung and prostate in men.
Risk factors for this kind of cancer include colon polyps, long-standing ulcerative colitis, diabetes II and genetic history (HNPCC or Lynch syndrome).
In order to get information about diagnosis, therapeutic effect evaluation on colorectal cancer, radiomic analysis can be performed on radiological images through the application of dedicated radiomic algorithms based on segmentation and features extraction.
As regard segmentation, in clinical routines, it is carried with manual or semi-manual techniques by radiologists, but this process is:
* time-consuming
* highly operator-dependent
* subject to operator expertise

The aim of this project is to implement an automatic pipeline based on automatic segmentation of T2 weighted Magnetic Resonance (MR) colorectal cancer images exploiting Convolutional Neural Networks in order to predict the response to neo-adjuvant chemo-radiotherapy in colorectal cancer using radiomics features such as:
* GLCM gray level recurrence matrices
* ZSM zone sizes matrices
* NGTDM neighborhood gray tone difference matrix