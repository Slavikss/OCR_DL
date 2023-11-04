## Chinese Number OCR Detector

This repository contains the code for an Optical Character Recognition (OCR) system designed specifically to detect and recognize Chinese numerical characters.

### Overview

The OCR system is designed to process images and extract Chinese numerical characters with high accuracy. The script uses image processing techniques and machine learning models to identify and interpret the numbers.

## Description
This model extracts number of car license plate from image 

**Input example:** 
![Автомобильный номер](https://algocode.ru/files/course_dlfall22/number.png) \
**Output example:** 皖AD16688


## Architecture
Fully-convolutional CNN (CNN) and Bi-LSTM. At the output of CTC-loss. \

![Архитектура](https://algocode.ru/files/course_dlfall22/architecture.png)
### Prerequisites

To run this script, you need the following:

- Python 3.6 or higher
-  PyTorch (depending on the model implementation)
- torchmetrics



