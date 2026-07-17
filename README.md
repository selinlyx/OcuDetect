# OcuDetect

Download the dataset here (not preprocessed one): https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

Clone repository

```
git clone https://github.com/selinlyx/OcuDetect
cd OcuDetect
```

Set up Python environment

```
conda create --name ocudetect python=3.12
```

Install dependencies
```
pip install -r requirements.txt
```

Run the following files in order
`image_label_processing.py`
`data_processing.py`
`split.py`

Run the `train.py` file to train

Run the `evaluate.py` file to evaluate