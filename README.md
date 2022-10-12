# dvc-hello-world
This is a simple "Hello World" project for DVC. In this spirit, this project contains the necessary python scripts to 
train a Convolutional Network, to classify digits of the classic MNIST image dataset. This is: (i) training and testing 
datasets, (ii) some python scripts an (iii) a trained model.

Although it seems simple, this set of files represents the assets we have in a Machine Learning project 99% of the time, 
and it will be the scenario that will allow us to introduce DVC

The project is organized as follows:

```shell
.
├── data
│    ├── 4.jpeg
│    ├── 7.jpeg
│    ├── delete.csv
│    ├── test.csv
│    └── train.csv
├── model
│    └── mnist_model.h5
├── model.py
├── requirements.txt
└── train.py
```

**data**: Inside this folder, you will find train / test data sets in CSV format, together with two examples of 
images to test the model. Read section **About MNIST** for more details.

**model**: Inside this folder is the trained model in the h5 TensorFlow format.

**model.py**: Script that define model architecture and train/predict methods.

**train.py**: Just a main script to put everything together.

**requirements.txt**: Project dependencies

## DVC: Step-by-step

In this section, we will integrate DVC into this project, step-by-step.

1. Installation
```shell
$ pip install dvc[all]
```

2. Add remote storage: Add the remote source address based on the cloud provider of your choice (s3, azure, gcloud, etc.).
As an example, these are how remote storage would look in two of the top cloud providers:
- AWS (S3 service): `s3://my-storage/dvc-hello-world`
- Azure (Blob Storage): `azure://my-storage/dvc-hello-world`

```shell
$ dvc remote add -d storage [MY_STORAGE_ADDRESS_URL]
```

3. Initialization: Run following command from project root, to initialize DVC
```shell
$ dvc init
```

A few internal files are created that should be added to Git:

```shell
$ git status

Changes to be committed:
        new file:   .dvc/.gitignore
        new file:   .dvc/config
        new file:   .dvcignore

```

These changes are important so you should commit it!

```shell
$ git commit -m "Initialize DVC"
```

4. Add data folder to DVC remote storage:

First is required to remove `data` folder from git tracked files, to do so execute the following lines.
```shell
git rm -r --cached 'data'
git commit -m "stop tracking data"
```

Now we are in conditions to add `data` folder to DVC tracked files, to do so execute the following command:
```shell
$ dvc add data
```

DVC stores information about the added file in a special .dvc file named data.dvc — a small text file with a 
human-readable format. This metadata file is a placeholder for the original data, and can be easily versioned like 
source code with Git

```shell
$ git add data.dvc data/ .gitignore
$ git commit -m "Add raw data into DVC"
$ git push
```

5. Upload changes in remote storage
```shell
$ dvc push 
```

A few health checks:

- Check file `.dvc/config` content, should contain the remote defined above. Keep in mind that the file content will reflect your storage address.
```yaml
[core]
    remote = storage
['remote "storage"']
    url = [MY_STORAGE_ADDRESS_URL]
```

- A file named data.dvc must be created after execute `dvc add data` command. Check that the content is similar to this:
```yaml
- md5: 1b4f3d00866b57617ad294d85f051dcb.dir
  size: 109638334
  nfiles: 4
  path: data
```

Now you have the steps, you can repeat step 3 and 4 to also exclude model folder from git and store the model 
binaries in the remote storage using DVC.

## About MNIST

The MNIST database of handwritten digits is a good database for people who want to try learning techniques and pattern 
recognition methods on real-world data while spending minimal efforts on preprocessing and formatting. 

The dataset is inside a CSV file, where each row represents a single image example. For each row, the first column represents 
the label, and the rest columns represent each pixel value. Since images are 28x28 pixels in gray scale, 
each row has 785 columns: 1 label + 784 (28x28).

Examples:

![imagen](data/7.jpeg)
![imagen](data/4.jpeg)

**Read more about the MNIST dataset in the following link:** [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/).

## Setup

In the event that you intend to retrain this model, follow the steps below. Before training, be sure that the data 
directory with the corresponding training and evaluation data exists.

1. Install dependencies:
```shell
pip install -r requirements.txt
```

2. Train model
```shell
python train.py
```

## Common errors:

#### Folder/File is already tracked by Git

**Error**:
```shell
ERROR:  output 'data' is already tracked by SCM (e.g. Git).
    You can remove it from Git, then add to DVC.
        To stop tracking from Git:
            git rm -r --cached 'data'
            git commit -m "stop tracking data" 
```

**Solution**: Follow output instructions to untrack folder/file from git and start tracking using DVC.
