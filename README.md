# scCNM

scCNM is a deep joint representation learning framework. The backbone of scCNM is a dual autoencoder that jointly processes single-cell copy number and point mutation data. It fuses individual and commonality information among the two data modalities of the cells to generate embeddings and identify cell subpopulations.



## Requirements

* Python 3.9+.

# Installation
## Clone repository
First, download scCNM from github and change to the directory:
```bash
git clone https://github.com/zhyu-lab/sccnm
cd sccnm
```

## Create conda environment (optional)
Create a new environment named "scCNM":
```bash
conda create --name sccnm python=3.9.16
```

Then activate it:
```bash
conda activate sccnm
```

## Install requirements
Use pip to install the requirements:
```bash
python -m pip install -r requirements.txt
```

Now you are ready to run **scCNM**!

## Usage

scCNM uses single-cell copy number and point mutation data to aggregate tumor cells into distinct subpopulations.

Example:

```
python run_scCNM.py
```

## Input Files

The input data should contain paired single-cell copy number and point mutation data. Both data are in matrix form, with each row representing the variant or mutation status of each cell and each column representing a copy number variant or point mutation. 
We use the true labeling of cells to calculate ARI and NMI to assess the quality of clustering.

## Output Files
The output directory is provided by users.

### Low-dimensional representations

The low-dimensional representations are written to a file with name "latent.txt".

### Cell labels

The cell-to-cluster assignments are written to a file with name "labels.txt".
## Arguments

* `--cnv <filename>` Replace \<filename\> with the file containing the  copy number matrix.
* `--snv <filename>` Replace \<filename\> with the file containing the snv matrix.
* `--label <filename>` Replace \<filename\> with the file containing the real label of cells.


## Optional arguments

Parameter | Description | Possible values
---- | ----- | ------
--epochs | number of epoches to train the scCNM | Ex: epochs=100  default:30
--lr | learning rate | Ex: lr=0.0001  default:0.0006
--latent_dim | latent dimension | Ex: latent_dim=3  default:10
--kernel_size | latent dimension | Ex: latent_dim=5  default:7
--seed | random seed | Ex: seed=0  default:1


## Contact

If you have any questions, please contact lrx102@stu.nxu.edu.cn.