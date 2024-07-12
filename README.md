# scCNM

scCNM is a deep joint representation learning framework. The backbone of scCNM is a dual autoencoder that jointly processes single-cell copy number and point mutation data. It fuses individual and commonality information among the two data modalities of the cells to identify cell subpopulations and clonal mutation patterns.



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
Create a new environment named "sccnm":
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
pip install torch_geometric einops timm
```

Now you are ready to run **scCNM**!

## Usage

scCNM uses single-cell copy number and point mutation data to aggregate tumor cells into distinct subpopulations.

Example:

```
python run_scCNM.py --gpu 0 --cnv ./data/real/A/CHISEL_A_CN.txt --snv ./data/real/A/CHISEL_A_SNV.txt --label ./data/real/A/A_label.txt --output ./data
```

## Input Files

The input data should contain paired single-cell copy number and point mutation data. Both data are in matrix form, with each row representing a cell and each column representing a copy number state or point mutation. 
We use the true labels of cells to calculate ARI and NMI to assess the quality of clustering.

## Output Files
The output directory is provided by users.

### Low-dimensional representations

The low-dimensional representations are written to a file with name "latent.txt".

### Cell labels

The cell-to-cluster assignments are written to a file with name "label.txt".

### Reconstructed SNV data

The reconstructed SNV data are written to a file with name "rec-snv.txt".

## Arguments

* `--cnv <filename>` Replace \<filename\> with the file containing the  copy number matrix.
* `--snv <filename>` Replace \<filename\> with the file containing the snv matrix.
* `--label <filename>` Replace \<filename\> with the file containing the real label of cells.
* `--output <directory>` Replace \<directory\> with the directory to save results.


## Optional arguments

Parameter | Description | Possible values
---- | ----- | ------
--epochs | number of epoches to train scCNM | Ex: epochs=100  default:150
--lr | learning rate | Ex: lr=0.0005  default:0.0001
--latent_dim | latent dimension | Ex: latent_dim=30  default:50
--w | the hyperparameter lambda | Ex: w=500  default:1000
--neighbors | number of neighbors for each cell to construct the cell graph | Ex: neighbors=10  default:5
--seed | random seed | Ex: seed=0  default:1


## Contact

If you have any questions, please contact 12023132086@stu.nxu.edu.cn.