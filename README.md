# multi-label softmax networks for pulmonary nodule classification using unbalanced and dependent categories


## Preprocess LIDC data
To generate multi-label classification dataset,
1. Download LIDC-IDRI dataset.
2. Install [_pylid_](https://github.com/notmatthancock/pylidc) library, and set LIDC dataset to its path (see [tutorial](https://pylidc.github.io/)).
3. Run the following script:
   `python genlidc_multilabel_data.py --data_dir ../LIDC_DATA `
   Set data dir as you want. 

## For run
`python main configs/test.json`

## Citation and contact


## Acknowledgements
We express our deeply appreciation to the author of this [blog](https://kexue.fm/archives/7359) for the valuable formulation for multi-label classification. 
This equation excitate our consideration to address data imbalance issue by ranking within and between labels. 
We also recommend readers of interest to refer to their research paper: [ZLPR: A Novel Loss for Multi-label Classification](https://arxiv.org/abs/2208.02955). 
