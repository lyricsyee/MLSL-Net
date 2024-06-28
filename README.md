# multi-label softmax networks for pulmonary nodule classification using unbalanced and dependent categories


## Preprocess LIDC data
To generate multi-label classification dataset, 

1. Download [LIDC-IDRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254) dataset.

2. Install [_pylid_](https://github.com/notmatthancock/pylidc) library, and set LIDC dataset to its path (see [tutorial](https://pylidc.github.io/)).

3. Run the following script:  

    `python genlidc_multilabel_data.py --data_dir ../LIDC_DATA `
   
Set `--data_dir` as you want. 

## 
For run the code, use the following script:

    `python main configs/test.json`

Please check the data path you have set when producing data. 

## Citation and contact
For academic inference, please use the following citation

    <code>
        @article{yi2023multi,
            title={Multi-label softmax networks for pulmonary nodule classification using unbalanced and dependent categories},
            author={Yi, Le and Zhang, Lei and Xu, Xiuyuan and Guo, Jixiang},
            journal={IEEE Transactions on Medical Imaging},
            volume={42},
            number={1},
            pages={317--328},
            year={2023},
            publisher={IEEE}
        }
    </code>

Feel free to contact me via `odriewyile@gmail.com`.

## Acknowledgements
We express our deep appreciation to the author of this [blog](https://kexue.fm/archives/7359) for the valuable formulation and code. This function is helpful, shaping our research to address data imbalance and label depedency issues. We also recommend interested readers refer to their research paper: [ZLPR: A Novel Loss for Multi-label Classification](https://arxiv.org/abs/2208.02955). 
