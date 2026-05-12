# CFSPMNet

## Paper Title

CFSPMNet: Cross-subject Fourier-guided Spatial-Patch Mamba Network for EEG Motor Imagery Decoding in Stroke Patients

## SPPM & FRSM

![Fameworks](./Fig/Fig_framework.png)

![Model](./Fig/Fig_SPPM.png)

![Fig_Model](./Fig/Fig_Model.png)

## Results

The experimental and visualization results are presented below.
![Results](./Fig/results.png)
![Vis](./Fig/Fig_OverallVis.png)

## Environment

Code developed and tested in Python 3.12.12 using PyTorch 2.5.1.
```
Python      : 3.12.12
PyTorch     : 2.5.1
CUDA        : 12.4
Device      : cuda
```


## Datasets
The experiments are conducted on publicly available datasets, which can be accessed at:

- XW-Stroke: https://doi.org/10.6084/m9.figshare.21679035.v5 
- 2019-Stroke: https://doi.org/10.6084/m9.figshare.7636301


## Citation
If you find our codes helpful, please star our project and cite our following papers:
```
@misc{wang2026cfspmnetcrosssubjectfourierguidedspatialpatch,
      title={CFSPMNet: Cross-subject Fourier-guided Spatial-Patch Mamba Network for EEG Motor Imagery Decoding in Stroke Patients}, 
      author={Xiangkai Wang and Yun Zhao and Dongyi He and Qingling Xia and Gen Li and Xinlai Xing and Yuchi Pan and Bin Jiang},
      year={2026},
      eprint={2605.10111},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.10111}, 
}
```
