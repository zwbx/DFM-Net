# DFM-Net (ACM MM 2021)
This repository provides code for paper [Depth Quality-Inspired Feature Manipulation for Efficient RGB-D Salient Object Detection](https://arxiv.org/pdf/2107.01779.pdf). 

This method contains hyper light-weight RGB-D SOD model DFM-Net and its big version DFN-Net*. 

If you have any questions about our paper, feel free to contact us. 
<p align="center">
    <img src="img/structure_diagram.png" width="80%"/> <br />
 <em> 
     Block diagram of the proposed BTS-Net.
    </em>
</p>

## Features
- High accuracy: Experiments on 6 public datasets demonstrate that the proposed DFM-Net achieves state-of-the-art performance even compared to non-light-weight models.
- High Speed: cost 140ms on CPU (Core i7-8700 CPU), which is 2.2× faster than the prior fastest efficient model A2dele.
- Low model size: Without any general model compression technology (such as quantification, distillation), its model size is only 8.5Mb, which is 14.9% of the prior smallest model A2dele.



## results

<p align="center">
    <img src="img/quantitative_results.png" width="95%"/> <br />
 <em> 
     Quantitative comparison with 15 SOTA over 4 metrics (S-measure, max F-measure, max E-measure and MAE) on 6 datasets.
    </em>
</p>

<p align="center">
    <img src="img/benchmark.png" width="40%"/> <br />
 <em> 
      Performance visualization. The vertical axis indicates the accuracy on SIP. The horizontal axis indicates the CPU speed (FPS). The circle area is proportional
to the model size.
    </em>
</p>

###  Download
 - We provide test results of above datasets for DFM-Net and the big version DFM-Net*.
 - [DFM-Net](https://pan.baidu.com/s/1e3iqZZHGjy2OAUxVwHhDeg) [code: 0ybv].
 - [DFM-Net*](https://pan.baidu.com/s/1kwp9Y8lbwnXhLgMuaVjgrA) [code: mho4].
 
## Citation

Please cite the following paper if you use this repository in your reseach

	@inproceedings{Zhang2021DFMNet,
 	 title={Depth Quality-Inspired Feature Manipulation for Efficient RGB-D Salient Object Detection},
	  author={Wenbo Zhang and Ge-Peng Ji and Zhuo Wang and Keren Fu and Qijun Zhao},
	  booktitle={ACM MM 2021},
	  year={2021}
	}
