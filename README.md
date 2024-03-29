This is the code for the paper 
# F2DeepRS: A Deep Recommendation Framework Applied to ICRC Platform
accepted by _the 34th International Conference on Industrial, Engineering & Other Applications of Applied Intelligent Systems_ (IEA/AIE 2021).

The proposed framework belongs to hybrid recommendation systems. The core idea of the F2DeepRS framework is to leverage well-trained user, item or content latent representations and learn the matching function. The model we built runs much faster in calculation/referencing (saved 70% time), and is thus applicable to environtments where calculation resources is not sufficient, in our case, the Information and computational resource constrained (ICRC) platform.

To implement the BPR algorithm, please refer to [BPR](https://implicit.readthedocs.io/en/latest/bpr.html). We got the inspirations in part from the DeepCF model, which can be found at  [DeepCF](https://github.com/familyld/DeepCF). Many thanks to the authors. Please be advised that the dataset for learning user/item representations and the dataset for learning the matching function are different. For details refer to our paper.

The dataset YahooR2 music dataset can be downloaded at [Yahoo! R2](https://webscope.sandbox.yahoo.com/). You need to download the data, explore it, and preprocess the data by yourself.

The configuration of the ICRC platform, and any details related to the in-vehicle deployment are confidential due to our internal confidential protocals. They are therefore not released here.

# requirements
- Python 3
- Pytorch 1.2.0 or later

# About the code
- config.py: configurations
- contentNet_run.py: definition of an affliated model for obtaining content representations, can be replaced with other content representation learning methods, or ignored if contents are unavailable.
- F2deepRS_net.py: definitions of model architectures
- F2deepRS_run.py: python script, train the F2DeepRS model
- F2deepRSwithContent_run.py: python script, learn the content representation, can be ignored if contents are unavailable.
- letor_metrics.py: written by other people.  provide some functions of metrics.
- load_eval_run.py: load saved model and evaluate the performance.
- utils.py: some helpful functions

If you find the code useful, please cite our paper "F2DeepRS: A Deep Recommendation Framework Applied to ICRC Platform":

@inproceedings{xie2021f2deeprs,\
  title={F2DeepRS: A Deep Recommendation Framework Applied to ICRC Platforms},\
  author={Xie, Yongquan and Tseng, Finn and Johannes, Kristinsson and Qiu, Shiqi and Murphey, Yi Lu},\
  booktitle={International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},\
  pages={3--14},\
  year={2021},\
  organization={Springer}\
}
