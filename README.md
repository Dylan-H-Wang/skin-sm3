# sm3
Official Pytorch implementation of Self-Supervised Multi-Modality Learning for Multi-Label Skin Lesion Classification (SM3). Code will be available upon paper acceptance.

# Note
Sorry for the delay of code release since our paper is still under review ... :smiling_face_with_tear:

However, we provide the minimal coding for reproducing our results in `inference.py`.

Pretrained weights are also uploaded in the [release page](https://github.com/Dylan-H-Wang/skin-sm3/releases/tag/v0.1).
* `best_linear.pth` is the weight for `SSL + Linear Probing SM3-linear`, and detailed results are shown in `linear_results.csv`.
* `best_finetune.pth` is the weight for `SSL + Fine-tuning SM3-finetune`, and detailed results are shown in `finetune_results.csv`.
* If you want to finetune on other datasets or tasks, maybe you only need backbone (ResNet-50) weights after the SSL pre-training. Then, you can also find it in `best_linear.pth` by filtering the key `extractor`.

# :scroll: Citation
If you find our [paper](https://arxiv.org/abs/2310.18583) useful for your research, please cite our work.

```
@misc{wang2023selfsupervised,
      title={Self-Supervised Multi-Modality Learning for Multi-Label Skin Lesion Classification}, 
      author={Hao Wang and Euijoon Ahn and Lei Bi and Jinman Kim},
      year={2023},
      eprint={2310.18583},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```