# TrojanNetDetector
# Practical Detection of Trojan Neural Networks
Code of Data-Limited TrojanNet Detector (DL-TND) and Data-Free TrojanNet Detector (DF-TND) from the paper: [Practical Detection of Trojan Neural Networks: Data-Limited and Data-Free Cases](https://eccv2020.eu/accepted-papers/)

## Data-Limited TrojanNet Detector (DL-TND)
### Platform
* Python: 3.7
* TensorFlow: 1.13.1
### Howto
1. Directly run the file DLTND_main (We provide the test Trojan model and the test clean model)
2. If you want to train your own models, run the file train (You could change the path and the model name)
3. The results will
## Data-Free TrojanNet Detector (DF-TND)
### Platform
* Python: 3.7
* PyTorch: 1.5.0
### Howto
1. Before running the code, please go to the robustness_lib and install the robustness package
2. Run the file main_dftnd.
3. The results will show the original images, recovered images, and the perturbations. The results will also provide the logits outputs after optimization, before optimization, and the logits output increase. The last row of the result tell you whether this is a Trojan model and what is the target label. (There are several parameters you can control. You could change gamma, which controls the sparsity of the perturbation. You could change the preset threshold T, which controls the confidence of the detection) 
4. If you want to train your own models, run the file train_model (You could change the path and the model name)
## Refer to this Rep.
If you use this code, please cite the following reference

```
@inproceedings{wang2020practical,
  title={Practical Detection of Trojan Neural Networks: Data-Limited and Data-Free Cases},  
  author={Wang, Ren and Zhang, Gaoyuan and Liu, Sijia and Chen, Pin-Yu and Xiong, Jinjun and Wang, Meng},  
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},  
  pages={},  
  year={2020}  
}
```
