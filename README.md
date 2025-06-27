# Activation Fine-Tuning of Convolutional Neural Networks for Improved Input Attribution Based on Class Activation Maps, Applied Sciences'22
**Sungmin Han, Jeonghyun Lee, Sangkyun Lee (Corresponding author)**  

AIRLAB, School of Cybersecurity, Korea University

### 📄 Abstract

Model induction is one of the most popular methods to extract information to better understand AI's decisions by estimating the contribution of input features for a class of interest. However, we found a potential issue: most model induction methods, especially those that compute class activation maps, rely on arbitrary thresholding to mute some of their computed attribution scores. From our observation that the quality of input attribution can be improved by more careful threshold optimization, we suggest a simple procedure with a new quality metric to choose an optimal cut-off threshold of attribution scores. Since the optimal threshold has to be computed on the per-input basis, we further suggest an activation fine-tuning framework using thresholding masks as auxiliary data, which guides the original convolutional neural network to produce more regulated activations better suited for computing input attribution based on class activation maps. Our experiments show that the suggested activation fine-tuning can significantly improve the quality of input attribution of the underlying convolutional neural networks, making the application of per-input threshold optimization unnecessary.

Keywords: Activation Fine-Tuning (AFT); Class Activation Map; Input Attribution; Convolutional Neural Network

You can read the full paper at  
👉 [Official Paper] (https://www.mdpi.com/2076-3417/12/24/12961)

---

## Citation

If you found this work or code useful, please cite us:

```
@article{han2022activation,
  title={Activation Fine-Tuning of Convolutional Neural Networks for Improved Input Attribution Based on Class Activation Maps},
  author={Han, Sungmin and Lee, Jeonghyun and Lee, Sangkyun},
  journal={Applied Sciences},
  volume={12},
  number={24},
  pages={12961},
  year={2022},
  publisher={MDPI}
}
```
