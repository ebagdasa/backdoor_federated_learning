**11/20/2020: We are developing a new framework for backdoors with FL: [Backdoors101](https://github.com/ebagdasa/backdoors101).**
It extends to many new attacks (clean-label, physical backdoors, etc) and has improved user experience. Check it out!

# backdoor_federated_learning
This code includes experiments for paper "How to Backdoor Federated Learning" (https://arxiv.org/abs/1807.00459)


All experiments are done using Python 3.7 and PyTorch 1.0.

```mkdir saved_models```

```python training.py --params utils/params.yaml```


I encourage to contact me (eugene@cs.cornell.edu) or raise Issues in GitHub, so I can provide more details and fix bugs. 

Most of the experiments resulted by tweaking parameters in utils/params.yaml (for images) 
and utils/words.yaml (for text), you can play with them yourself.

## Reddit dataset
* Corpus parsed dataset: https://drive.google.com/file/d/1qTfiZP4g2ZPS5zlxU51G-GDCGGr23nvt/view?usp=sharing 
* Whole dataset: https://drive.google.com/file/d/1yAmEbx7ZCeL45hYj5iEOvNv7k9UoX3vp/view?usp=sharing
* Dictionary: https://drive.google.com/file/d/1gnS5CO5fGXKAGfHSzV3h-2TsjZXQXe39/view?usp=sharing


