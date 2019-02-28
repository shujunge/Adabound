## keras implement of [Adabound](https://openreview.net/forum?id=Bkg3g2R9FX)
refer to [AdaBound](https://github.com/Luolc/AdaBound).

## Usage
You can use AdaBound just like any other keras optimizers.
```python

from keras.optimizers import Adam
from Adabound import Adabound
my_adam=Adam(lr=0.001)
my_adabound=Adabound(lr=1e-3, final_lr=0.1) or
my_adabound=Adabound(lr=1e-3, betas=(0.9, 0.999),final_lr=0.1,gamma=1e-3,epsilon=1e-8,)
```
## Demo
Implement cifr10 classification  with the  Adabound Optimizer.
* [cifar10 example](https://colab.research.google.com/drive/130x0fV9uoohXO_iVGV2ULan4ieJz-dpo)

##Requirements
* keras>=2.2.0
* tensorflow>=1.8.0

