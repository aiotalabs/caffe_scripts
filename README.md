# caffe_scripts
Scripts to evaluate caffe models

## Detailed Information of a network
Use prototxt file to evaluate a network's contribution of 1x1/3x3/5x5/7x7 Convolutions, InnerProduct, BatchNorm and Scale towards parameter and MAC.

### Example output of [detailed_info.py](https://github.com/aiotalabs/caffe_scripts/blob/master/detailed_info.py)
```
[INFO] Input Prototxt   : models/resnet50/ResNet-50-deploy.prototxt
[INFO] Input Caffemodel : models/resnet50/ResNet-50-model.caffemodel
[INFO] Total number of parameters in models/resnet50/ResNet-50-deploy.prototxt: 25556032
[INFO] Total number of MACC       in models/resnet50/ResNet-50-deploy.prototxt: 3868560384
[INFO] Parameters in initial layer           :      9408  MAC :    118013952
[INFO] Parameters in conv_1x1_weights        :  12128256  MAC :   1888223232
[INFO] Parameters in conv_3x3_weights        :  11317248  MAC :   1849688064
[INFO] Parameters in inner_product_weights   :   2048000  MAC :      2048000
[INFO] Parameters in batchnorm_scale_weights :     53120  MAC :     10587136


[INFO] Output Prototxt   : models/resnet50/emdnn.prototxt
[INFO] Output Caffemodel : models/resnet50/emdnn.caffemodel
[INFO] Total number of parameters in models/resnet50/emdnn.prototxt: 7644341
[INFO] Total number of MACC       in models/resnet50/emdnn.prototxt: 1312133266
[INFO] Parameters in initial layer           :      9408  MAC :    118013952
[INFO] Parameters in conv_1x1_weights        :   4231576  MAC :    926666440
[INFO] Parameters in conv_3x3_weights        :   1302237  MAC :    254817738
[INFO] Parameters in inner_product_weights   :   2048000  MAC :      2048000
[INFO] Parameters in batchnorm_scale_weights :     53120  MAC :     10587136
```
