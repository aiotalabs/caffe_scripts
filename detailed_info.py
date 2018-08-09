import caffe
from numpy import prod

def print_detailed_net_parameters (deploy_file):
    net = caffe.Net(deploy_file, caffe.TEST)
    initial_layer_weights = 0
    conv_1x1_weights = 0
    conv_3x3_weights = 0
    conv_5x5_weights = 0
    conv_7x7_weights = 0
    inner_product_weights = 0
    batchnorm_scale_weights = 0
    
    initial_layer_macc = 0
    conv_1x1_macc = 0
    conv_3x3_macc = 0
    conv_5x5_macc = 0
    conv_7x7_macc = 0
    inner_product_macc = 0
    batchnorm_scale_macc = 0

    conv_counter = 0
    for idx, layer in enumerate(net.layers):
        if layer.type == "Convolution":
            conv_counter += 1
            top_tensor = net.blobs[net._blob_names[net._top_ids(idx)[0]]].data.shape
            top_shape = top_tensor[2]*top_tensor[3]
            if conv_counter == 1:
                initial_layer_weights = prod(layer.blobs[0].data.shape)
                initial_layer_macc = prod(layer.blobs[0].data.shape) * top_shape
            else :
                if (layer.blobs[0].data.shape[2], layer.blobs[0].data.shape[3]) == (1,1):
                    conv_1x1_weights += prod(layer.blobs[0].data.shape)
                    conv_1x1_macc += prod(layer.blobs[0].data.shape) * top_shape
                elif (layer.blobs[0].data.shape[2], layer.blobs[0].data.shape[3]) == (3,3):
                    conv_3x3_weights += prod(layer.blobs[0].data.shape)
                    conv_3x3_macc += prod(layer.blobs[0].data.shape) * top_shape
                elif (layer.blobs[0].data.shape[2], layer.blobs[0].data.shape[3]) == (5,5):
                    conv_5x5_weights += prod(layer.blobs[0].data.shape)
                    conv_5x5_macc += prod(layer.blobs[0].data.shape) * top_shape
                elif (layer.blobs[0].data.shape[2], layer.blobs[0].data.shape[3]) == (7,7):
                    conv_7x7_weights += prod(layer.blobs[0].data.shape)
                    conv_7x7_macc += prod(layer.blobs[0].data.shape) * top_shape
        elif layer.type == "InnerProduct":
            inner_product_weights += prod(layer.blobs[0].data.shape)
            inner_product_macc += prod(layer.blobs[0].data.shape)
        elif layer.type == "BatchNorm":
            top_tensor = net.blobs[net._blob_names[net._top_ids(idx)[0]]].data.shape
            batchnorm_scale_weights += 2*prod(layer.blobs[0].data.shape)
            batchnorm_scale_macc += top_tensor[2] * top_tensor[3] * top_tensor[1]

    print ("[INFO] Total number of parameters in " + deploy_file + ": " + str(initial_layer_weights+conv_1x1_weights+conv_3x3_weights+conv_5x5_weights+conv_7x7_weights+inner_product_weights+batchnorm_scale_weights))
    print ("[INFO] Total number of MACC       in " + deploy_file + ": " + str(initial_layer_macc+conv_1x1_macc+conv_3x3_macc+conv_5x5_macc+conv_7x7_macc+inner_product_macc+batchnorm_scale_macc))

    if initial_layer_weights   != 0 : print("[INFO] Parameters in initial layer           : %9d  MAC : %12d" %(initial_layer_weights , initial_layer_macc))
    if conv_1x1_weights        != 0 : print("[INFO] Parameters in conv_1x1_weights        : %9d  MAC : %12d" %(conv_1x1_weights , conv_1x1_macc))
    if conv_3x3_weights        != 0 : print("[INFO] Parameters in conv_3x3_weights        : %9d  MAC : %12d" %(conv_3x3_weights , conv_3x3_macc))
    if conv_5x5_weights        != 0 : print("[INFO] Parameters in conv_5x5_weights        : %9d  MAC : %12d" %(conv_5x5_weights , conv_5x5_macc))
    if conv_7x7_weights        != 0 : print("[INFO] Parameters in conv_7x7_weights        : %9d  MAC : %12d" %(conv_7x7_weights , conv_7x7_macc))
    if inner_product_weights   != 0 : print("[INFO] Parameters in inner_product_weights   : %9d  MAC : %12d" %(inner_product_weights , inner_product_macc))
    if batchnorm_scale_weights != 0 : print("[INFO] Parameters in batchnorm_scale_weights : %9d  MAC : %12d" %(batchnorm_scale_weights , batchnorm_scale_macc))