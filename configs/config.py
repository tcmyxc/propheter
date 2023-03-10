root = r"/mnt/nfs1-homes/xwx"

# the result root dir
output_dir = root + '/model-doctor-xwx/output'

# image data dir
output_data = root + "/dataset"

# channel, mask
output_result = output_dir + '/result'

# image data
data_cifar10            = output_data + '/cifar10/images'
data_cifar100           = output_data + '/cifar100/images'

data_imagenet_lt        = output_data + "/ImageNet_LT"

data_places_lt          = output_data + "/Places_LT"

data_inaturalist2018    = output_data + "/iNaturalist2018"

data_cifar10_lt_ir10    = output_data + "/cifar10_lt_ir10/images"
data_cifar10_lt_ir50    = output_data + "/cifar10_lt_ir50/images"
data_cifar10_lt_ir100   = output_data + "/cifar10_lt_ir100/images"

data_cifar100_lt_ir10   = output_data + "/cifar100_lt_ir10/images"
data_cifar100_lt_ir50   = output_data + "/cifar100_lt_ir50/images"
data_cifar100_lt_ir100  = output_data + "/cifar100_lt_ir100/images"

# result
result_channels      = output_result + '/channels'

# model
output_model = output_dir + '/model'
model_pretrained = output_model + '/pretrained'
model_retrain = output_model + '/retrain'
model_im2b = output_model + '/im_use_balance_data'
