# Require to run the train script under the same directory to run this hls script.
python3 hls4ml_build.py --load_model ./exp_cifar_bayes_resnet/cifar_resnet_spt_2samples --output_dir ./exp_cifar_bayes_resnet/cifar_resnet_spt_2samples_hls --strategy latency  --model_name resnet
python3 hls4ml_build.py --load_model ./exp_cifar_bayes_resnet/cifar_resnet_spt_3samples --output_dir ./exp_cifar_bayes_resnet/cifar_resnet_spt_3samples_hls --strategy latency  --model_name resnet
python3 hls4ml_build.py --load_model ./exp_cifar_bayes_resnet/cifar_resnet_spt_5samples --output_dir ./exp_cifar_bayes_resnet/cifar_resnet_spt_5samples_hls --strategy latency  --model_name resnet
python3 hls4ml_build.py --load_model ./exp_cifar_bayes_resnet/cifar_resnet_spt_7samples --output_dir ./exp_cifar_bayes_resnet/cifar_resnet_spt_7samples_hls --strategy latency  --model_name resnet
python3 hls4ml_build.py --load_model ./exp_cifar_bayes_resnet/cifar_resnet_spt_9samples --output_dir ./exp_cifar_bayes_resnet/cifar_resnet_spt_9samples_hls --strategy latency  --model_name resnet