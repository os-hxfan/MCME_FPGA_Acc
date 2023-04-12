# Require to run the train script under the same directory to run this hls script.
python3 hls4ml_build.py --load_model ./exp_svhn_bayes_vgg/svhn_vgg_tmp_0bayeslayer --output_dir ./exp_svhn_bayes_vgg/svhn_vgg_tmp_0bayeslayer_hls --strategy resource  --model_name vgg
python3 hls4ml_build.py --load_model ./exp_svhn_bayes_vgg/svhn_vgg_tmp_1bayeslayer --output_dir ./exp_svhn_bayes_vgg/svhn_vgg_tmp_1bayeslayer_hls --strategy resource  --model_name vgg
python3 hls4ml_build.py --load_model ./exp_svhn_bayes_vgg/svhn_vgg_tmp_2bayeslayer --output_dir ./exp_svhn_bayes_vgg/svhn_vgg_tmp_2bayeslayer_hls --strategy resource  --model_name vgg
python3 hls4ml_build.py --load_model ./exp_svhn_bayes_vgg/svhn_vgg_tmp_3bayeslayer --output_dir ./exp_svhn_bayes_vgg/svhn_vgg_tmp_3bayeslayer_hls --strategy resource  --model_name vgg
python3 hls4ml_build.py --load_model ./exp_svhn_bayes_vgg/svhn_vgg_tmp_4bayeslayer --output_dir ./exp_svhn_bayes_vgg/svhn_vgg_tmp_4bayeslayer_hls --strategy resource  --model_name vgg
python3 hls4ml_build.py --load_model ./exp_svhn_bayes_vgg/svhn_vgg_tmp_5bayeslayer --output_dir ./exp_svhn_bayes_vgg/svhn_vgg_tmp_5bayeslayer_hls --strategy resource  --model_name vgg
python3 hls4ml_build.py --load_model ./exp_svhn_bayes_vgg/svhn_vgg_tmp_6bayeslayer --output_dir ./exp_svhn_bayes_vgg/svhn_vgg_tmp_6bayeslayer_hls --strategy resource  --model_name vgg
python3 hls4ml_build.py --load_model ./exp_svhn_bayes_vgg/svhn_vgg_tmp_7bayeslayer --output_dir ./exp_svhn_bayes_vgg/svhn_vgg_tmp_7bayeslayer_hls --strategy resource  --model_name vgg