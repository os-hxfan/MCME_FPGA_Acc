# As we just need the model to test hardware performance, number of epoch is set as 5 to reduce the training time
python3 train_qkeras_mcme.py --dataset mnist --num_epoch 5 --batch_size 128 --lr 0.01 --gpus 1 --save_model mnist_lenet_spt_2samples --quant_tbit 8 --model_name lenet --save_dir ./exp_mnist_bayes_lenet --opt_mode spatial --mc_samples 2 
python3 train_qkeras_mcme.py --dataset mnist --num_epoch 5 --batch_size 128 --lr 0.01 --gpus 1 --save_model mnist_lenet_spt_3samples --quant_tbit 8 --model_name lenet --save_dir ./exp_mnist_bayes_lenet --opt_mode spatial --mc_samples 3
python3 train_qkeras_mcme.py --dataset mnist --num_epoch 5 --batch_size 128 --lr 0.01 --gpus 1 --save_model mnist_lenet_spt_5samples --quant_tbit 8 --model_name lenet --save_dir ./exp_mnist_bayes_lenet --opt_mode spatial --mc_samples 5
python3 train_qkeras_mcme.py --dataset mnist --num_epoch 5 --batch_size 128 --lr 0.01 --gpus 1 --save_model mnist_lenet_spt_7samples --quant_tbit 8 --model_name lenet --save_dir ./exp_mnist_bayes_lenet --opt_mode spatial --mc_samples 7
python3 train_qkeras_mcme.py --dataset mnist --num_epoch 5 --batch_size 128 --lr 0.01 --gpus 1 --save_model mnist_lenet_spt_9samples --quant_tbit 8 --model_name lenet --save_dir ./exp_mnist_bayes_lenet --opt_mode spatial --mc_samples 9