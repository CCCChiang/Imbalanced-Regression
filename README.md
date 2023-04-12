# Imbalanced-Regression
原作者的colab範例程式碼<img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;">  
原作者的github連結: <a> https://github.com/YyzHarry/imbalanced-regression </a>  

# 使用說明
1. 所有可用參數及說明皆在main.py裡，可以直接去裡面查看與設定
<pre>
cd imbalanceregression
python main.py --data_dir ${data_dir} --model ${model} --batch_size 256 256 256 --epoch 200 --lds --reweight "sqrt_inv" --fds --fds_kernel 'gaussian' --fds_ks 5 --fds_sigma 1 --start_update 0 --start_smooth 1 --bucket_num 15 --bucket_start 3 --fds_mmt 0.9
</pre>
2. 程式碼裡內建的模型是Linear Regression與Fully Connect Feedforward Network作為示範
3. 若需要應用於新的資料集，需要自行新增DataLoader於dataloader.py，並將train.py第32-66行改成您的Data Preprocess
4. 若需要應用LDS或FDS於其他深度學習模型，建議是將LDS與FDS模塊拆出來，鑲嵌至您的模型中。
