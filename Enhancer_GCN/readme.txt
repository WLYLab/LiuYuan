本项目使用DNA的二级结构作为输入对增强子进行预测
参考论文ncRNA Classification with Graph Convolutional Networks而来
目录结构：
	data文件夹下存放数据集
	models_family_classification文件夹存放训练来的模型
	results_family_classification文件夹下存放结果
	src存放代码
使用方法：
	1.首先使用ViennaRNA(https://www.tbi.univie.ac.at/RNA/)对多有数据集(训练集，测试集，验证集都放在一个fasta文件里)的DNA的二级结构进行预测，预测得到的二级结构文件GM12878_200bp_ViennaRNA_predict_fold.txt。
	2.使用pickle_fold.py程序处理GM12878_200bp_ViennaRNA_predict_fold.txt文件得到GM12878_200bp_fold.pkl
	3.请在training/train_model.py以及evaluate_model.py文件中修改相应的文件路径，网络参数等
	4.运行命令：
		训练：
			cd src/
			python training/train_model.py
		测试：
			python evaluation/evaluate_model.py

	注：本程序所依赖的环境在requirements.txt文件中，安装环境过程中torch-cluster，torch-geometric==1.2.1,torch-scatter==1.2.0，torch-sparse==0.4.0可能会遇到安装错误，可通过降低版本来进行尝试。
