runTests:
	@echo "Model 1"
	@python3 train_miniplaces.py
	@python eval_miniplaces.py --load ./outputs/model_best.pth.tar | grep "Accuracy" >> results.txt

	@echo "Model 2"
	@python3 train_miniplaces.py --batch-size 8
	@python eval_miniplaces.py --load ./outputs/model_best.pth.tar | grep "Accuracy" >> results.txt

	@echo "Model 3"
	@python3 train_miniplaces.py --batch-size 16
	@python eval_miniplaces.py --load ./outputs/model_best.pth.tar | grep "Accuracy" >> results.txt

	@echo "Model 4"
	@python3 train_miniplaces.py --lr 0.05
	@python eval_miniplaces.py --load ./outputs/model_best.pth.tar | grep "Accuracy" >> results.txt

	@echo "Model 5"
	@python3 train_miniplaces.py --lr 0.01
	@python eval_miniplaces.py --load ./outputs/model_best.pth.tar | grep "Accuracy" >> results.txt

	@echo "Model 6"
	@python3 train_miniplaces.py --epochs 20
	@python eval_miniplaces.py --load ./outputs/model_best.pth.tar | grep "Accuracy" >> results.txt

	@echo "Model 7"
	@python3 train_miniplaces.py --epochs 5
	@python eval_miniplaces.py --load ./outputs/model_best.pth.tar | grep "Accuracy" >> results.txt

