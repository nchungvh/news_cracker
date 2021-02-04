1. Gen input format (csv file) from json graph: merge-G.json:
	python gen_csv.py
2. Train model:
	python Media-LSTM-pytorch.py
3. Gen top-40 from trained model:
	python gen_top40.py
