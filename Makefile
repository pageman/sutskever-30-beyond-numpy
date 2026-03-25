PYTHON := python3
AGDA := agda

.PHONY: test run-02 run-03 run-26 agda-check agda-check-02 agda-check-03 agda-check-26

test:
	$(PYTHON) -m pytest papers/02_char_rnn_karpathy/tests/test_char_rnn.py papers/03_lstm_understanding/tests/test_lstm.py papers/26_cs231n_cnn_fundamentals/tests/test_cnn.py -q

run-02:
	$(PYTHON) scripts/run_paper.py --paper 02

run-03:
	$(PYTHON) scripts/run_paper.py --paper 03

run-26:
	$(PYTHON) scripts/run_paper.py --paper 26

agda-check: agda-check-02 agda-check-03 agda-check-26

agda-check-02:
	$(AGDA) -i . papers/02_char_rnn_karpathy/cubical-agda/CharRNN.agda

agda-check-03:
	$(AGDA) -i . papers/03_lstm_understanding/cubical-agda/LSTM.agda

agda-check-26:
	$(AGDA) -i . papers/26_cs231n_cnn_fundamentals/cubical-agda/CNN.agda

