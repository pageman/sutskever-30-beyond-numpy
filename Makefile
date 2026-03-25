PYTHON := python3
AGDA := agda

.PHONY: test run-02 run-03 run-04 run-07 run-10 run-15 run-26 agda-check agda-check-02 agda-check-03 agda-check-04 agda-check-07 agda-check-10 agda-check-15 agda-check-26

test:
	$(PYTHON) -m pytest papers/02_char_rnn_karpathy/tests/test_char_rnn.py papers/03_lstm_understanding/tests/test_lstm.py papers/04_rnn_regularization/tests/test_rnn_regularization.py papers/07_alexnet_cnn/tests/test_alexnet.py papers/10_resnet_deep_residual/tests/test_resnet.py papers/15_identity_mappings_resnet/tests/test_identity_mappings.py papers/26_cs231n_cnn_fundamentals/tests/test_cnn.py -q

run-02:
	$(PYTHON) scripts/run_paper.py --paper 02

run-03:
	$(PYTHON) scripts/run_paper.py --paper 03

run-04:
	$(PYTHON) scripts/run_paper.py --paper 04

run-07:
	$(PYTHON) scripts/run_paper.py --paper 07

run-10:
	$(PYTHON) scripts/run_paper.py --paper 10

run-15:
	$(PYTHON) scripts/run_paper.py --paper 15

run-26:
	$(PYTHON) scripts/run_paper.py --paper 26

agda-check: agda-check-02 agda-check-03 agda-check-04 agda-check-07 agda-check-10 agda-check-15 agda-check-26

agda-check-02:
	$(AGDA) -i papers/02_char_rnn_karpathy/cubical-agda -i . papers/02_char_rnn_karpathy/cubical-agda/CharRNN.agda

agda-check-03:
	$(AGDA) -i papers/03_lstm_understanding/cubical-agda -i . papers/03_lstm_understanding/cubical-agda/LSTM.agda

agda-check-04:
	$(AGDA) -i papers/04_rnn_regularization/cubical-agda -i . papers/04_rnn_regularization/cubical-agda/RNNRegularization.agda

agda-check-07:
	$(AGDA) -i papers/07_alexnet_cnn/cubical-agda -i . papers/07_alexnet_cnn/cubical-agda/AlexNet.agda

agda-check-10:
	$(AGDA) -i papers/10_resnet_deep_residual/cubical-agda -i . papers/10_resnet_deep_residual/cubical-agda/ResNet.agda

agda-check-15:
	$(AGDA) -i papers/15_identity_mappings_resnet/cubical-agda -i . papers/15_identity_mappings_resnet/cubical-agda/IdentityMappings.agda

agda-check-26:
	$(AGDA) -i papers/26_cs231n_cnn_fundamentals/cubical-agda -i . papers/26_cs231n_cnn_fundamentals/cubical-agda/CNN.agda
