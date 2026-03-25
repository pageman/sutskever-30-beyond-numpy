PYTHON := python3
AGDA := agda

.PHONY: test run-02 run-03 run-04 run-07 run-10 run-11 run-14 run-15 run-16 run-18 run-20 run-26 agda-check agda-check-02 agda-check-03 agda-check-04 agda-check-07 agda-check-10 agda-check-11 agda-check-14 agda-check-15 agda-check-16 agda-check-18 agda-check-20 agda-check-26

test:
	$(PYTHON) -m pytest papers/02_char_rnn_karpathy/tests/test_char_rnn.py papers/03_lstm_understanding/tests/test_lstm.py papers/04_rnn_regularization/tests/test_rnn_regularization.py papers/07_alexnet_cnn/tests/test_alexnet.py papers/10_resnet_deep_residual/tests/test_resnet.py papers/11_dilated_convolutions/tests/test_dilated_convolutions.py papers/14_bahdanau_attention/tests/test_bahdanau_attention.py papers/15_identity_mappings_resnet/tests/test_identity_mappings.py papers/16_relational_reasoning/tests/test_relational_reasoning.py papers/18_relational_rnn/tests/test_relational_rnn.py papers/20_neural_turing_machine/tests/test_ntm.py papers/26_cs231n_cnn_fundamentals/tests/test_cnn.py -q

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

run-11:
	$(PYTHON) scripts/run_paper.py --paper 11

run-14:
	$(PYTHON) scripts/run_paper.py --paper 14

run-15:
	$(PYTHON) scripts/run_paper.py --paper 15

run-16:
	$(PYTHON) scripts/run_paper.py --paper 16

run-18:
	$(PYTHON) scripts/run_paper.py --paper 18

run-20:
	$(PYTHON) scripts/run_paper.py --paper 20

run-26:
	$(PYTHON) scripts/run_paper.py --paper 26

agda-check: agda-check-02 agda-check-03 agda-check-04 agda-check-07 agda-check-10 agda-check-11 agda-check-14 agda-check-15 agda-check-16 agda-check-18 agda-check-20 agda-check-26

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

agda-check-11:
	$(AGDA) -i papers/11_dilated_convolutions/cubical-agda -i . papers/11_dilated_convolutions/cubical-agda/DilatedConvolutions.agda

agda-check-14:
	$(AGDA) -i papers/14_bahdanau_attention/cubical-agda -i . papers/14_bahdanau_attention/cubical-agda/BahdanauAttention.agda

agda-check-15:
	$(AGDA) -i papers/15_identity_mappings_resnet/cubical-agda -i . papers/15_identity_mappings_resnet/cubical-agda/IdentityMappings.agda

agda-check-16:
	$(AGDA) -i papers/16_relational_reasoning/cubical-agda -i . papers/16_relational_reasoning/cubical-agda/RelationalReasoning.agda

agda-check-18:
	$(AGDA) -i papers/18_relational_rnn/cubical-agda -i . papers/18_relational_rnn/cubical-agda/RelationalRNN.agda

agda-check-20:
	$(AGDA) -i papers/20_neural_turing_machine/cubical-agda -i . papers/20_neural_turing_machine/cubical-agda/NeuralTuringMachine.agda

agda-check-26:
	$(AGDA) -i papers/26_cs231n_cnn_fundamentals/cubical-agda -i . papers/26_cs231n_cnn_fundamentals/cubical-agda/CNN.agda
