PYTHON := python3
AGDA := agda

.PHONY: test run-02 run-03 run-04 run-05 run-06 run-07 run-08 run-09 run-10 run-11 run-12 run-13 run-14 run-15 run-16 run-17 run-18 run-20 run-21 run-22 run-23 run-24 run-25 run-26 run-27 run-28 run-29 run-30 agda-check agda-check-02 agda-check-03 agda-check-04 agda-check-05 agda-check-06 agda-check-07 agda-check-08 agda-check-09 agda-check-10 agda-check-11 agda-check-12 agda-check-13 agda-check-14 agda-check-15 agda-check-16 agda-check-17 agda-check-18 agda-check-20 agda-check-21 agda-check-22 agda-check-23 agda-check-24 agda-check-25 agda-check-26 agda-check-27 agda-check-28 agda-check-29 agda-check-30

test:
	$(PYTHON) -m pytest papers/02_char_rnn_karpathy/tests/test_char_rnn.py papers/03_lstm_understanding/tests/test_lstm.py papers/04_rnn_regularization/tests/test_rnn_regularization.py papers/05_neural_network_pruning/tests/test_pruning.py papers/06_pointer_networks/tests/test_pointer_networks.py papers/07_alexnet_cnn/tests/test_alexnet.py papers/08_seq2seq_for_sets/tests/test_seq2seq_sets.py papers/09_gpipe/tests/test_gpipe.py papers/10_resnet_deep_residual/tests/test_resnet.py papers/11_dilated_convolutions/tests/test_dilated_convolutions.py papers/12_graph_neural_networks/tests/test_gnn.py papers/13_attention_is_all_you_need/tests/test_transformer.py papers/14_bahdanau_attention/tests/test_bahdanau_attention.py papers/15_identity_mappings_resnet/tests/test_identity_mappings.py papers/16_relational_reasoning/tests/test_relational_reasoning.py papers/17_variational_autoencoder/tests/test_vae.py papers/18_relational_rnn/tests/test_relational_rnn.py papers/20_neural_turing_machine/tests/test_ntm.py papers/21_ctc_speech/tests/test_ctc.py papers/22_scaling_laws/tests/test_scaling_laws.py papers/23_mdl_principle/tests/test_mdl.py papers/24_machine_super_intelligence/tests/test_machine_super_intelligence.py papers/25_kolmogorov_complexity/tests/test_kolmogorov.py papers/26_cs231n_cnn_fundamentals/tests/test_cnn.py papers/27_multi_token_prediction/tests/test_multi_token_prediction.py papers/28_dense_passage_retrieval/tests/test_dense_passage_retrieval.py papers/29_rag/tests/test_rag.py papers/30_lost_in_middle/tests/test_lost_in_middle.py -q

run-02:
	$(PYTHON) scripts/run_paper.py --paper 02

run-03:
	$(PYTHON) scripts/run_paper.py --paper 03

run-04:
	$(PYTHON) scripts/run_paper.py --paper 04

run-05:
	$(PYTHON) scripts/run_paper.py --paper 05

run-06:
	$(PYTHON) scripts/run_paper.py --paper 06

run-07:
	$(PYTHON) scripts/run_paper.py --paper 07

run-08:
	$(PYTHON) scripts/run_paper.py --paper 08

run-09:
	$(PYTHON) scripts/run_paper.py --paper 09

run-10:
	$(PYTHON) scripts/run_paper.py --paper 10

run-11:
	$(PYTHON) scripts/run_paper.py --paper 11

run-12:
	$(PYTHON) scripts/run_paper.py --paper 12

run-13:
	$(PYTHON) scripts/run_paper.py --paper 13

run-14:
	$(PYTHON) scripts/run_paper.py --paper 14

run-15:
	$(PYTHON) scripts/run_paper.py --paper 15

run-16:
	$(PYTHON) scripts/run_paper.py --paper 16

run-17:
	$(PYTHON) scripts/run_paper.py --paper 17

run-18:
	$(PYTHON) scripts/run_paper.py --paper 18

run-20:
	$(PYTHON) scripts/run_paper.py --paper 20

run-21:
	$(PYTHON) scripts/run_paper.py --paper 21

run-22:
	$(PYTHON) scripts/run_paper.py --paper 22

run-23:
	$(PYTHON) scripts/run_paper.py --paper 23

run-24:
	$(PYTHON) scripts/run_paper.py --paper 24

run-25:
	$(PYTHON) scripts/run_paper.py --paper 25

run-26:
	$(PYTHON) scripts/run_paper.py --paper 26

run-27:
	$(PYTHON) scripts/run_paper.py --paper 27

run-28:
	$(PYTHON) scripts/run_paper.py --paper 28

run-29:
	$(PYTHON) scripts/run_paper.py --paper 29

run-30:
	$(PYTHON) scripts/run_paper.py --paper 30

agda-check: agda-check-02 agda-check-03 agda-check-04 agda-check-05 agda-check-06 agda-check-07 agda-check-08 agda-check-09 agda-check-10 agda-check-11 agda-check-12 agda-check-13 agda-check-14 agda-check-15 agda-check-16 agda-check-17 agda-check-18 agda-check-20 agda-check-21 agda-check-22 agda-check-23 agda-check-24 agda-check-25 agda-check-26 agda-check-27 agda-check-28 agda-check-29 agda-check-30

agda-check-02:
	$(AGDA) -i papers/02_char_rnn_karpathy/cubical-agda -i . papers/02_char_rnn_karpathy/cubical-agda/CharRNN.agda

agda-check-03:
	$(AGDA) -i papers/03_lstm_understanding/cubical-agda -i . papers/03_lstm_understanding/cubical-agda/LSTM.agda

agda-check-04:
	$(AGDA) -i papers/04_rnn_regularization/cubical-agda -i . papers/04_rnn_regularization/cubical-agda/RNNRegularization.agda

agda-check-05:
	$(AGDA) -i papers/05_neural_network_pruning/cubical-agda -i . papers/05_neural_network_pruning/cubical-agda/Pruning.agda

agda-check-06:
	$(AGDA) -i papers/06_pointer_networks/cubical-agda -i . papers/06_pointer_networks/cubical-agda/PointerNetworks.agda

agda-check-07:
	$(AGDA) -i papers/07_alexnet_cnn/cubical-agda -i . papers/07_alexnet_cnn/cubical-agda/AlexNet.agda

agda-check-08:
	$(AGDA) -i papers/08_seq2seq_for_sets/cubical-agda -i . papers/08_seq2seq_for_sets/cubical-agda/Seq2SeqForSets.agda

agda-check-09:
	$(AGDA) -i papers/09_gpipe/cubical-agda -i . papers/09_gpipe/cubical-agda/GPipe.agda

agda-check-10:
	$(AGDA) -i papers/10_resnet_deep_residual/cubical-agda -i . papers/10_resnet_deep_residual/cubical-agda/ResNet.agda

agda-check-11:
	$(AGDA) -i papers/11_dilated_convolutions/cubical-agda -i . papers/11_dilated_convolutions/cubical-agda/DilatedConvolutions.agda

agda-check-12:
	$(AGDA) -i papers/12_graph_neural_networks/cubical-agda -i . papers/12_graph_neural_networks/cubical-agda/GraphNeuralNetworks.agda

agda-check-13:
	$(AGDA) -i papers/13_attention_is_all_you_need/cubical-agda -i . papers/13_attention_is_all_you_need/cubical-agda/Transformer.agda

agda-check-14:
	$(AGDA) -i papers/14_bahdanau_attention/cubical-agda -i . papers/14_bahdanau_attention/cubical-agda/BahdanauAttention.agda

agda-check-15:
	$(AGDA) -i papers/15_identity_mappings_resnet/cubical-agda -i . papers/15_identity_mappings_resnet/cubical-agda/IdentityMappings.agda

agda-check-16:
	$(AGDA) -i papers/16_relational_reasoning/cubical-agda -i . papers/16_relational_reasoning/cubical-agda/RelationalReasoning.agda

agda-check-17:
	$(AGDA) -i papers/17_variational_autoencoder/cubical-agda -i . papers/17_variational_autoencoder/cubical-agda/VAE.agda

agda-check-18:
	$(AGDA) -i papers/18_relational_rnn/cubical-agda -i . papers/18_relational_rnn/cubical-agda/RelationalRNN.agda

agda-check-20:
	$(AGDA) -i papers/20_neural_turing_machine/cubical-agda -i . papers/20_neural_turing_machine/cubical-agda/NeuralTuringMachine.agda

agda-check-21:
	$(AGDA) -i papers/21_ctc_speech/cubical-agda -i . papers/21_ctc_speech/cubical-agda/CTC.agda

agda-check-22:
	$(AGDA) -i papers/22_scaling_laws/cubical-agda -i . papers/22_scaling_laws/cubical-agda/ScalingLaws.agda

agda-check-23:
	$(AGDA) -i papers/23_mdl_principle/cubical-agda -i . papers/23_mdl_principle/cubical-agda/MDL.agda

agda-check-24:
	$(AGDA) -i papers/24_machine_super_intelligence/cubical-agda -i . papers/24_machine_super_intelligence/cubical-agda/MachineSuperIntelligence.agda

agda-check-25:
	$(AGDA) -i papers/25_kolmogorov_complexity/cubical-agda -i . papers/25_kolmogorov_complexity/cubical-agda/KolmogorovComplexity.agda

agda-check-26:
	$(AGDA) -i papers/26_cs231n_cnn_fundamentals/cubical-agda -i . papers/26_cs231n_cnn_fundamentals/cubical-agda/CNN.agda

agda-check-27:
	$(AGDA) -i papers/27_multi_token_prediction/cubical-agda -i . papers/27_multi_token_prediction/cubical-agda/MultiTokenPrediction.agda

agda-check-28:
	$(AGDA) -i papers/28_dense_passage_retrieval/cubical-agda -i . papers/28_dense_passage_retrieval/cubical-agda/DensePassageRetrieval.agda

agda-check-29:
	$(AGDA) -i papers/29_rag/cubical-agda -i . papers/29_rag/cubical-agda/RAG.agda

agda-check-30:
	$(AGDA) -i papers/30_lost_in_middle/cubical-agda -i . papers/30_lost_in_middle/cubical-agda/LostInMiddle.agda
