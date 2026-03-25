#!/usr/bin/env python3
"""Run tiny demos for the populated papers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper02_char_rnn import (
    CharRNNConfig,
    build_dataset as build_char_dataset,
    forward_numpy as char_forward_numpy,
    init_params as init_char_params,
)
from s30bn.paper01_complexodynamics import (
    ComplexodynamicsConfig,
    forward_numpy as complexodynamics_forward_numpy,
    init_params as init_complexodynamics_params,
    synthetic_complexodynamics_batch,
)
from s30bn.paper03_lstm import (
    LSTMConfig,
    build_dataset as build_lstm_dataset,
    forward_numpy as lstm_forward_numpy,
    init_params as init_lstm_params,
)
from s30bn.paper04_rnn_regularization import (
    RNNRegularizationConfig,
    build_dataset as build_rnnreg_dataset,
    forward_numpy as rnnreg_forward_numpy,
    init_params as init_rnnreg_params,
)
from s30bn.paper05_pruning import PruningConfig, forward_numpy as pruning_forward_numpy, init_params as init_pruning_params, synthetic_pruning_batch
from s30bn.paper06_pointer_networks import PointerConfig, forward_numpy as pointer_forward_numpy, init_params as init_pointer_params, synthetic_pointer_batch
from s30bn.paper07_alexnet import AlexNetConfig, init_params as init_alexnet_params, synthetic_imagenet_like
from s30bn.paper08_seq2seq_sets import SetSeqConfig, forward_numpy as setseq_forward_numpy, init_params as init_setseq_params, synthetic_set_batch
from s30bn.paper10_resnet import ResNetConfig, init_params as init_resnet_params, synthetic_residual_dataset
from s30bn.paper11_dilated_convolutions import (
    DilatedConvConfig,
    init_params as init_dilated_params,
    synthetic_context_dataset,
)
from s30bn.paper12_gnn import GNNConfig, init_params as init_gnn_params, synthetic_graph_dataset
from s30bn.paper13_transformer import TransformerConfig, forward_numpy as transformer_forward_numpy, init_params as init_transformer_params, synthetic_attention_batch
from s30bn.paper14_bahdanau_attention import (
    BahdanauConfig,
    forward_numpy as bahdanau_forward_numpy,
    init_params as init_bahdanau_params,
    sample_pair as sample_bahdanau_pair,
)
from s30bn.paper17_vae import VAEConfig, forward_numpy as vae_forward_numpy, init_params as init_vae_params, synthetic_binary_data
from s30bn.paper15_identity_mappings import (
    IdentityResNetConfig,
    init_params as init_identity_params,
    synthetic_identity_dataset,
)
from s30bn.paper16_relational_reasoning import (
    RelationConfig,
    init_params as init_relation_params,
    synthetic_relation_dataset,
)
from s30bn.paper18_relational_rnn import (
    RelationalRNNConfig,
    build_dataset as build_relational_rnn_dataset,
    forward_numpy as relational_rnn_forward_numpy,
    init_params as init_relational_rnn_params,
)
from s30bn.paper19_coffee_automaton import (
    CoffeeConfig,
    forward_numpy as coffee_forward_numpy,
    init_params as init_coffee_params,
    synthetic_coffee_batch,
)
from s30bn.paper09_gpipe import GPipeConfig, forward_numpy as gpipe_forward_numpy, init_params as init_gpipe_params, synthetic_pipeline_batch
from s30bn.paper20_neural_turing_machine import (
    NTMConfig,
    build_dataset as build_ntm_dataset,
    forward_numpy as ntm_forward_numpy,
    init_params as init_ntm_params,
)
from s30bn.paper21_ctc import CTCConfig, forward_numpy as ctc_forward_numpy, init_params as init_ctc_params, synthetic_ctc_batch
from s30bn.paper22_scaling_laws import forward_numpy as scaling_forward_numpy, init_params as init_scaling_params, synthetic_scaling_data
from s30bn.paper23_mdl import MDLConfig, forward_numpy as mdl_forward_numpy, init_params as init_mdl_params, synthetic_mdl_batch
from s30bn.paper24_machine_super_intelligence import (
    SuperIntelligenceConfig,
    forward_numpy as superint_forward_numpy,
    init_params as init_superint_params,
    synthetic_superintelligence_batch,
)
from s30bn.paper25_kolmogorov import (
    KolmogorovConfig,
    forward_numpy as kolmogorov_forward_numpy,
    init_params as init_kolmogorov_params,
    synthetic_kolmogorov_batch,
)
from s30bn.paper27_multi_token_prediction import MultiTokenConfig, forward_numpy as mtp_forward_numpy, init_params as init_mtp_params, synthetic_mtp_batch
from s30bn.paper28_dense_passage_retrieval import DPRConfig, forward_numpy as dpr_forward_numpy, init_params as init_dpr_params, synthetic_dpr_batch
from s30bn.paper29_rag import RAGConfig, forward_numpy as rag_forward_numpy, init_params as init_rag_params, synthetic_rag_batch
from s30bn.paper30_lost_in_middle import LostMiddleConfig, forward_numpy as lost_middle_forward_numpy, init_params as init_lost_middle_params, synthetic_middle_batch
from s30bn.paper26_cs231n import CNNConfig, init_params as init_cnn_params, synthetic_cifar_like


def run_02() -> None:
    config = CharRNNConfig()
    inputs, targets = build_char_dataset(config)
    result = char_forward_numpy(init_char_params(config), inputs, targets)
    print(f"paper 02 numpy loss: {result['loss']:.6f}")


def run_01() -> None:
    config = ComplexodynamicsConfig()
    states, targets = synthetic_complexodynamics_batch()
    result = complexodynamics_forward_numpy(init_complexodynamics_params(config), states, targets)
    print(f"paper 01 numpy loss: {result['loss']:.6f}")


def run_03() -> None:
    config = LSTMConfig()
    inputs, targets = build_lstm_dataset(config)
    result = lstm_forward_numpy(init_lstm_params(config), inputs, targets)
    print(f"paper 03 numpy loss: {result['loss']:.6f}")


def run_04() -> None:
    config = RNNRegularizationConfig()
    inputs, targets = build_rnnreg_dataset(config)
    result = rnnreg_forward_numpy(init_rnnreg_params(config), inputs, targets, config)
    print(f"paper 04 numpy loss: {result['loss']:.6f}")


def run_05() -> None:
    config = PruningConfig()
    inputs, targets = synthetic_pruning_batch()
    result = pruning_forward_numpy(init_pruning_params(config), inputs, targets)
    print(f"paper 05 numpy loss: {result['loss']:.6f}")


def run_06() -> None:
    config = PointerConfig()
    seqs, targets = synthetic_pointer_batch()
    result = pointer_forward_numpy(init_pointer_params(config), seqs, targets)
    print(f"paper 06 numpy loss: {result['loss']:.6f}")


def run_07() -> None:
    config = AlexNetConfig()
    images, labels = synthetic_imagenet_like(config)
    params = init_alexnet_params(config)
    print(f"paper 07 synthetic dataset shape: {images.shape}, labels: {labels.tolist()}")
    print(f"paper 07 conv1 kernel shape: {params['conv1_w'].shape}")


def run_08() -> None:
    config = SetSeqConfig()
    sets, targets = synthetic_set_batch()
    result = setseq_forward_numpy(init_setseq_params(config), sets, targets)
    print(f"paper 08 numpy loss: {result['loss']:.6f}")


def run_09() -> None:
    config = GPipeConfig()
    batch, labels = synthetic_pipeline_batch(config)
    result = gpipe_forward_numpy(init_gpipe_params(config), batch, labels, config)
    print(f"paper 09 numpy loss: {result['loss']:.6f}")


def run_10() -> None:
    config = ResNetConfig()
    images, labels = synthetic_residual_dataset(config)
    params = init_resnet_params(config)
    print(f"paper 10 synthetic dataset shape: {images.shape}, labels: {labels.tolist()}")
    print(f"paper 10 stem kernel shape: {params['stem_w'].shape}")


def run_11() -> None:
    config = DilatedConvConfig()
    images, labels = synthetic_context_dataset(config)
    params = init_dilated_params(config)
    print(f"paper 11 synthetic dataset shape: {images.shape}, labels: {labels.tolist()}")
    print(f"paper 11 dilated kernel base shape: {params['dilated_w'].shape}")


def run_12() -> None:
    config = GNNConfig()
    graphs, adjacency, labels = synthetic_graph_dataset(config)
    params = init_gnn_params(config)
    print(f"paper 12 graphs shape: {graphs.shape}, labels: {labels.tolist()}")
    print(f"paper 12 message weight shape: {params['W_msg'].shape}")
    print(f"paper 12 adjacency shape: {adjacency.shape}")


def run_13() -> None:
    config = TransformerConfig()
    seqs, targets = synthetic_attention_batch()
    result = transformer_forward_numpy(init_transformer_params(config), seqs, targets)
    print(f"paper 13 numpy loss: {result['loss']:.6f}")


def run_14() -> None:
    config = BahdanauConfig()
    source, decoder_token, target = sample_bahdanau_pair(config)
    result = bahdanau_forward_numpy(init_bahdanau_params(config), source, decoder_token, target, config)
    print(f"paper 14 numpy loss: {result['loss']:.6f}")


def run_15() -> None:
    config = IdentityResNetConfig()
    images, labels = synthetic_identity_dataset(config)
    params = init_identity_params(config)
    print(f"paper 15 synthetic dataset shape: {images.shape}, labels: {labels.tolist()}")
    print(f"paper 15 residual kernel shape: {params['conv1_w'].shape}")


def run_16() -> None:
    config = RelationConfig()
    scenes, labels = synthetic_relation_dataset(config)
    params = init_relation_params(config)
    print(f"paper 16 scenes shape: {scenes.shape}, labels: {labels.tolist()}")
    print(f"paper 16 relation head shape: {params['f_w'].shape}")


def run_17() -> None:
    config = VAEConfig()
    data = synthetic_binary_data()
    result = vae_forward_numpy(init_vae_params(config), data, config)
    print(f"paper 17 numpy loss: {result['loss']:.6f}")


def run_18() -> None:
    config = RelationalRNNConfig()
    inputs, targets = build_relational_rnn_dataset(config)
    result = relational_rnn_forward_numpy(init_relational_rnn_params(config), inputs, targets, config)
    print(f"paper 18 numpy loss: {result['loss']:.6f}")


def run_19() -> None:
    config = CoffeeConfig()
    states, targets = synthetic_coffee_batch()
    result = coffee_forward_numpy(init_coffee_params(config), states, targets)
    print(f"paper 19 numpy loss: {result['loss']:.6f}")


def run_20() -> None:
    config = NTMConfig()
    inputs, targets = build_ntm_dataset(config)
    result = ntm_forward_numpy(init_ntm_params(config), inputs, targets, config)
    print(f"paper 20 numpy loss: {result['loss']:.6f}")


def run_21() -> None:
    config = CTCConfig()
    features, targets = synthetic_ctc_batch()
    result = ctc_forward_numpy(init_ctc_params(config), features, targets, config)
    print(f"paper 21 numpy loss: {result['loss']:.6f}")


def run_22() -> None:
    log_n, log_loss = synthetic_scaling_data()
    result = scaling_forward_numpy(init_scaling_params(), log_n, log_loss)
    print(f"paper 22 numpy loss: {result['loss']:.6f}")


def run_23() -> None:
    config = MDLConfig()
    inputs, targets = synthetic_mdl_batch()
    result = mdl_forward_numpy(init_mdl_params(config), inputs, targets, config)
    print(f"paper 23 numpy loss: {result['loss']:.6f}")


def run_24() -> None:
    config = SuperIntelligenceConfig()
    capabilities, targets = synthetic_superintelligence_batch()
    result = superint_forward_numpy(init_superint_params(config), capabilities, targets)
    print(f"paper 24 numpy loss: {result['loss']:.6f}")


def run_25() -> None:
    config = KolmogorovConfig()
    sequences, targets = synthetic_kolmogorov_batch()
    result = kolmogorov_forward_numpy(init_kolmogorov_params(config), sequences, targets)
    print(f"paper 25 numpy loss: {result['loss']:.6f}")


def run_26() -> None:
    config = CNNConfig()
    images, labels = synthetic_cifar_like(config)
    params = init_cnn_params(config)
    print(f"paper 26 synthetic dataset shape: {images.shape}, labels: {labels.tolist()}")
    print(f"paper 26 conv kernel shape: {params['conv_w'].shape}")


def run_27() -> None:
    config = MultiTokenConfig()
    contexts, targets = synthetic_mtp_batch()
    result = mtp_forward_numpy(init_mtp_params(config), contexts, targets)
    print(f"paper 27 numpy loss: {result['loss']:.6f}")


def run_28() -> None:
    config = DPRConfig()
    queries, passages, targets = synthetic_dpr_batch()
    result = dpr_forward_numpy(init_dpr_params(config), queries, passages, targets)
    print(f"paper 28 numpy loss: {result['loss']:.6f}")


def run_29() -> None:
    config = RAGConfig()
    queries, passages, targets = synthetic_rag_batch()
    result = rag_forward_numpy(init_rag_params(config), queries, passages, targets)
    print(f"paper 29 numpy loss: {result['loss']:.6f}")


def run_30() -> None:
    config = LostMiddleConfig()
    chunks, queries, targets = synthetic_middle_batch()
    result = lost_middle_forward_numpy(init_lost_middle_params(config), chunks, queries, targets)
    print(f"paper 30 numpy loss: {result['loss']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True, choices=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"])
    args = parser.parse_args()
    {
        "01": run_01,
        "02": run_02,
        "03": run_03,
        "04": run_04,
        "05": run_05,
        "06": run_06,
        "07": run_07,
        "08": run_08,
        "09": run_09,
        "10": run_10,
        "11": run_11,
        "12": run_12,
        "13": run_13,
        "14": run_14,
        "15": run_15,
        "16": run_16,
        "17": run_17,
        "18": run_18,
        "19": run_19,
        "20": run_20,
        "21": run_21,
        "22": run_22,
        "23": run_23,
        "24": run_24,
        "25": run_25,
        "26": run_26,
        "27": run_27,
        "28": run_28,
        "29": run_29,
        "30": run_30,
    }[args.paper]()


if __name__ == "__main__":
    main()
