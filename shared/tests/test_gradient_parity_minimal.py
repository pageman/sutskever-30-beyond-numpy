from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import numpy as np

from s30bn.test_support import assert_gradient_slice_close

from s30bn import paper01_complexodynamics as p01
from s30bn import paper02_char_rnn as p02
from s30bn import paper03_lstm as p03
from s30bn import paper04_rnn_regularization as p04
from s30bn import paper05_pruning as p05
from s30bn import paper06_pointer_networks as p06
from s30bn import paper07_alexnet as p07
from s30bn import paper08_seq2seq_sets as p08
from s30bn import paper09_gpipe as p09
from s30bn import paper10_resnet as p10
from s30bn import paper11_dilated_convolutions as p11
from s30bn import paper12_gnn as p12
from s30bn import paper13_transformer as p13
from s30bn import paper14_bahdanau_attention as p14
from s30bn import paper15_identity_mappings as p15
from s30bn import paper16_relational_reasoning as p16
from s30bn import paper17_vae as p17
from s30bn import paper18_relational_rnn as p18
from s30bn import paper19_coffee_automaton as p19
from s30bn import paper20_neural_turing_machine as p20
from s30bn import paper21_ctc as p21
from s30bn import paper22_scaling_laws as p22
from s30bn import paper23_mdl as p23
from s30bn import paper24_machine_super_intelligence as p24
from s30bn import paper25_kolmogorov as p25
from s30bn import paper26_cs231n as p26
from s30bn import paper27_multi_token_prediction as p27
from s30bn import paper28_dense_passage_retrieval as p28
from s30bn import paper29_rag as p29
from s30bn import paper30_lost_in_middle as p30


Builder = Callable[[], tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]]


@dataclass(frozen=True)
class GradientSpec:
    paper_id: str
    build: Builder


def _slice_gradient(array: Any, width: int = 8) -> np.ndarray:
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    return flat[: min(width, flat.size)]


def _build_01() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p01.ComplexodynamicsConfig()
    params = p01.init_params(config)
    states, targets = p01.synthetic_complexodynamics_batch()
    return params, lambda current: p01.forward_torch(current, states, targets)[0], lambda current: p01.forward_jax(current, states, targets)[0]


def _build_02() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p02.CharRNNConfig()
    inputs, targets = p02.build_dataset(config)
    params = p02.init_params(config)
    return params, lambda current: p02.forward_torch(current, inputs, targets)[0], lambda current: p02.forward_jax(current, inputs, targets)[0]


def _build_03() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p03.LSTMConfig()
    inputs, targets = p03.build_dataset(config)
    params = p03.init_params(config)
    return params, lambda current: p03.forward_torch(current, inputs, targets)[0], lambda current: p03.forward_jax(current, inputs, targets)[0]


def _build_04() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p04.RNNRegularizationConfig()
    inputs, targets = p04.build_dataset(config)
    params = p04.init_params(config)
    return params, lambda current: p04.forward_torch(current, inputs, targets, config)[0], lambda current: p04.forward_jax(current, inputs, targets, config)[0]


def _build_05() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p05.PruningConfig()
    inputs, targets = p05.synthetic_pruning_batch()
    params = p05.init_params(config)
    return params, lambda current: p05.forward_torch(current, inputs, targets)[0], lambda current: p05.forward_jax(current, inputs, targets)[0]


def _build_06() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p06.PointerConfig()
    seqs, targets = p06.synthetic_pointer_batch()
    params = p06.init_params(config)
    return params, lambda current: p06.forward_torch(current, seqs, targets)[0], lambda current: p06.forward_jax(current, seqs, targets)[0]


def _build_07() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p07.AlexNetConfig()
    images, labels = p07.synthetic_imagenet_like(config)
    params = p07.init_params(config)
    return params, lambda current: p07.forward_torch(current, images, labels)[0], lambda current: p07.forward_jax(current, images, labels)[0]


def _build_08() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p08.SetSeqConfig()
    sets, targets = p08.synthetic_set_batch()
    params = p08.init_params(config)
    return params, lambda current: p08.forward_torch(current, sets, targets)[0], lambda current: p08.forward_jax(current, sets, targets)[0]


def _build_09() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p09.GPipeConfig()
    batch, labels = p09.synthetic_pipeline_batch(config)
    params = p09.init_params(config)
    return params, lambda current: p09.forward_torch(current, batch, labels, config)[0], lambda current: p09.forward_jax(current, batch, labels, config)[0]


def _build_10() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p10.ResNetConfig()
    images, labels = p10.synthetic_residual_dataset(config)
    params = p10.init_params(config)
    return params, lambda current: p10.forward_torch(current, images, labels)[0], lambda current: p10.forward_jax(current, images, labels)[0]


def _build_11() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p11.DilatedConvConfig()
    images, labels = p11.synthetic_context_dataset(config)
    params = p11.init_params(config)
    return params, lambda current: p11.forward_torch(current, images, labels, config)[0], lambda current: p11.forward_jax(current, images, labels, config)[0]


def _build_12() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p12.GNNConfig()
    graphs, adjacency, labels = p12.synthetic_graph_dataset(config)
    params = p12.init_params(config)
    return params, lambda current: p12.forward_torch(current, graphs, adjacency, labels)[0], lambda current: p12.forward_jax(current, graphs, adjacency, labels)[0]


def _build_13() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p13.TransformerConfig()
    seqs, targets = p13.synthetic_attention_batch()
    params = p13.init_params(config)
    return params, lambda current: p13.forward_torch(current, seqs, targets)[0], lambda current: p13.forward_jax(current, seqs, targets)[0]


def _build_14() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p14.BahdanauConfig()
    source, decoder_token, target = p14.sample_pair(config)
    params = p14.init_params(config)
    return params, lambda current: p14.forward_torch(current, source, decoder_token, target, config)[0], lambda current: p14.forward_jax(current, source, decoder_token, target, config)[0]


def _build_15() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p15.IdentityResNetConfig()
    images, labels = p15.synthetic_identity_dataset(config)
    params = p15.init_params(config)
    return params, lambda current: p15.forward_torch(current, images, labels)[0], lambda current: p15.forward_jax(current, images, labels)[0]


def _build_16() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p16.RelationConfig()
    scenes, labels = p16.synthetic_relation_dataset(config)
    params = p16.init_params(config)
    return params, lambda current: p16.forward_torch(current, scenes, labels, config)[0], lambda current: p16.forward_jax(current, scenes, labels, config)[0]


def _build_17() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p17.VAEConfig()
    data = p17.synthetic_binary_data()
    params = p17.init_params(config)
    return params, lambda current: p17.forward_torch(current, data, config)[0], lambda current: p17.forward_jax(current, data, config)[0]


def _build_18() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p18.RelationalRNNConfig()
    inputs, targets = p18.build_dataset(config)
    params = p18.init_params(config)
    return params, lambda current: p18.forward_torch(current, inputs, targets, config)[0], lambda current: p18.forward_jax(current, inputs, targets, config)[0]


def _build_19() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p19.CoffeeConfig()
    states, targets = p19.synthetic_coffee_batch()
    params = p19.init_params(config)
    return params, lambda current: p19.forward_torch(current, states, targets)[0], lambda current: p19.forward_jax(current, states, targets)[0]


def _build_20() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p20.NTMConfig()
    inputs, targets = p20.build_dataset(config)
    params = p20.init_params(config)
    return params, lambda current: p20.forward_torch(current, inputs, targets, config)[0], lambda current: p20.forward_jax(current, inputs, targets, config)[0]


def _build_21() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p21.CTCConfig()
    features, targets = p21.synthetic_ctc_batch()
    params = p21.init_params(config)
    return params, lambda current: p21.forward_torch(current, features, targets, config)[0], lambda current: p21.forward_jax(current, features, targets, config)[0]


def _build_22() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    log_n, log_loss = p22.synthetic_scaling_data()
    params = p22.init_params()
    return params, lambda current: p22.forward_torch(current, log_n, log_loss)[0], lambda current: p22.forward_jax(current, log_n, log_loss)[0]


def _build_23() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p23.MDLConfig()
    description_length, errors = p23.synthetic_mdl_batch()
    params = p23.init_params(config)
    return params, lambda current: p23.forward_torch(current, description_length, errors, config)[0], lambda current: p23.forward_jax(current, description_length, errors, config)[0]


def _build_24() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p24.SuperIntelligenceConfig()
    capabilities, targets = p24.synthetic_superintelligence_batch()
    params = p24.init_params(config)
    return params, lambda current: p24.forward_torch(current, capabilities, targets)[0], lambda current: p24.forward_jax(current, capabilities, targets)[0]


def _build_25() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p25.KolmogorovConfig()
    lengths, targets = p25.synthetic_kolmogorov_batch()
    params = p25.init_params(config)
    return params, lambda current: p25.forward_torch(current, lengths, targets)[0], lambda current: p25.forward_jax(current, lengths, targets)[0]


def _build_26() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p26.CNNConfig()
    images, labels = p26.synthetic_cifar_like(config)
    params = p26.init_params(config)
    return params, lambda current: p26.forward_torch(current, images, labels)[0], lambda current: p26.forward_jax(current, images, labels)[0]


def _build_27() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p27.MultiTokenConfig()
    contexts, targets = p27.synthetic_mtp_batch()
    params = p27.init_params(config)
    return params, lambda current: p27.forward_torch(current, contexts, targets)[0], lambda current: p27.forward_jax(current, contexts, targets)[0]


def _build_28() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p28.DPRConfig()
    queries, passages, targets = p28.synthetic_dpr_batch()
    params = p28.init_params(config)
    return params, lambda current: p28.forward_torch(current, queries, passages, targets)[0], lambda current: p28.forward_jax(current, queries, passages, targets)[0]


def _build_29() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p29.RAGConfig()
    queries, passages, targets = p29.synthetic_rag_batch()
    params = p29.init_params(config)
    return params, lambda current: p29.forward_torch(current, queries, passages, targets)[0], lambda current: p29.forward_jax(current, queries, passages, targets)[0]


def _build_30() -> tuple[dict[str, np.ndarray], Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    config = p30.LostMiddleConfig()
    chunks, queries, targets = p30.synthetic_middle_batch()
    params = p30.init_params(config)
    return params, lambda current: p30.forward_torch(current, chunks, queries, targets)[0], lambda current: p30.forward_jax(current, chunks, queries, targets)[0]


GRADIENT_SPECS = [
    GradientSpec("01", _build_01),
    GradientSpec("02", _build_02),
    GradientSpec("03", _build_03),
    GradientSpec("04", _build_04),
    GradientSpec("05", _build_05),
    GradientSpec("06", _build_06),
    GradientSpec("07", _build_07),
    GradientSpec("08", _build_08),
    GradientSpec("09", _build_09),
    GradientSpec("10", _build_10),
    GradientSpec("11", _build_11),
    GradientSpec("12", _build_12),
    GradientSpec("13", _build_13),
    GradientSpec("14", _build_14),
    GradientSpec("15", _build_15),
    GradientSpec("16", _build_16),
    GradientSpec("17", _build_17),
    GradientSpec("18", _build_18),
    GradientSpec("19", _build_19),
    GradientSpec("20", _build_20),
    GradientSpec("21", _build_21),
    GradientSpec("22", _build_22),
    GradientSpec("23", _build_23),
    GradientSpec("24", _build_24),
    GradientSpec("25", _build_25),
    GradientSpec("26", _build_26),
    GradientSpec("27", _build_27),
    GradientSpec("28", _build_28),
    GradientSpec("29", _build_29),
    GradientSpec("30", _build_30),
]


def test_minimal_torch_jax_gradient_parity_all_papers() -> None:
    for spec in GRADIENT_SPECS:
        params, torch_loss_fn, jax_loss_fn = spec.build()
        key = next(iter(params))

        torch_params = globals()[f"p{spec.paper_id}"].params_to_torch(params)
        torch_loss = torch_loss_fn(torch_params)
        torch_loss.backward()
        torch_grad = _slice_gradient(torch_params[key].grad.detach().cpu().numpy())

        jax_params = globals()[f"p{spec.paper_id}"].params_to_jax(params)
        jax_grads = jax.grad(jax_loss_fn)(jax_params)
        jax_grad = _slice_gradient(jax_grads[key])

        assert_gradient_slice_close(
            torch_grad,
            {f"paper {spec.paper_id} jax": jax_grad},
            atol=1e-6,
        )
