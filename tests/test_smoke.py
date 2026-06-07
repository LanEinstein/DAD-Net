"""Forward, loss, and configuration sanity checks that need no dataset."""

from __future__ import annotations

import torch

from dadnet import build_dadnet, get_benchmark, get_dad_net, get_microflownext
from dadnet.configs import BENCHMARKS, parse_stage_mask
from dadnet.losses import gaussian_sliced_wasserstein_distance, get_kd_loss


def test_microflownext_forward():
    model = get_microflownext(num_classes=3, model_size="micro")
    x = torch.randn(2, 2, 224, 224)
    predictions, logits = model(x)
    assert logits.shape == (2, 3)
    assert predictions.shape == (2,)


def test_microflownext_ablations():
    model = get_microflownext(
        num_classes=3, model_size="micro",
        skip_stn=True, skip_channel_attention=True, skip_head_attention=True,
    )
    logits = model(torch.randn(2, 2, 224, 224))[1]
    assert logits.shape == (2, 3)


def test_dadnet_train_outputs():
    model = get_dad_net(num_classes=3, micro_model_size="micro", macro_model_size="micro",
                        alignment_stages=parse_stage_mask("FFTT"), alignment_weight=1.0)
    x = torch.randn(2, 2, 224, 224)
    y = torch.randint(0, 3, (2,))
    predictions, logits, _, loss, alignment = model(x, y)
    assert logits.shape == (2, 3)
    assert loss.requires_grad
    assert alignment.numel() == 1


def test_dadnet_adapter_for_width_mismatch():
    model = get_dad_net(num_classes=3, micro_model_size="base", macro_model_size="nano",
                        alignment_stages=parse_stage_mask("TFTT"), alignment_weight=1.0)
    assert model.adapter_map  # adapters created for mismatched stages
    loss = model(torch.randn(2, 2, 224, 224), torch.randint(0, 3, (2,)))[3]
    assert torch.isfinite(loss)


def test_gswd_zero_for_identical_inputs():
    feature = torch.randn(4, 64, 8, 8)
    value = gaussian_sliced_wasserstein_distance(feature, feature, num_projections=64)
    assert value.item() < 1e-3


def test_kd_baselines_dispatch():
    s = torch.randn(4, 96, 7, 7)
    t = torch.randn(4, 96, 7, 7)
    for method in ("fitnet", "l2", "mmd", "kl", "pkt"):
        assert torch.isfinite(get_kd_loss(method)(s, t))
    assert torch.isfinite(get_kd_loss("crd", student_dim=96, teacher_dim=96)(s, t))


def test_benchmark_configs_match_paper():
    expected = {
        "4dme": ("base", "TFTT", 1.5),
        "casme3": ("base", "TFFT", 1.0),
        "dfme_3class": ("small", "FFTT", 0.0),
        "dfme_4class": ("base", "FFTT", 0.0),
        "dfme_7class": ("micro", "FFTT", 1.0),
    }
    for name, (size, mask, lam) in expected.items():
        config = get_benchmark(name)
        assert config.student_size == size
        assert config.stage_mask == mask
        assert config.alignment_weight == lam


def test_builder_matches_config():
    config = get_benchmark("dfme_7class")
    model = build_dadnet(config)
    assert model.num_classes == 7
    assert model.micro_branch.skip_stn is True
    assert len(BENCHMARKS) == 5
