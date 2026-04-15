"""Tests for Legendre polynomial parameterized encoder."""

import torch
import pytest

from tabicl.model.legendre import legendre_basis, LegendreLinear, LegendreEncoder
from tabicl.model.encoders import Encoder
from tabicl.model.kv_cache import KVCache


class TestLegendreBasis:
    def test_shape(self):
        B = legendre_basis(24, 6)
        assert B.shape == (24, 6)

    def test_orthogonality(self):
        """Legendre polynomials should be approximately orthogonal on uniform grid."""
        n = 1000
        B = legendre_basis(n, 6)
        # Gram matrix: B^T B / n should approximate diagonal
        gram = B.T @ B / n
        off_diag = gram - torch.diag(gram.diag())
        assert off_diag.abs().max() < 0.05

    def test_boundary_values(self):
        """P_k(-1) and P_k(1) should match known values."""
        B = legendre_basis(2, 5)  # points at -1 and 1
        # P_k(1) = 1 for all k, P_k(-1) = (-1)^k
        for k in range(5):
            assert abs(B[1, k] - 1.0) < 1e-6, f"P_{k}(1) should be 1"
            assert abs(B[0, k] - (-1) ** k) < 1e-6, f"P_{k}(-1) should be {(-1)**k}"

    def test_single_point(self):
        """Single point should be at t=0."""
        B = legendre_basis(1, 4)
        assert B.shape == (1, 4)
        assert B[0, 0] == 1.0  # P_0(0) = 1
        assert abs(B[0, 1]) < 1e-6  # P_1(0) = 0


class TestLegendreLinear:
    def test_reconstruction_shape(self):
        ll = LegendreLinear(12, 6, 128, 64)
        W = ll.reconstruct_weights()
        assert W.shape == (12, 128, 64)

    def test_single_layer_matches_full(self):
        ll = LegendreLinear(12, 6, 128, 64)
        W_all = ll.reconstruct_weights()
        for i in range(12):
            W_i = ll.get_weight(i)
            assert torch.allclose(W_all[i], W_i, atol=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow through reconstruction to coefficients."""
        ll = LegendreLinear(8, 4, 32, 16)
        x = torch.randn(2, 5, 16)
        W = ll.reconstruct_weights()
        # Use first layer's weight
        out = torch.nn.functional.linear(x, W[0], ll.get_bias(0))
        loss = out.sum()
        loss.backward()
        assert ll.coeffs.grad is not None
        assert ll.coeffs.grad.abs().sum() > 0

    def test_no_bias(self):
        ll = LegendreLinear(8, 4, 32, 16, bias=False)
        assert ll.bias is None
        assert ll.get_bias(0) is None

    def test_layer_scales(self):
        """Per-layer scales should be applied to reconstructed weights."""
        ll = LegendreLinear(4, 2, 16, 8)
        with torch.no_grad():
            ll.layer_scales[0] = 2.0
            ll.layer_scales[1] = 0.5
        W = ll.reconstruct_weights()
        # With scale=2, layer 0 should be 2x the unscaled value
        ll.layer_scales.data.fill_(1.0)
        W_unscaled = ll.reconstruct_weights()
        assert torch.allclose(W[0], 2.0 * W_unscaled[0], atol=1e-5)

    def test_layer_scales_gradient(self):
        """Gradients should flow to layer_scales."""
        ll = LegendreLinear(4, 2, 16, 8)
        x = torch.randn(2, 5, 8)
        W = ll.reconstruct_weights()
        out = torch.nn.functional.linear(x, W[0])
        out.sum().backward()
        assert ll.layer_scales.grad is not None

    def test_order_decaying_init(self):
        """Higher-order coefficients should have smaller magnitude at init."""
        ll = LegendreLinear(8, 6, 64, 32)
        norms = [ll.coeffs[k].norm().item() for k in range(6)]
        # Each order should be roughly half the previous (with randomness)
        for k in range(1, 6):
            assert norms[k] < norms[k - 1], f"Order {k} norm ({norms[k]}) >= order {k-1} norm ({norms[k-1]})"


class TestLegendreEncoder:
    @pytest.fixture
    def encoder_kwargs(self):
        return dict(
            num_blocks=8,
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation="gelu",
            norm_first=True,
        )

    def test_forward_shape(self, encoder_kwargs):
        enc = LegendreEncoder(num_coeffs=4, **encoder_kwargs)
        x = torch.randn(2, 10, 64)
        out = enc(x)
        assert out.shape == (2, 10, 64)

    def test_forward_with_train_size(self, encoder_kwargs):
        enc = LegendreEncoder(num_coeffs=4, **encoder_kwargs)
        x = torch.randn(2, 10, 64)
        out = enc(x, train_size=7)
        assert out.shape == (2, 10, 64)

    def test_gradient_flow_to_coefficients(self, encoder_kwargs):
        enc = LegendreEncoder(num_coeffs=4, **encoder_kwargs)
        x = torch.randn(2, 10, 64)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        # Verify gradients flow to all coefficient groups
        assert enc.attn_in_proj.coeffs.grad is not None
        assert enc.attn_out_proj.coeffs.grad is not None
        assert enc.ff1.coeffs.grad is not None
        assert enc.ff2.coeffs.grad is not None

    def test_separate_ffn_coeffs(self, encoder_kwargs):
        """Attention and FFN can have different number of coefficients."""
        enc = LegendreEncoder(num_coeffs=6, num_coeffs_ffn=3, **encoder_kwargs)
        assert enc.attn_in_proj.num_coeffs == 6
        assert enc.attn_out_proj.num_coeffs == 6
        assert enc.ff1.num_coeffs == 3
        assert enc.ff2.num_coeffs == 3

        x = torch.randn(2, 10, 64)
        out = enc(x)
        assert out.shape == (2, 10, 64)

        # Should have fewer params than uniform 6 coeffs
        enc_uniform = LegendreEncoder(num_coeffs=6, **encoder_kwargs)
        params_split = sum(p.numel() for p in enc.parameters())
        params_uniform = sum(p.numel() for p in enc_uniform.parameters())
        assert params_split < params_uniform

    def test_forward_with_rope(self, encoder_kwargs):
        enc = LegendreEncoder(num_coeffs=4, use_rope=True, **encoder_kwargs)
        x = torch.randn(2, 10, 64)
        out = enc(x)
        assert out.shape == (2, 10, 64)

    def test_forward_with_cache_store(self, encoder_kwargs):
        enc = LegendreEncoder(num_coeffs=4, **encoder_kwargs)
        x = torch.randn(2, 10, 64)
        cache = KVCache()
        out = enc.forward_with_cache(x, cache, train_size=7, store_cache=True, use_cache=False)
        assert out.shape == (2, 10, 64)
        assert len(cache.kv) == encoder_kwargs["num_blocks"]

    def test_forward_with_cache_use(self, encoder_kwargs):
        enc = LegendreEncoder(num_coeffs=4, **encoder_kwargs)
        enc.eval()
        x = torch.randn(2, 10, 64)

        # Store cache
        cache = KVCache()
        with torch.no_grad():
            out_store = enc.forward_with_cache(x, cache, train_size=7, store_cache=True, use_cache=False)

        # Use cache (test-only input)
        x_test = torch.randn(2, 3, 64)
        with torch.no_grad():
            out_use = enc.forward_with_cache(x_test, cache, use_cache=True, store_cache=False)
        assert out_use.shape == (2, 3, 64)

    def test_fewer_coeffs_fewer_params(self, encoder_kwargs):
        """LegendreEncoder with fewer coefficients should have fewer total parameters."""
        enc_std = Encoder(**encoder_kwargs)
        enc_leg = LegendreEncoder(num_coeffs=4, **encoder_kwargs)

        std_params = sum(p.numel() for p in enc_std.parameters())
        leg_params = sum(p.numel() for p in enc_leg.parameters())
        # Legendre should use fewer params for weight matrices
        # (though per-layer LN/SSMax may offset some savings)
        assert leg_params < std_params

    def test_recompute(self, encoder_kwargs):
        enc = LegendreEncoder(num_coeffs=4, recompute=True, **encoder_kwargs)
        x = torch.randn(2, 10, 64)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert enc.attn_in_proj.coeffs.grad is not None


class TestFromEncoder:
    def test_warm_start_reconstruction_quality(self):
        """Warm-started LegendreEncoder should closely approximate the original."""
        torch.manual_seed(42)
        enc = Encoder(
            num_blocks=8,
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            norm_first=True,
        )
        # Random init to simulate a pretrained encoder
        for p in enc.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        enc.eval()
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            out_orig = enc(x)

        # Convert with many coefficients (should approximate well)
        leg_enc = LegendreEncoder.from_encoder(enc, num_coeffs=8)
        leg_enc.eval()
        with torch.no_grad():
            out_leg = leg_enc(x)

        # With num_coeffs == num_blocks, reconstruction should be near-exact
        assert torch.allclose(out_orig, out_leg, atol=1e-3), (
            f"Max diff: {(out_orig - out_leg).abs().max().item()}"
        )

    def test_warm_start_with_compression(self):
        """Warm-started with fewer coefficients should still produce reasonable output."""
        torch.manual_seed(42)
        enc = Encoder(
            num_blocks=8,
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            norm_first=True,
        )

        enc.eval()
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            out_orig = enc(x)

        # 4 coefficients for 8 layers = 2x compression
        leg_enc = LegendreEncoder.from_encoder(enc, num_coeffs=4)
        leg_enc.eval()
        with torch.no_grad():
            out_leg = leg_enc(x)

        # Should produce finite, reasonable-magnitude output
        assert torch.isfinite(out_leg).all()
        assert out_leg.abs().max() < 100

    def test_warm_start_with_rope(self):
        """Warm start should work with RoPE-enabled encoder."""
        enc = Encoder(
            num_blocks=4,
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            use_rope=True,
        )
        leg_enc = LegendreEncoder.from_encoder(enc, num_coeffs=3)
        assert leg_enc.rope is not None

        x = torch.randn(2, 10, 64)
        out = leg_enc(x)
        assert out.shape == (2, 10, 64)


class TestIntegration:
    def _make_tabicl_data(self, max_classes=5):
        """Create reproducible test data with all classes present in each batch."""
        torch.manual_seed(123)
        B, T, H = 2, 20, 5
        X = torch.randn(B, T, H)
        # Ensure all classes appear in each batch element
        base = torch.arange(max_classes)
        extra = torch.randint(0, max_classes, (15 - max_classes,))
        labels = torch.cat([base, extra])
        y_train = labels.unsqueeze(0).expand(B, -1).clone()
        return X, y_train

    def test_tabicl_with_legendre_icl(self):
        """TabICL should work with Legendre-parameterized ICL encoder."""
        from tabicl.model.tabicl import TabICL

        model = TabICL(
            max_classes=5,
            embed_dim=32,
            col_num_blocks=1,
            col_nhead=4,
            col_num_inds=16,
            row_num_blocks=1,
            row_nhead=4,
            row_num_cls=2,
            icl_num_blocks=4,
            icl_nhead=4,
            icl_legendre_coeffs=2,
            ff_factor=2,
        )
        model.eval()

        X, y_train = self._make_tabicl_data()

        with torch.no_grad():
            out = model._train_forward(X, y_train)
        assert out.shape[0] == 2

    def test_tabicl_with_legendre_row(self):
        """TabICL should work with Legendre-parameterized row encoder."""
        from tabicl.model.tabicl import TabICL

        model = TabICL(
            max_classes=5,
            embed_dim=32,
            col_num_blocks=1,
            col_nhead=4,
            col_num_inds=16,
            row_num_blocks=2,
            row_nhead=4,
            row_num_cls=2,
            icl_num_blocks=2,
            icl_nhead=4,
            row_legendre_coeffs=2,
            ff_factor=2,
        )
        model.eval()

        X, y_train = self._make_tabicl_data()

        with torch.no_grad():
            out = model._train_forward(X, y_train)
        assert out.shape[0] == 2
