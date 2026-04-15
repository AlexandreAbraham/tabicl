"""Legendre polynomial parameterized encoder for weight compression.

Instead of storing independent weights for each transformer block, this module
parameterizes the weight matrices as smooth functions of depth using a Legendre
polynomial basis. A small set of coefficient matrices (num_coeffs << num_blocks)
is learned, and virtual layer weights are reconstructed at inference via:

    W[l] = sum_k C_k * P_k(d_l)

where P_k are Legendre polynomials evaluated at normalized depth positions d_l,
and C_k are the learnable coefficient matrices.
"""

from __future__ import annotations

from typing import Optional, Union
from functools import partial

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import multi_head_attention_forward
from .layers import MultiheadAttentionBlock
from .rope import RotaryEmbedding
from .ssmax import create_ssmax_layer
from .kv_cache import KVCacheEntry, KVCache


def legendre_basis(num_points: int, num_coeffs: int, device: torch.device = None) -> Tensor:
    """Compute Legendre polynomial basis matrix.

    Parameters
    ----------
    num_points : int
        Number of evaluation points (= number of virtual layers).
    num_coeffs : int
        Number of polynomial coefficients (= max degree + 1).
    device : torch.device, optional
        Device for the output tensor.

    Returns
    -------
    Tensor
        Basis matrix of shape (num_points, num_coeffs).
    """
    # Evaluation points in [-1, 1]
    if num_points == 1:
        t = torch.zeros(1, device=device)
    else:
        t = torch.linspace(-1, 1, num_points, device=device)

    # Build Legendre polynomials via recurrence: (n+1)P_{n+1}(t) = (2n+1)tP_n(t) - nP_{n-1}(t)
    B = torch.zeros(num_points, num_coeffs, device=device)
    if num_coeffs >= 1:
        B[:, 0] = 1.0
    if num_coeffs >= 2:
        B[:, 1] = t
    for k in range(2, num_coeffs):
        B[:, k] = ((2 * k - 1) * t * B[:, k - 1] - (k - 1) * B[:, k - 2]) / k

    return B


class LegendreLinear(nn.Module):
    """A linear layer whose weights are reconstructed from Legendre coefficients.

    Stores ``num_coeffs`` coefficient matrices and reconstructs ``num_layers``
    virtual weight matrices on the fly. Each virtual layer also has a learned
    scalar that adjusts magnitude independently (cheap per-layer expressivity).

    Coefficients are initialized with order-dependent decay: C_k gets
    std * 0.5^k, so that C_0 captures the "mean" weight across depth and
    higher orders capture progressively finer variation.

    Parameters
    ----------
    num_layers : int
        Number of virtual layers to reconstruct.
    num_coeffs : int
        Number of Legendre polynomial coefficients.
    out_features : int
        Output dimension of the linear layer.
    in_features : int
        Input dimension of the linear layer.
    bias : bool
        Whether to include per-layer bias vectors (stored independently, not parameterized).
    """

    def __init__(
        self,
        num_layers: int,
        num_coeffs: int,
        out_features: int,
        in_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_coeffs = num_coeffs
        self.out_features = out_features
        self.in_features = in_features

        # Learnable coefficient matrices: (num_coeffs, out_features, in_features)
        # Initialized with order-dependent decay: C_k gets std * 0.5^k
        self.coeffs = nn.Parameter(torch.empty(num_coeffs, out_features, in_features))
        self._init_coeffs()

        # Per-layer scalar for magnitude adjustment (cheap per-layer expressivity)
        self.layer_scales = nn.Parameter(torch.ones(num_layers))

        if bias:
            # Per-layer biases are small, keep them as free parameters
            self.bias = nn.Parameter(torch.zeros(num_layers, out_features))
        else:
            self.register_parameter("bias", None)

        # Precompute and register the basis matrix
        self.register_buffer("basis", legendre_basis(num_layers, num_coeffs))

    def _init_coeffs(self):
        """Initialize coefficients with order-dependent decay.

        C_0 (the "mean" weight) gets the largest initialization. Higher
        polynomial orders get progressively smaller init (0.5^k decay),
        reflecting that fine depth variation should start small.
        """
        std = 1.0 / (self.in_features ** 0.5)
        with torch.no_grad():
            for k in range(self.num_coeffs):
                decay = 0.5 ** k
                nn.init.normal_(self.coeffs[k], std=std * decay)

    def reconstruct_weights(self) -> Tensor:
        """Reconstruct all layer weights from coefficients.

        Returns
        -------
        Tensor
            Weight tensor of shape (num_layers, out_features, in_features).
        """
        # basis: (L, K), coeffs: (K, O, I) -> weights: (L, O, I)
        W = torch.einsum("lk,koi->loi", self.basis, self.coeffs)
        return self.layer_scales[:, None, None] * W

    def get_weight(self, layer_idx: int) -> Tensor:
        """Get reconstructed weight for a single layer.

        Parameters
        ----------
        layer_idx : int
            Index of the virtual layer.

        Returns
        -------
        Tensor
            Weight matrix of shape (out_features, in_features).
        """
        # basis[layer_idx]: (K,), coeffs: (K, O, I) -> (O, I)
        W = torch.einsum("k,koi->oi", self.basis[layer_idx], self.coeffs)
        return self.layer_scales[layer_idx] * W

    def get_bias(self, layer_idx: int) -> Optional[Tensor]:
        """Get bias for a single layer."""
        if self.bias is None:
            return None
        return self.bias[layer_idx]


class LegendreAttentionBlock(nn.Module):
    """A single virtual attention block that receives reconstructed weights.

    This is a lightweight shell — it holds only the non-parameterized components
    (LayerNorm, dropout, SSMax) and delegates weight matrices from the parent
    LegendreEncoder at forward time.

    Parameters
    ----------
    d_model : int
        Model dimension.
    nhead : int
        Number of attention heads.
    dim_feedforward : int
        FFN hidden dimension.
    dropout : float
        Dropout probability.
    activation : str or callable
        Activation function for the FFN.
    norm_first : bool
        Pre-norm (True) or post-norm (False).
    bias_free_ln : bool
        If True, removes bias from LayerNorm.
    ssmax : bool or str
        SSMax type for this block.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        ssmax: Union[bool, str] = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.norm_first = norm_first

        # LayerNorms are per-layer (small, not worth parameterizing)
        ln_bias = not bias_free_ln
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=True, bias=ln_bias)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=True, bias=ln_bias)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        # Activation
        if isinstance(activation, str):
            self.activation = {"relu": F.relu, "gelu": F.gelu}[activation]
        else:
            self.activation = activation

        # SSMax (per-layer, small)
        if isinstance(ssmax, bool):
            ssmax = "qassmax-mlp-elementwise" if ssmax else "none"
        self.ssmax_layer = create_ssmax_layer(ssmax_type=ssmax, num_heads=nhead, embed_dim=d_model)

    def _attn_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        rope: Optional[RotaryEmbedding],
        key_padding_mask: Optional[Tensor] = None,
        cached_kv: Optional[KVCacheEntry] = None,
        need_kv: bool = False,
    ):
        """Run multi-head attention with externally provided weights."""
        dropout_p = self.dropout1.p if self.training else 0.0
        return multi_head_attention_forward(
            q,
            self.nhead,
            in_proj_weight,
            in_proj_bias,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            key=k,
            value=v,
            cached_kv=cached_kv,
            training=self.training,
            key_padding_mask=key_padding_mask,
            rope=rope,
            ssmax_layer=self.ssmax_layer,
            need_kv=need_kv,
        )

    def _ff_forward(self, x: Tensor, ff1_weight: Tensor, ff1_bias: Optional[Tensor],
                    ff2_weight: Tensor, ff2_bias: Optional[Tensor]) -> Tensor:
        """Run FFN with externally provided weights."""
        out = F.linear(x, ff1_weight, ff1_bias)
        out = self.activation(out)
        out = self.dropout_ff(out)
        return F.linear(out, ff2_weight, ff2_bias)

    def forward(
        self,
        q: Tensor,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        ff1_weight: Tensor,
        ff1_bias: Optional[Tensor],
        ff2_weight: Tensor,
        ff2_bias: Optional[Tensor],
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        train_size: Optional[int] = None,
        rope: Optional[RotaryEmbedding] = None,
        key_padding_mask: Optional[Tensor] = None,
        cached_kv: Optional[KVCacheEntry] = None,
        need_kv: bool = False,
    ):
        """Forward pass with externally provided weights.

        Parameters
        ----------
        q : Tensor
            Query input of shape (..., seq_len, d_model).
        in_proj_weight, in_proj_bias : Tensor
            QKV projection weights/biases.
        out_proj_weight, out_proj_bias : Tensor
            Output projection weights/biases.
        ff1_weight, ff1_bias : Tensor
            First FFN layer weights/biases.
        ff2_weight, ff2_bias : Tensor
            Second FFN layer weights/biases.
        k : Optional[Tensor]
            Key tensor for cross-attention. If None, uses q (self-attention).
        v : Optional[Tensor]
            Value tensor for cross-attention. If None, uses k.
        train_size : Optional[int]
            When set, keys/values are sliced to first train_size positions.
        rope : Optional[RotaryEmbedding]
            Rotary positional encoding.
        key_padding_mask : Optional[Tensor]
            Mask for padding positions in key sequence.
        cached_kv : Optional[KVCacheEntry]
            Cached key/value projections.
        need_kv : bool
            Whether to return K/V projections for caching.

        Returns
        -------
        Tensor or Tuple[Tensor, Tensor, Tensor]
        """
        # Resolve k, v for self-attention vs cross-attention
        if cached_kv is not None:
            pass  # k, v ignored when using cache
        elif k is None and v is None:
            if train_size is not None:
                k = v = q[..., :train_size, :]
            else:
                k = v = q
        else:
            if k is None:
                k = q
            if v is None:
                v = k

        use_cache = cached_kv is not None
        k_proj, v_proj = None, None

        attn_kwargs = dict(
            in_proj_weight=in_proj_weight, in_proj_bias=in_proj_bias,
            out_proj_weight=out_proj_weight, out_proj_bias=out_proj_bias,
            rope=rope, key_padding_mask=key_padding_mask,
        )

        if self.norm_first:
            q_normed = self.norm1(q)
            if use_cache:
                attn = self._attn_forward(q_normed, None, None, cached_kv=cached_kv, **attn_kwargs)
            else:
                k_normed = self.norm1(k) if k is not q else q_normed
                v_normed = self.norm1(v) if v is not k else k_normed
                result = self._attn_forward(q_normed, k_normed, v_normed, need_kv=need_kv, **attn_kwargs)
                if need_kv and isinstance(result, tuple):
                    attn, k_proj, v_proj = result
                else:
                    attn = result

            x = q + self.dropout1(attn)
            x = x + self.dropout2(self._ff_forward(self.norm2(x), ff1_weight, ff1_bias, ff2_weight, ff2_bias))
        else:
            if use_cache:
                attn = self._attn_forward(q, None, None, cached_kv=cached_kv, **attn_kwargs)
            else:
                result = self._attn_forward(q, k, v, need_kv=need_kv, **attn_kwargs)
                if need_kv and isinstance(result, tuple):
                    attn, k_proj, v_proj = result
                else:
                    attn = result

            x = self.norm1(q + self.dropout1(attn))
            x = self.norm2(x + self.dropout2(self._ff_forward(x, ff1_weight, ff1_bias, ff2_weight, ff2_bias)))

        if need_kv and k_proj is not None:
            return x, k_proj, v_proj
        return x


class LegendreEncoder(nn.Module):
    """Transformer encoder with Legendre polynomial weight parameterization.

    Compatible with the Encoder interface including ``call_block()`` used by
    RowInteraction.

    Instead of ``num_blocks`` independent attention blocks, this encoder stores
    ``num_coeffs`` Legendre coefficient matrices per weight group and reconstructs
    virtual layer weights at forward time. This provides:

    - ~(num_blocks / num_coeffs)x compression on weight storage
    - ~(num_blocks / num_coeffs)x reduction in optimizer state memory
    - Implicit smoothness regularization across depth

    The non-weight parameters (LayerNorm, SSMax, dropout) remain per-layer.

    Parameters
    ----------
    num_blocks : int
        Number of virtual transformer blocks to reconstruct.
    num_coeffs : int
        Number of Legendre polynomial coefficients for attention weights.
    d_model : int
        Model dimension.
    nhead : int
        Number of attention heads.
    dim_feedforward : int
        FFN hidden dimension.
    dropout : float
        Dropout probability.
    activation : str or callable
        Activation function for the FFN.
    norm_first : bool
        Pre-norm (True) or post-norm (False).
    bias_free_ln : bool
        If True, removes bias from LayerNorm.
    use_rope : bool
        Whether to use rotary positional encoding.
    rope_base : int
        Base for rotary positional encoding.
    rope_interleaved : bool
        Interleaved or contiguous RoPE.
    ssmax : bool or str
        SSMax type.
    recompute : bool
        Use gradient checkpointing.
    num_coeffs_ffn : int or None
        Number of Legendre coefficients for FFN weights. If None, uses the
        same as ``num_coeffs`` (the attention value). FFN weights are often
        more compressible than attention weights, so fewer coefficients may
        suffice (e.g., num_coeffs=6 for attention, num_coeffs_ffn=3 for FFN).
    sandwich : bool
        If True (default), the first and last blocks are independent standard
        ``MultiheadAttentionBlock`` with their own weights, and only the
        middle blocks use Legendre parameterization. This follows the
        empirical observation that boundary layers behave differently.
        Requires ``num_blocks >= 3``. If False, all blocks use Legendre.
    """

    def __init__(
        self,
        num_blocks: int,
        num_coeffs: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        use_rope: bool = False,
        rope_base: int = 100000,
        rope_interleaved: bool = True,
        ssmax: Union[bool, str] = False,
        recompute: bool = False,
        num_coeffs_ffn: Optional[int] = None,
        sandwich: bool = True,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        if sandwich and num_blocks < 3:
            sandwich = False  # Not enough blocks for sandwich structure

        if num_coeffs_ffn is None:
            num_coeffs_ffn = num_coeffs

        self.num_blocks = num_blocks
        self.num_coeffs = num_coeffs
        self.num_coeffs_ffn = num_coeffs_ffn
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.recompute = recompute
        self.sandwich = sandwich

        # Number of Legendre-parameterized middle blocks
        num_middle = num_blocks - 2 if sandwich else num_blocks

        # Legendre-parameterized weight groups (for middle blocks only)
        self.attn_in_proj = LegendreLinear(num_middle, num_coeffs, 3 * d_model, d_model, bias=True)
        self.attn_out_proj = LegendreLinear(num_middle, num_coeffs, d_model, d_model, bias=True)
        self.ff1 = LegendreLinear(num_middle, num_coeffs_ffn, dim_feedforward, d_model, bias=True)
        self.ff2 = LegendreLinear(num_middle, num_coeffs_ffn, d_model, dim_feedforward, bias=True)

        block_kwargs = dict(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, norm_first=norm_first,
            bias_free_ln=bias_free_ln, ssmax=ssmax,
        )

        # Build block list: [independent?, legendre..., independent?]
        blocks = []
        for i in range(num_blocks):
            if sandwich and (i == 0 or i == num_blocks - 1):
                blocks.append(MultiheadAttentionBlock(**block_kwargs))
            else:
                blocks.append(LegendreAttentionBlock(**block_kwargs))
        self.blocks = nn.ModuleList(blocks)

        self.rope = (
            RotaryEmbedding(dim=d_model // nhead, theta=rope_base, interleaved=rope_interleaved)
            if use_rope
            else None
        )

    def _is_independent(self, block_idx: int) -> bool:
        """Check if a block is an independent (non-Legendre) block."""
        return self.sandwich and (block_idx == 0 or block_idx == self.num_blocks - 1)

    def _legendre_idx(self, block_idx: int) -> int:
        """Map outer block index to Legendre weight generator index."""
        return block_idx - 1 if self.sandwich else block_idx

    def _get_block_weights(self, block_idx: int, weights=None) -> dict:
        """Build weight kwargs for a Legendre block.

        Parameters
        ----------
        block_idx : int
            Outer block index (0..num_blocks-1).
        weights : tuple of Tensor, optional
            Pre-reconstructed (attn_in, attn_out, ff1, ff2) weight tensors
            indexed by Legendre index (0..num_middle-1).
            If None, reconstructs for this single block via get_weight().
        """
        li = self._legendre_idx(block_idx)
        if weights is not None:
            attn_in_weights, attn_out_weights, ff1_weights, ff2_weights = weights
            return dict(
                in_proj_weight=attn_in_weights[li],
                in_proj_bias=self.attn_in_proj.get_bias(li),
                out_proj_weight=attn_out_weights[li],
                out_proj_bias=self.attn_out_proj.get_bias(li),
                ff1_weight=ff1_weights[li],
                ff1_bias=self.ff1.get_bias(li),
                ff2_weight=ff2_weights[li],
                ff2_bias=self.ff2.get_bias(li),
            )
        return dict(
            in_proj_weight=self.attn_in_proj.get_weight(li),
            in_proj_bias=self.attn_in_proj.get_bias(li),
            out_proj_weight=self.attn_out_proj.get_weight(li),
            out_proj_bias=self.attn_out_proj.get_bias(li),
            ff1_weight=self.ff1.get_weight(li),
            ff1_bias=self.ff1.get_bias(li),
            ff2_weight=self.ff2.get_weight(li),
            ff2_bias=self.ff2.get_bias(li),
        )

    def _reconstruct_all(self):
        """Reconstruct all weight matrices. Returns a tuple of 4 tensors."""
        return (
            self.attn_in_proj.reconstruct_weights(),
            self.attn_out_proj.reconstruct_weights(),
            self.ff1.reconstruct_weights(),
            self.ff2.reconstruct_weights(),
        )

    def call_block(self, block_idx: int, *args, **kwargs):
        """Call a single block by index.

        For independent (sandwich) blocks, forwards args directly to the
        standard MultiheadAttentionBlock. For Legendre blocks, reconstructs
        weights and passes them as kwargs.

        Used by RowInteraction._aggregate_embeddings.
        """
        if self._is_independent(block_idx):
            # Standard block — forward args/kwargs directly
            return self.blocks[block_idx](*args, **kwargs)

        # Legendre block — reconstruct weights and translate kwargs
        weight_kwargs = self._get_block_weights(block_idx)

        # Handle positional arg (checkpoint passes q as positional)
        if args:
            kwargs["q"] = args[0]

        weight_kwargs.update(
            {key: kwargs[key] for key in ("q", "k", "v", "train_size", "rope",
                                          "key_padding_mask", "cached_kv", "need_kv")
             if key in kwargs}
        )

        return self.blocks[block_idx](**weight_kwargs)

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        """Process input through virtual Legendre-parameterized blocks.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model).
        train_size : Optional[int]
            Number of training samples for causal context.

        Returns
        -------
        Tensor
            Output tensor with same shape as ``src``.
        """
        weights = self._reconstruct_all()

        out = src
        for i, block in enumerate(self.blocks):
            if self._is_independent(i):
                if self.recompute:
                    out = checkpoint(partial(block, train_size=train_size, rope=self.rope),
                                     out, use_reentrant=False)
                else:
                    out = block(q=out, train_size=train_size, rope=self.rope)
            else:
                block_kwargs = self._get_block_weights(i, weights)
                block_kwargs.update(q=out, train_size=train_size, rope=self.rope)
                if self.recompute:
                    out = checkpoint(partial(block, **{k: v for k, v in block_kwargs.items() if k != "q"}),
                                     block_kwargs["q"], use_reentrant=False)
                else:
                    out = block(**block_kwargs)

        return out

    def forward_with_cache(
        self,
        src: Tensor,
        icl_cache: KVCache,
        train_size: Optional[int] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        """Process input through virtual blocks with KV caching support.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model).
        icl_cache : KVCache
            Cache for K/V projections per layer.
        train_size : Optional[int]
            Number of training samples.
        use_cache : bool
            Whether to use cached K/V.
        store_cache : bool
            Whether to store computed K/V in cache.

        Returns
        -------
        Tensor
            Output tensor with same shape as ``src``.
        """
        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if store_cache and train_size is None:
            raise ValueError("train_size must be provided when store_cache=True")

        weights = self._reconstruct_all()

        out = src
        for i, block in enumerate(self.blocks):
            if self._is_independent(i):
                if use_cache:
                    out = block(q=out, rope=self.rope, cached_kv=icl_cache.kv[i])
                else:
                    out, k_proj, v_proj = block(q=out, train_size=train_size, rope=self.rope, need_kv=True)
                    icl_cache.kv[i] = KVCacheEntry(key=k_proj, value=v_proj)
            else:
                block_kwargs = self._get_block_weights(i, weights)
                block_kwargs["rope"] = self.rope
                if use_cache:
                    block_kwargs.update(q=out, cached_kv=icl_cache.kv[i])
                    out = block(**block_kwargs)
                else:
                    block_kwargs.update(q=out, train_size=train_size, need_kv=True)
                    out, k_proj, v_proj = block(**block_kwargs)
                    icl_cache.kv[i] = KVCacheEntry(key=k_proj, value=v_proj)

        return out

    @staticmethod
    def from_encoder(encoder, num_coeffs: int, num_coeffs_ffn: Optional[int] = None) -> "LegendreEncoder":
        """Create a LegendreEncoder warm-started from a pretrained standard Encoder.

        Fits Legendre coefficients via least-squares projection of the pretrained
        weight matrices.

        Parameters
        ----------
        encoder : Encoder
            A pretrained standard Encoder instance.
        num_coeffs : int
            Number of Legendre coefficients for attention weights.
        num_coeffs_ffn : int or None
            Number of Legendre coefficients for FFN weights. If None, uses
            ``num_coeffs``.

        Returns
        -------
        LegendreEncoder
            A new LegendreEncoder with coefficients initialized from the
            pretrained weights.
        """
        from .encoders import Encoder
        from .ssmax import SSMax, SSMaxMLP, QASSMaxMLP

        assert isinstance(encoder, Encoder), "encoder must be an Encoder instance"

        num_blocks = len(encoder.blocks)
        block0 = encoder.blocks[0]
        d_model = block0.attn.embed_dim
        nhead = block0.attn.num_heads
        dim_feedforward = block0.linear1.out_features
        dropout = block0.dropout1.p
        norm_first = block0.norm_first
        bias_free_ln = block0.norm1.bias is None

        # Detect SSMax type from the source encoder's attention blocks
        ssmax_layer = block0.attn.ssmax_layer
        if ssmax_layer is None:
            ssmax = False
        elif isinstance(ssmax_layer, QASSMaxMLP):
            ssmax = "qassmax-mlp-elementwise" if ssmax_layer.elementwise else "qassmax-mlp"
        elif isinstance(ssmax_layer, SSMaxMLP):
            ssmax = "ssmax-mlp-elementwise" if getattr(ssmax_layer, 'elementwise', False) else "ssmax-mlp"
        elif isinstance(ssmax_layer, SSMax):
            ssmax = "ssmax"
        else:
            ssmax = False

        # Detect RoPE
        use_rope = encoder.rope is not None
        rope_base = 100000
        rope_interleaved = True
        if use_rope and hasattr(encoder.rope, '_theta'):
            rope_base = int(encoder.rope._theta)
        if use_rope and hasattr(encoder.rope, 'interleaved'):
            rope_interleaved = encoder.rope.interleaved

        leg_encoder = LegendreEncoder(
            num_blocks=num_blocks,
            num_coeffs=num_coeffs,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            use_rope=use_rope,
            rope_base=rope_base,
            rope_interleaved=rope_interleaved,
            ssmax=ssmax,
            recompute=encoder.recompute,
            num_coeffs_ffn=num_coeffs_ffn,
        )

        # Copy RoPE if present
        if use_rope and encoder.rope is not None:
            leg_encoder.rope.load_state_dict(encoder.rope.state_dict())

        # Identify which source blocks are middle (Legendre) blocks
        middle_blocks = []
        for i, blk in enumerate(encoder.blocks):
            if not leg_encoder._is_independent(i):
                middle_blocks.append(blk)

        # Collect per-layer weights from the middle blocks
        device = block0.attn.in_proj_weight.device
        num_middle = len(middle_blocks)

        # Precompute pseudo-inverses for each coefficient count
        _pinv_cache = {}
        def _get_pinv(nc):
            if nc not in _pinv_cache:
                basis = legendre_basis(num_middle, nc, device=device).float()
                _pinv_cache[nc] = torch.linalg.pinv(basis)
            return _pinv_cache[nc]

        # Stack weights along layer dimension and project
        def _stack_and_project(get_param_fn, leg_linear: LegendreLinear):
            """Stack middle-block params and fit Legendre coefficients via least-squares."""
            weights = torch.stack([get_param_fn(blk) for blk in middle_blocks], dim=0)
            pinv = _get_pinv(leg_linear.num_coeffs)
            coeffs = torch.einsum("kl,loi->koi", pinv, weights.float())
            leg_linear.coeffs.data.copy_(coeffs.to(leg_linear.coeffs.dtype))
            leg_linear.layer_scales.data.fill_(1.0)

        def _copy_biases(get_bias_fn, leg_linear: LegendreLinear):
            """Copy per-layer biases from middle blocks."""
            if leg_linear.bias is not None:
                for i, blk in enumerate(middle_blocks):
                    bias = get_bias_fn(blk)
                    if bias is not None:
                        leg_linear.bias.data[i].copy_(bias.data)

        # Project middle block weights into Legendre coefficients
        _stack_and_project(lambda blk: blk.attn.in_proj_weight, leg_encoder.attn_in_proj)
        _copy_biases(lambda blk: blk.attn.in_proj_bias, leg_encoder.attn_in_proj)
        _stack_and_project(lambda blk: blk.attn.out_proj.weight, leg_encoder.attn_out_proj)
        _copy_biases(lambda blk: blk.attn.out_proj.bias, leg_encoder.attn_out_proj)
        _stack_and_project(lambda blk: blk.linear1.weight, leg_encoder.ff1)
        _copy_biases(lambda blk: blk.linear1.bias, leg_encoder.ff1)
        _stack_and_project(lambda blk: blk.linear2.weight, leg_encoder.ff2)
        _copy_biases(lambda blk: blk.linear2.bias, leg_encoder.ff2)

        # Copy per-block components
        for i, (src_block, dst_block) in enumerate(zip(encoder.blocks, leg_encoder.blocks)):
            if leg_encoder._is_independent(i):
                # Independent block — copy full state_dict
                dst_block.load_state_dict(src_block.state_dict())
            else:
                # Legendre block — copy LayerNorm and SSMax
                dst_block.norm1.load_state_dict(src_block.norm1.state_dict())
                dst_block.norm2.load_state_dict(src_block.norm2.state_dict())
                if src_block.attn.ssmax_layer is not None:
                    dst_block.ssmax_layer.load_state_dict(src_block.attn.ssmax_layer.state_dict())

        return leg_encoder
