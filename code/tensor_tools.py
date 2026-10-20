from dataclasses import dataclass, field
import math
from typing import Dict, List, Literal
import torch
import torch.nn as nn

# Note: if weight matrices are initialized badly -> starting loss is more than ln(vocab)
def init_custom_weight(
    tensor: torch.Tensor, 
    is_residual_output: bool = False, 
    num_layers: int = 12,
    mode: str = "trunc_normal"
):
    """
    Initializes a weight tensor of shape [in_features, out_features].
    """
    in_features, out_features = tensor.shape[0], tensor.shape[1]
    
    if mode == "trunc_normal":
        # Standard GPT-style truncated normal
        std = 0.02
        if is_residual_output:
            std = std / math.sqrt(2 * num_layers)
            
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2*std, b=2*std)

    elif mode == "xavier":
        # Glorot / Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
        std = math.sqrt(2.0 / (in_features + out_features))
        if is_residual_output:
            std = std / math.sqrt(2 * num_layers)
            
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2*std, b=2*std)

    elif mode == "kaiming":
        # He / Kaiming initialization (ideal for GELU/SiLU in FFN): std = sqrt(2 / fan_in)
        std = math.sqrt(2.0 / in_features)
        if is_residual_output:
            std = std / math.sqrt(2 * num_layers)
            
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2*std, b=2*std)
    else: 
        raise ValueError


# ------------------------------------

# tensor health checking 


TensorCategory = Literal["weight", "activation", "gradient", "optimizer_state"]


@dataclass
class TensorHealthReport:
    name: str
    category: TensorCategory
    status: str  # "HEALTHY", "WARNING", "CRITICAL"
    metrics: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)

    def print_summary(self):
        color = {
            "HEALTHY": "\033[92m",
            "WARNING": "\033[93m",
            "CRITICAL": "\033[91m",
        }.get(self.status, "\033[0m")
        reset = "\033[0m"

        print(
            f"=== {self.name} [{self.category.upper()}] Status: {color}{self.status}{reset} ==="
        )
        print("Metrics:")
        for k, v in self.metrics.items():
            print(f"  - {k}: {v:.6e}" if isinstance(v, float) else f"  - {k}: {v}")

        if self.warnings:
            print(f"\n{color}Warnings & Diagnostics:{reset}")
            for w in self.warnings:
                print(f"  ! {w}")

        if self.consequences:
            print("\nConsequences:")
            for c in self.consequences:
                print(f"  -> {c}")
        print("=" * 60 + "\n")


def inspect_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    category: TensorCategory = "activation",
    weight_decay: float = 0.0,
    lr: float = 1e-3,
) -> TensorHealthReport:
    """Analyzes a tensor's statistical health and provides warnings on training dynamics."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    # Detach and cast to float32 for metric precision without modifying autograd graph
    t = tensor.detach().to(dtype=torch.float32)

    total_elements = t.numel()
    if total_elements == 0:
        return TensorHealthReport(
            name=name,
            category=category,
            status="CRITICAL",
            metrics={"numel": 0},
            warnings=["Tensor is empty."],
            consequences=["Downstream matrix operations will fail or yield shape errors."],
        )

    # Core Sanity Metrics
    nan_count = torch.isnan(t).sum().item()
    inf_count = torch.isinf(t).sum().item()
    zero_count = (t == 0).sum().item()
    zero_fraction = zero_count / total_elements

    mean_val = torch.mean(t).item()
    std_val = torch.std(t, unbiased=False).item()
    var_val = torch.var(t, unbiased=False).item()
    min_val = torch.min(t).item()
    max_val = torch.max(t).item()
    l2_norm = torch.linalg.norm(t.view(-1)).item()

    metrics = {
        "numel": total_elements,
        "mean": mean_val,
        "std": std_val,
        "var": var_val,
        "min": min_val,
        "max": max_val,
        "l2_norm": l2_norm,
        "zero_fraction": zero_fraction,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }

    warnings = []
    consequences = []
    status = "HEALTHY"

    # --- FATAL CHECKS (ALL CATEGORIES) ---
    if nan_count > 0 or inf_count > 0:
        status = "CRITICAL"
        warnings.append(
            f"Detected {nan_count} NaNs and {inf_count} Infs in tensor."
        )
        consequences.append(
            "Numerical overflow or illegal math operation (e.g., log(0), 0/0). "
            "Grids, activations, and loss values will corrupt permanently across autograd."
        )
        return TensorHealthReport(name, category, status, metrics, warnings, consequences)

    # --- CATEGORY-SPECIFIC HEALTH ANALYSIS ---

    if category == "weight":
        # Outlier detection (3-sigma rule)
        outliers = (torch.abs(t - mean_val) > 3 * (std_val + 1e-8)).sum().item()
        outlier_ratio = outliers / total_elements
        metrics["outlier_ratio_3sigma"] = outlier_ratio

        if std_val < 1e-6:
            status = "CRITICAL" if std_val == 0 else "WARNING"
            warnings.append(f"Near-zero weight variance: std = {std_val:.2e}")
            consequences.append(
                "Weight symmetry break failed. Neurons in this layer will compute identical "
                "features and receive identical gradient updates."
            )

        if max_val > 10.0 or min_val < -10.0:
            status = "WARNING" if status != "CRITICAL" else status
            warnings.append(f"Large weight magnitudes detected: range [{min_val:.2f}, {max_val:.2f}]")
            consequences.append(
                "Increases susceptibility to exploding activation variance. "
                "Weight decay hyperparameter may be too weak."
            )

        if zero_fraction > 0.70:
            status = "WARNING" if status != "CRITICAL" else status
            warnings.append(f"High weight sparsity: {zero_fraction * 100:.1f}% zeros.")
            consequences.append(
                "Unintended network pruning. Excess capacity is unutilized unless using structured sparsity."
            )

    elif category == "activation":
        dead_relu_fraction = (t <= 0).sum().item() / total_elements
        metrics["dead_relu_fraction"] = dead_relu_fraction

        # Saturated values check (for Sigmoid/Tanh or Softmax logits)
        saturated_high = (t > 0.99).sum().item() / total_elements
        saturated_low = (t < -0.99).sum().item() / total_elements
        logit_spread = max_val - min_val
        metrics["logit_spread"] = logit_spread

        if var_val > 50.0:
            status = "WARNING" if status != "CRITICAL" else status
            warnings.append(f"High activation variance: var = {var_val:.2f}")
            consequences.append(
                "Activation explosion. Subsequent Normalization layers (RMSNorm/LayerNorm) "
                "will apply extreme scaling factors, suppressing residual stream updates."
            )

        if var_val < 1e-4:
            status = "WARNING" if status != "CRITICAL" else status
            warnings.append(f"Collapsed activation variance: var = {var_val:.2e}")
            consequences.append(
                "Activation collapse / signal starvation. Later layers receive a constant input vector, "
                "effectively disabling deep representation learning."
            )

        if logit_spread > 20.0:
            status = "WARNING" if status != "CRITICAL" else status
            warnings.append(f"Extreme logit spread before Softmax: max - min = {logit_spread:.2f}")
            consequences.append(
                "Softmax saturation. Probabilities will collapse into single one-hot vectors, "
                "driving cross-entropy gradients to zero."
            )

        if dead_relu_fraction > 0.80:
            status = "WARNING" if status != "CRITICAL" else status
            warnings.append(f"High inactive activation ratio: {dead_relu_fraction * 100:.1f}% <= 0")
            consequences.append(
                "Dying ReLU problem or aggressive thresholding. Large portions of the layer "
                "are completely inactive and receive zero gradient updates."
            )

    elif category == "gradient":
        mean_abs_grad = torch.mean(torch.abs(t)).item()
        metrics["mean_abs_grad"] = mean_abs_grad

        if l2_norm < 1e-7:
            status = "WARNING" if status != "CRITICAL" else status
            warnings.append(f"Vanishing gradient detected: L2 norm = {l2_norm:.2e}")
            consequences.append(
                "Vanishing gradients. Early layers will cease updating. Check for missing "
                "residual paths, deep unnormalized blocks, or saturated activation functions."
            )

        elif l2_norm > 100.0 or max_val > 50.0:
            status = "WARNING" if status != "CRITICAL" else status
            warnings.append(f"Exploding gradient detected: L2 norm = {l2_norm:.2f}, max = {max_val:.2f}")
            consequences.append(
                "Exploding gradients. Optimization steps will take massive jumps, overshooting "
                "minima and causing loss spikes or NaN values. Enable gradient clipping (`clip_grad_norm_`)."
            )

        if zero_fraction > 0.90:
            status = "WARNING" if status != "CRITICAL" else status
            warnings.append(f"Gradient sparsity is high: {zero_fraction * 100:.1f}% zeros")
            consequences.append(
                "Sparse gradient updates. Check if custom autograd functions are masking gradients, "
                "or if embedding indices are rarely accessed."
            )

    elif category == "optimizer_state":
        # Evaluates momentum (m) or variance tracking (v) tensors in Adam/AdamW
        metrics["state_rms"] = torch.sqrt(torch.mean(t**2)).item()

        if max_val > 1e4:
            status = "WARNING" if status != "CRITICAL" else status
            warnings.append(f"Large optimizer state magnitudes: max = {max_val:.2e}")
            consequences.append(
                "Unstable second-moment estimates ($v_t$ in Adam). The Effective step size "
                r"$\frac{\eta}{\sqrt{v_t} + \epsilon}$ will collapse to near zero, freezing parameter updates."
            )

        if zero_fraction > 0.50:
            status = "WARNING" if status != "CRITICAL" else status
            warnings.append(f"Uninitialized or unupdated optimizer state: {zero_fraction * 100:.1f}% zeros")
            consequences.append(
                "Optimizer state imbalance. Parameters with zero momentum state will take unconditioned "
                "first-order steps relative to warm parameters."
            )

    return TensorHealthReport(
        name=name,
        category=category,
        status=status,
        metrics=metrics,
        warnings=warnings,
        consequences=consequences,
    )
