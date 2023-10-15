from time import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from batch_ops_pytorch import BatchConv2DLayer, BatchLinearLayer

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def benchmark(
    layer: nn.Module,
    x: torch.Tensor,
    weights: Tuple[torch.Tensor],
    biases: Optional[Tuple[torch.Tensor]] = None,
    y: Optional[torch.Tensor] = None,
    backward: bool = True,
    R: int = 1000,
    dtype: torch.dtype = torch.float32,
    jit_method: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark a given layer/model for forward and backward pass.
    """
    fprop_timings = []
    bprop_timings = []
    combined_timings = []

    # Convert tensors to the desired data type
    layer = layer.to(dtype=dtype)
    x = x.to(dtype=dtype)
    weights = tuple(w.to(dtype=dtype) for w in weights)
    if biases:
        biases = tuple(b.to(dtype=dtype) for b in biases)

    # Apply JIT methods if specified
    if jit_method == "trace":
        layer = torch.jit.trace(layer, (x, *weights, *biases))
    elif jit_method == "script":
        layer = torch.jit.script(layer)

    for _ in range(R):
        if x.grad:
            x.grad.zero_()
        for w in weights:
            if w.grad:
                w.grad.zero_()

        start_time = time()
        out = layer(x, *weights, *biases)
        fprop_timings.append(time() - start_time)

        if y is not None and backward:
            criterion = torch.nn.CrossEntropyLoss()
            out = out.reshape(-1, out.shape[-1])
            loss = criterion(out, y.view(-1))

            start_time = time()
            loss.backward()
            bprop_timings.append(time() - start_time)

        combined_timings.append(
            fprop_timings[-1] + (bprop_timings[-1] if bprop_timings else 0)
        )

    def compute_stats(timings):
        timings_tensor = torch.tensor(timings)
        mean_time = torch.mean(timings_tensor).item()
        std_time = torch.std(timings_tensor).item()
        max_time = torch.max(timings_tensor).item()
        min_time = torch.min(timings_tensor).item()
        median_time = torch.median(timings_tensor).item()

        return {
            "mean": mean_time,
            "std": std_time,
            "max": max_time,
            "min": min_time,
            "median": median_time,
        }

    return {
        "fprop": compute_stats(fprop_timings),
        "bprop": compute_stats(bprop_timings),
        "combined": compute_stats(combined_timings),
    }


class ConvNetWithGlobalPooling(nn.Module):
    def __init__(self, channel_in):
        super().__init__()
        self.conv1 = BatchConv2DLayer(channel_in, 64).to(device)
        self.conv2 = BatchConv2DLayer(64, 128).to(device)
        self.conv3 = BatchConv2DLayer(128, 256).to(device)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = BatchLinearLayer().to(device)

    def forward(self, x, *weights_and_biases):
        (
            conv1_weight,
            conv2_weight,
            conv3_weight,
            fc_weight,
            *biases,
        ) = weights_and_biases
        x = F.relu(self.conv1(x, weight=conv1_weight, bias=biases[0]))
        x = F.relu(self.conv2(x, weight=conv2_weight, bias=biases[1]))
        x = F.relu(self.conv3(x, weight=conv3_weight, bias=biases[2]))
        x = self.global_pool(x)
        B, N = x.shape[:2]
        x = x.view(B, N, -1)
        x = self.fc(x, weight=fc_weight, bias=biases[3])
        return x


class FourLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = BatchLinearLayer().to(device)
        self.fc2 = BatchLinearLayer().to(device)
        self.fc3 = BatchLinearLayer().to(device)
        self.fc4 = BatchLinearLayer().to(device)

    def forward(self, x, *weights_and_biases):
        (
            fc1_weight,
            fc2_weight,
            fc3_weight,
            fc4_weight,
            *biases,
        ) = weights_and_biases
        x = x.view(x.shape[0], x.shape[1], -1)
        x = F.relu(self.fc1(x, weight=fc1_weight, bias=biases[0]))
        x = F.relu(self.fc2(x, weight=fc2_weight, bias=biases[1]))
        x = F.relu(self.fc3(x, weight=fc3_weight, bias=biases[2]))
        x = self.fc4(x, weight=fc4_weight, bias=biases[3])
        return x


def single_layer_bench(x, y, B, N, C, H, W, num_classes):
    # MLP Benchmark
    mlp_weights = (torch.randn(B, C * H * W, num_classes).to(device),)
    mlp_biases = (torch.randn(B, num_classes).to(device),)
    mlp = BatchLinearLayer().to(device)
    mlp_time = benchmark(
        mlp, x.view(B, N, -1), mlp_weights, mlp_biases, backward=False
    )
    wandb.log({"MLP Benchmark Time": mlp_time})

    # Convolutional Layer Benchmark
    conv_weights = (torch.randn(B, num_classes, C, 3, 3).to(device),)
    conv_biases = (torch.randn(B, num_classes).to(device),)
    conv = BatchConv2DLayer(in_channels=C, out_channels=num_classes).to(device)
    conv_time = benchmark(conv, x, conv_weights, conv_biases, backward=False)
    wandb.log({"Conv Layer Benchmark Time": conv_time})


def main():
    C, H, W = 3, 32, 32
    num_classes = 10

    # Looping over different values of B and N for benchmarking
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    modes = ["normal", "trace", "script"]

    for dtype in dtypes:
        for mode in modes:
            for B in [1, 10, 50, 100]:
                for N in [1, 10, 50, 100, 500]:
                    try:
                        wandb.init(
                            project="pytorch-vs-jax-benchmarking",
                            name=f"pytorch-benchmark-B{B}-N{N}-dtype{dtype}-mode{mode}",
                            reinit=True,
                        )
                        wandb.config.num_models = B
                        wandb.config.batch_size = N

                        x = torch.randn(B, N, C, H, W, requires_grad=True).to(
                            device
                        )
                        y = torch.randint(0, num_classes, (B, N)).to(device)

                        # Single Layer Benchmark
                        single_layer_bench(x, y, B, N, C, H, W, num_classes)

                        # ConvNet Benchmark
                        conv_net = ConvNetWithGlobalPooling(channel_in=3).to(
                            device
                        )
                        conv_weights = (
                            torch.randn(B, 64, C, 3, 3).to(device),
                            torch.randn(B, 128, 64, 3, 3).to(device),
                            torch.randn(B, 256, 128, 3, 3).to(device),
                            torch.randn(B, 256, num_classes).to(device),
                        )
                        conv_biases = (
                            torch.randn(B, 64).to(device),
                            torch.randn(B, 128).to(device),
                            torch.randn(B, 256).to(device),
                            torch.randn(B, num_classes).to(device),
                        )
                        conv_net_time = benchmark(
                            conv_net, x, conv_weights, conv_biases, y=y
                        )
                        wandb.log(
                            {
                                "ConvNet with Global Pooling Benchmark Time": conv_net_time
                            }
                        )

                        # 4-layer MLP Benchmark
                        mlp_net = FourLayerMLP().to(device)
                        mlp_weights = (
                            torch.randn(B, C * H * W, 512).to(device),
                            torch.randn(B, 512, 256).to(device),
                            torch.randn(B, 256, 128).to(device),
                            torch.randn(B, 128, num_classes).to(device),
                        )
                        mlp_biases = (
                            torch.randn(B, 512).to(device),
                            torch.randn(B, 256).to(device),
                            torch.randn(B, 128).to(device),
                            torch.randn(B, num_classes).to(device),
                        )
                        mlp_net_time = benchmark(
                            mlp_net, x, mlp_weights, mlp_biases, y=y
                        )
                        wandb.log({"4-layer MLP Benchmark Time": mlp_net_time})
                        wandb.log({"num_models": B, "batch_size": N})
                    except Exception as e:
                        # Log the exception to the console
                        print(
                            f"Error during benchmark with B={B} and N={N}: {e}"
                        )
                        # Log the exception to wandb
                        wandb.log({"error": str(e)})


if __name__ == "__main__":
    main()
