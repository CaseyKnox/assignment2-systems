from cs336_basics import model, optimizer
import timeit
import torch

if __name__ == "__main__":
    # Defaults
    batch_size = 4
    warmup_steps = 10
    test_runs = 10
    device="cuda" if torch.cuda.is_available() else "cpu"

    # Create an instance of the BasicsTransformerLM class
    model = model.BasicsTransformerLM(
        vocab_size=10000,
        context_length=512,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
    ).to(device)
    optim = optimizer.AdamW(model.parameters())

    # Define a sample input for benchmarking
    sample_input = torch.randint(0, 10000, (batch_size, 10)).to(device)

    # Warmup
    for _ in range(warmup_steps):
        _ = model.forward(sample_input.to(device))

    def _time_forward_pass() -> float:
        start = timeit.default_timer()
        _ = model.forward(sample_input.to(device))
        torch.cuda.synchronize(device)
        end = timeit.default_timer()
        return end - start

    def _time_forward_backward() -> float:
        start = timeit.default_timer()
        logits = model.forward(sample_input.to(device))
        loss = logits.mean()  # Dummy loss for backward pass
        loss.backward()
        torch.cuda.synchronize(device)
        end = timeit.default_timer()
        return end - start

    def _time_forward_backward_optim():
        start = timeit.default_timer()
        logits = model.forward(sample_input.to(device))
        loss = logits.mean()  # Dummy loss for backward pass
        loss.backward()

        optim.step()
        optim.zero_grad()
        torch.cuda.synchronize(device)
        end = timeit.default_timer()
        return end - start

    forward_pass_times = []
    forward_backward_times = []
    forward_backward_optim_times = []
    for test_run in range(test_runs):
        forward_pass_times.append(_time_forward_pass())
        forward_backward_times.append(_time_forward_backward())
        forward_backward_optim_times.append(_time_forward_backward_optim())

    forward_mean = sum(forward_pass_times) / test_runs
    forward_backward_mean = sum(forward_backward_times) / test_runs
    forward_backward_optim_mean = sum(forward_backward_optim_times) / test_runs
    forward_std = (sum((x - forward_mean) ** 2 for x in forward_pass_times) / test_runs) ** 0.5
    forward_backward_std = (sum((x - forward_backward_mean) ** 2 for x in forward_backward_times) / test_runs) ** 0.5
    forward_backward_optim_std = (sum((x - forward_backward_optim_mean) ** 2 for x in forward_backward_optim_times) / test_runs) ** 0.5

    print(f"Forward pass time: {forward_mean*1e3:.2f} ± {forward_std*1e3:.2f} ms")
    print(f"Forward + backward pass time: {forward_backward_mean*1e3:.2f} ± {forward_backward_std*1e3:.2f} ms")
    print(f"Forward + backward + optimizer step time: {forward_backward_optim_mean*1e3:.2f} ± {forward_backward_optim_std*1e3:.2f} ms")