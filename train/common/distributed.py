"""Explicit distributed reductions used by the original training recipes."""
import os
import torch

class Context:
    def __init__(self, device):
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.device = device
        if 'RANK' in os.environ:
            import torch.distributed as dist
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            if device.startswith('cuda'):
                torch.cuda.set_device(local_rank)
                self.device = f'cuda:{local_rank}'
            from datetime import timedelta
            timeout = timedelta(minutes=int(os.environ.get('DIST_TIMEOUT_MINUTES', 60)))
            dist.init_process_group('nccl' if device.startswith('cuda') else 'gloo', timeout=timeout)
        self.primary = self.rank == 0

    @property
    def distributed(self):
        return self.world_size > 1

    def sum(self, value):
        if not self.distributed:
            return value
        import torch.distributed as dist
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor)
        return tensor.item()

    def gather(self, value):
        if not self.distributed:
            return [value]
        import torch.distributed as dist
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def on_primary(self, function):
        error = None
        if self.primary:
            try:
                function()
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        if self.distributed:
            import torch.distributed as dist
            payload = [error]
            dist.broadcast_object_list(payload, src=0)
            error = payload[0]
        if error is not None:
            raise RuntimeError(error)

    def broadcast(self, parameters):
        if self.distributed:
            import torch.distributed as dist
            for parameter in parameters:
                dist.broadcast(parameter.data, src=0)

    def mean_gradients(self, parameters, global_questions):
        if global_questions <= 0:
            raise ValueError('No labeled questions in update')
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            if self.distributed:
                import torch.distributed as dist
                dist.all_reduce(parameter.grad)
            parameter.grad.div_(global_questions)

    def agree_step(self, want_step):
        if not self.distributed:
            return want_step
        import torch.distributed as dist
        flag = torch.tensor([int(want_step)], device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def average_rank_gradients(self, parameters):
        # SFT-2B divides by world size, not by the number of questions.
        if not self.distributed:
            return
        import torch.distributed as dist
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(self.world_size)

    def close(self):
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()

