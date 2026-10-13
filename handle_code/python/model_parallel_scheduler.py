import torch
from torch import nn


class ModelParallelScheduler(nn.Module):
    """单进程流水线前向调度：每个 stage 放到指定设备。

    stage 接收并返回一个 Tensor，batch 维为第 0 维。
    这里只展示 micro-batch 的调度顺序，不实现通信与计算重叠或反向调度。
    """

    def __init__(self, stages, devices):
        super().__init__()
        if not stages or len(stages) != len(devices):
            raise ValueError("每个 stage 必须对应一个 device")
        self.devices = [torch.device(device) for device in devices]
        self.stages = nn.ModuleList([
            stage.to(device) for stage, device in zip(stages, self.devices)
        ])

    def forward(self, x, micro_batch_size=1):
        if micro_batch_size <= 0 or x.shape[0] == 0:
            raise ValueError("batch 和 micro_batch_size 必须大于 0")
        micro_batches = list(x.split(micro_batch_size, dim=0))
        num_batches = len(micro_batches)
        num_stages = len(self.stages)

        # buffers[s][m] 保存第 m 个 micro-batch 进入 stage s 的输入。
        buffers = [dict() for _ in range(num_stages + 1)]
        buffers[0] = dict(enumerate(micro_batches))

        # 例如两个 stage、三个 micro-batch：
        # tick 0: S0(M0)
        # tick 1: S1(M0), S0(M1)
        # tick 2: S1(M1), S0(M2)
        # tick 3: S1(M2)
        for tick in range(num_batches + num_stages - 1):
            for s in reversed(range(num_stages)):
                m = tick - s
                if 0 <= m < num_batches:
                    stage_input = buffers[s].pop(m).to(self.devices[s])
                    buffers[s + 1][m] = self.stages[s](stage_input)

        return torch.cat([buffers[-1][m] for m in range(num_batches)], dim=0)


# 使用示例：
# model = ModelParallelScheduler(
#     [nn.Linear(8, 16), nn.Linear(16, 4)], ["cuda:0", "cuda:1"]
# )
# output = model(torch.randn(6, 8), micro_batch_size=2)
