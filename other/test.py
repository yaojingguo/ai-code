import torch
print(torch.tensor([1.0, 2.0]).cuda())
print(torch.cuda.get_arch_list())
