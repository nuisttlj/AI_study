import torch

data = torch.Tensor([[3, 4], [5, 6], [7, 8]])
label = torch.Tensor([1, 1, 0])
center = torch.Tensor([[1, 1], [2, 2]])

center_exp = center.index_select(dim=0, index=label.long())
print(center_exp)

count = torch.histc(label, bins=2, min=0, max=1)
print(count, "...........")

count_dis = count.index_select(dim=0, index=label.long())
print(count_dis)

loss = torch.sum(torch.sqrt(torch.sum((data - center_exp) ** 2, dim=1)) / count_dis)
print(loss)
# print(ds.shape)
