import torch

data = torch.Tensor([[3, 4], [5, 6], [7, 8]])
label = torch.LongTensor([0, 0, 1])
label_ = torch.Tensor([0, 0, 1])
center = torch.Tensor([[1, 1], [2, 2], [3, 3], [4, 4]])

print(data)

center_exp = center.index_select(dim=0, index=label)
print(center_exp)

count = torch.histc(label_, bins=2, min=0, max=1)
print(count)
ccc = count.index_select(dim=0, index=label)
print(ccc)

a = (data - center_exp) ** 2
print(a)
b = torch.sum(a, 1)
print(b)
c = torch.sqrt(b)
print(c)
d = c / ccc
print(d)
f = torch.sum(d)
print(f)

# center_loss =torch.sum(torch.sum((data-center_exp)**2,dim=1)/count)
