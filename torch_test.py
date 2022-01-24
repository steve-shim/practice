import torch

x = torch.tensor([2,1]) # 입력 (1x2)
w1 = torch.tensor([[3,2,-4],[2,-3,1]]) # (2,3) -> 입력x출력
b1 = 1
w2 = torch.tensor([[-1,1],[1,2],[3,1]]) # (3,2) -> 입력x출력
b2 = -1

h_preact = torch.matmul(x, w1) + b1
h= torch.nn.functional.relu(h_preact)
y = torch.matmul(h,w2) + b2

print(h_preact) # tensor([ 9,  2, -6]) 선형회귀 출력!
print(h)        # tensor([9, 2, 0]) 은닉층 출력!
print(y)        # tensor([-8, 12]) 출력층 출력!
