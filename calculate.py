import torch
import torch_ext

A = torch.tensor([[1.0,2.0],[2.0,3.0]])
B = torch.tensor([[1.0,2.0],[3.0,2.0]])

GPU_C = torch_ext.MatrixMul(A,B)
print("GPU: ")
print(GPU_C)

CPU_C = torch.mm(A,B)
print("CPU: ")
print(CPU_C)

print("GPU_C == CPU_C: ")
print(torch.equal(GPU_C,CPU_C))
