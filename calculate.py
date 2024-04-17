import torch
import torch_ext

A = torch.rand(1024,1024)
B = torch.rand(1024,1024)

GPU_C = torch_ext.MatrixMul(A,B)
print("GPU: ")
print(GPU_C)

CPU_C = torch.mm(A,B)
print("CPU: ")
print(CPU_C)

print("GPU_C == CPU_C: ")
print(torch.allclose(GPU_C,CPU_C))
