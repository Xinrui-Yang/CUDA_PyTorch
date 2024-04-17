#include <torch/extension.h>

// CUDA declaration
void MatrixMul_cuda(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B);

// C++ interface
torch::Tensor MatrixMul(torch::Tensor A, torch::Tensor B){
    // check input
    TORCH_CHECK(A.size(1) == B.size(0));
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2);

    auto C = torch::empty({A.size(0), B.size(1)}, A.options());

    // cuda kernel
    MatrixMul_cuda(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), A.size(0), A.size(1), B.size(1));

    return C;
}

// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("MatrixMul", &MatrixMul, "Matrix multiplication (CUDA)");
}

