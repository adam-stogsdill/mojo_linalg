from tensor import Tensor, TensorShape, TensorSpec, rand
from utils.index import Index
from testing import assert_true, assert_equal
from math import sqrt

# For this library, im choosing float32 by default
fn vector_dot_product(tensor_a: Tensor[DType.float32], tensor_b: Tensor[DType.float32]) -> Float32:
    var total: SIMD[DType.float32, 1] = 0
    for i in range(tensor_a.shape()[0]):
        total = total + (tensor_a[i] * tensor_b[i])
    return total

fn is_matrix(vector: Tensor[DType.float32]) -> Bool:
    return vector.shape().rank() == 2

fn transpose(owned tensor: Tensor[DType.float32]) raises -> Tensor[DType.float32]:
    print(tensor.shape().num_elements())
    if is_matrix(tensor):
        var rows = tensor.shape()[0]
        var columns = tensor.shape()[1]
        var output_tensor = Tensor[DType.float32](TensorSpec(DType.float32, columns, rows))
        for i in range(rows):
            for j in range(columns):
                output_tensor[Index(j, i)] = tensor[Index(i, j)]
        return output_tensor
    else:
        var columns = tensor.shape()[0]
       
        return tensor.reshape(new_shape=TensorShape(columns, 1))

fn cholesky_decomposition(tensor: Tensor[DType.float32]) raises -> Tensor[DType.float32]:
    # As per the Cholesky-Banachiewicz Algorithm
    print(tensor.shape().rank(), not is_matrix(tensor))
    assert_true(is_matrix(tensor), msg='Given Tensor in Cholesky Decomposition is NOT A MATRIX')
    assert_equal(lhs=tensor.shape()[0], rhs=tensor.shape()[1], msg='Tensor in Cholesky Decomposition is NOT SQUARE')
    var n = tensor.shape()[0]
    var L = Tensor[DType.float32](TensorSpec(DType.float32, n, n))
    for i in range(0, n, 1):
        for j in range(0, i+1, 1):
            var s: Float32 = 0
            for k in range(0, j, 1):
                s += L[Index(i,k)] * L[Index(j,k)]

            if i == j:
                L[Index(i,j)] = sqrt(tensor[Index(i,i)]-s)
                #print(tensor[Index(i,i)]-s)
            else:
                L[Index(i,j)] = (1.0 / L[Index(j,j)] * (tensor[Index(i,j)]-  s))
    return L


fn main() raises:
    var a_elements = List[SIMD[DType.float32, 1]](4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0)
    var a_tensor = Tensor[DType.float32](TensorShape(3, 3), a_elements)
    print(a_tensor)
    print(cholesky_decomposition(a_tensor))
    print(transpose(cholesky_decomposition(a_tensor)))