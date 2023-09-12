# Lil grad

Lil grad is a simple and small autograd implementation and an extension on Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd). Instead of operating only on scalars it can also operate on Tensors. In addition to simple element-wise operations like addition, subtraction, multiplication, etc., it is also able to process matrix multiplications both in forward and backward operation.

## broadcasting

Currently Lil grad is only able to process the gradients for scalars/single element lists which are broadcasted. Other broadcasted shapes will raise an error.
