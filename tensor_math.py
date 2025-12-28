import torch
import numpy as np

# Basic tensor math operations

tensor_a = torch.tensor([1, 2, 3, 4])
tensor_b = torch.tensor([5, 6, 7, 8])


# Addition
print("Adding tensor_a and tensor_b with actual arithmetic xxxxxxxxx.")
print(tensor_a + tensor_b)


# Addition Longhand result same as above
print("Addition 'Longhand' but clean syntax xxxxxx.")
print(torch.add(tensor_a, tensor_b))


# Subtraction
print("Here's subtraction with Pytorch xxxxxx.")
print(tensor_b - tensor_a)


# Multiplication
print("Here's multiplication with tensors xxxxxx.")
print(tensor_a * tensor_b)


# Multiplication Longhand
print("Multiplication longhand xxxxxxx")
print(torch.mul(tensor_a, tensor_b))


# Division with tensors
print("Here's divison with tensors which depending on modulus xxxxxxx")
print(tensor_b / tensor_a)
print("Here's division long hand xxxx")
print(tensor_b, tensor_a)
print(torch.div(tensor_b, tensor_a))


# Remainder Modulus
print("Here's the modulus programming xxxxxx")
print(tensor_b, tensor_a)
print(tensor_b % tensor_a)

# Remainer longhand
print("This is remainder longhand xxxxxx")
print(torch.remainder(tensor_b, tensor_a))

# Exponents / power
print("Here's Pytorch exponents xxxxxx")
print(torch.pow(tensor_a, tensor_b))
