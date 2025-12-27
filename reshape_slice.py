import torch
my_torch = torch.arange(10)
print("Here's my_torch xxxxx")
print(my_torch)

# Reshape and View
my_torch = my_torch.reshape(2, 5)
print("Printing my my_torch reshape tensor")
print(my_torch)

# Reshape if we don't know the number of items using -1
my_torch2 = torch.arange(15)
print("Here's my torch2 xxxxx")
print(my_torch2)

# If you don't know the number of items in the tensor then use -1 to reshape
# But the division of numbers need to make sense a tensor with 15 items won't work
# due to it's not divisible by (2, -1) but (3, -1) will due to 15 is divisble by 3
my_torch2 = my_torch2.reshape(2, -1)
print("Here's my_torch I'm not sure how may reshaping")
print(my_torch2)
