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

my_torch2 = my_torch2.reshape(3, -1)
print("Here's my_torch I'm not sure how may reshaping")
print(my_torch2)

# You can also do the same in the opposite direction
my_torch2 = my_torch2.reshape(-1, 3)
print("Here's my_torch2 in the opposite direction xxxxxx")
print(my_torch2)

# Here's a lesson on views
my_torch3 = torch.arange(10)
my_torch4 = my_torch3.view(2, 5)
print("Here's my my_torch4 with a view xxxxxx")
print(my_torch4)

# Differences between shape and view
# with reshape and view, they will update
my_torch5 = torch.arange(10)
my_torch6 = my_torch5.reshape(2, 5)
print("Here's my_torch6 which is my reshaped my_torch5 xxxxxx")
print(my_torch6)
my_torch5[1] = 4141
print("Here's my new entry to the original tensor my_torch5 with 4141")
print(my_torch5)
print("Here's my new my_torch6 xxxxxxx")
print(my_torch6)

# Slicing
my_torch7 = torch.arange(10)
print("Here's my_torch7 sliced xxxxxxxx")
print(my_torch7[5])
my_torch8 = my_torch7.reshape(5,2)
print("Here's my_torch8 xxxxxx")
print(my_torch8)
print("Here's my_torch8 grabbing only the second column xxxxxxx")
print(my_torch8[:,1:])
