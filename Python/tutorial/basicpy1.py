'''
This is a presentation into the basics of python. 
'''

# This is a comment
'''
This is also a comment.
'''

# This is how you assign values to variables
'''
i = 1
print(i)
'''

# This is how you create functions
# Functions in CS do much more than in math
'''
def function():
    print("Hello World")

function()

'''

# This is a function that takes in two inputs and outputs their sum
'''
def add(x, y):
    return x + y

x = 1
y = 2
sum = add(x,y)
print(sum)
'''


# There are lists/arrays in python
# This is how you 
'''
list = ["cat", "dog", "monkey", "fish"]
print(list)
print(list[0])
print(list[1])
print(list[2])
print(list[3])
print(list[4])
# print(list['cat'])
'''

# There's also lambdas in python, which can be constructed
# and then passed al ways of treating functions like objects
'''
def f(x, y):
    return x + y

sum1 = f
sum2 = lambda a, b : a + b
print(sum1(1, 100))
print(sum2(1, 100))
'''
