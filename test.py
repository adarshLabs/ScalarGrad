from value import Value

a = Value(2)
b = a*a + a*3
print(a, b)
b.backward()
print(a, b)