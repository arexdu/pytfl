def fibonaccigenerator(all_num):
    a, b = 0, 1
    current_num = 0
    while current_num < all_num:
        yield a
        a, b = b, a+b
        current_num += 1

for num in fibonaccigenerator(100):
    print(num)
