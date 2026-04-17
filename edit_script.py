with open('check.txt', 'w') as f:
    s = '\\n'
    f.write(repr(s) + '\n')
    f.write(str([ord(c) for c in s]) + '\n')
