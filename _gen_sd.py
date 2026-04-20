import base64, os

# Decode the main section from the b64 file
with open(chr(95)+chr(115)+chr(100)+chr(95)+chr(109)+chr(97)+chr(105)+chr(110)+chr(46)+chr(98)+chr(54)+chr(52)) as f:
    main_b64 = f.read().strip()
main = base64.b64decode(main_b64).decode(chr(117)+chr(116)+chr(102)+chr(45)+chr(56))

# Now build tests section
Q = chr(34)
NL = chr(10)
L = []
