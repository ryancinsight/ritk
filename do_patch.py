# do_patch.py - auto-generated patch script
import re

DQ = chr(34)
BS = chr(92)
NL = chr(10)
BT = chr(96)
ARROW = BS + chr(117) + chr(123) + chr(50) + chr(49) + chr(57) + chr(50) + chr(125)
ALPHA = BS + chr(117) + chr(123) + chr(48) + chr(51) + chr(98) + chr(49) + chr(125)
LEQ = chr(0x2264)
DELTA = chr(0x0394)
PARTIAL = chr(0x2202)
NABLA = chr(0x2207)
DOT = chr(0x00b7)
BOX = chr(0x2500)
TIMES = chr(0x00d7)
SIGMA = chr(963)
PLUSMINUS = chr(0x00b1)

# ------- FILTER.RS -------
fp = 'crates/ritk-cli/src/commands/filter.rs'
with open(fp, 'r', encoding='utf-8') as fh:
    fsrc = fh.read()

