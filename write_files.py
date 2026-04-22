import pathlib
p = pathlib.Path("test_output.txt")
p.write_text("hello world")
print("ok")
