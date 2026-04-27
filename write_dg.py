import pathlib, textwrap

dg_content = textwrap.dedent('''
dummy placeholder
''').strip()

pathlib.Path(r'crates/ritk-core/src/filter/discrete_gaussian.rs').write_text('placeholder\n', encoding='utf-8')
print('ok')
