import os

files = [
    r'D:\ritk\crates\ritk-core\src\filter\discrete_gaussian.rs',
    r'D:\ritk\crates\ritk-core\src\filter\diffusion\curvature.rs',
    r'D:\ritk\crates\ritk-core\src\filter\edge\gradient_magnitude.rs',
    r'D:\ritk\crates\ritk-core\src\interpolation\sinc.rs',
    r'D:\ritk\crates\ritk-core\src\segmentation\level_set\helpers.rs',
    r'D:\ritk\crates\ritk-core\src\segmentation\level_set\geodesic_active_contour.rs',
    r'D:\ritk\crates\ritk-core\src\segmentation\watershed\marker_controlled.rs'
]

for source in files:
    with open(source, 'r', encoding='utf-8') as f:
        content = f.read()

    test_marker = '#[cfg(test)]\nmod tests {'
    parts = content.split(test_marker)
    if len(parts) == 2:
        main_code = parts[0].strip()
        
        # the test code has a trailing `}` that we need to remove
        test_code_body = parts[1].strip()
        if test_code_body.endswith('}'):
            test_code_body = test_code_body[:-1].strip()
            
        mod_name = os.path.basename(source).replace('.rs', '')
        dir_name = os.path.dirname(source)
        test_filename = f'tests_{mod_name}.rs'
        test_path = os.path.join(dir_name, test_filename)
        
        # Replace the test module with the #[path] mod tests; statement
        main_code += f'\n\n#[cfg(test)]\n#[path = "{test_filename}"]\nmod tests;\n'
        
        with open(source, 'w', encoding='utf-8') as f:
            f.write(main_code)
            
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_code_body)
            
        print(f'Split {mod_name}.rs successfully via #[path]')
