import os
import re

source = r'D:\ritk\crates\ritk-core\src\filter\intensity\arithmetic.rs'
out_dir = r'D:\ritk\crates\ritk-core\src\filter\intensity\arithmetic'

with open(source, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# find indices of sections
sections = []
for i, line in enumerate(lines):
    if line.startswith('// ── ') and '──' in line:
        name = line.strip().replace('// ── ', '').replace(' ──', '').replace('─', '').strip()
        sections.append((i, name))

sections.append((len(lines), 'EOF'))

blocks = {}
for i in range(len(sections)-1):
    start_idx = sections[i][0]
    end_idx = sections[i+1][0]
    name = sections[i][1]
    blocks[name] = "".join(lines[start_idx:end_idx])

# The header is lines before the first section
header = "".join(lines[0:sections[0][0]])

file_map = {
    'AbsImageFilter': 'abs.rs',
    'InvertIntensityFilter': 'invert.rs',
    'NormalizeImageFilter': 'normalize.rs',
    'SquareImageFilter': 'square.rs',
    'SqrtImageFilter': 'sqrt.rs',
    'LogImageFilter': 'log.rs',
    'ExpImageFilter': 'exp.rs',
}

os.makedirs(out_dir, exist_ok=True)

# Write mod.rs
mod_content = header.strip() + '\n\n'
for name, fname in file_map.items():
    mod_name = fname.replace('.rs', '')
    mod_content += f'pub mod {mod_name};\n'
    
mod_content += '\n'
for name, fname in file_map.items():
    mod_name = fname.replace('.rs', '')
    mod_content += f'pub use {mod_name}::{name};\n'

with open(os.path.join(out_dir, 'mod.rs'), 'w', encoding='utf-8') as f:
    f.write(mod_content)

# Extract tests from the Tests block
tests_block = blocks.get('Tests', '')
test_lines = tests_block.split('\n')
test_sections = []
for i, line in enumerate(test_lines):
    if '    // ── ' in line:
        name = line.strip().replace('// ── ', '').replace(' ──', '').replace('─', '').strip()
        test_sections.append((i, name))
test_sections.append((len(test_lines)-2, 'EOF')) # exclude closing brace of mod tests

test_blocks = {}
for i in range(len(test_sections)-1):
    start_idx = test_sections[i][0]
    end_idx = test_sections[i+1][0]
    name = test_sections[i][1]
    test_blocks[name] = "\n".join(test_lines[start_idx:end_idx]).strip()

for name, fname in file_map.items():
    body = blocks.get(name, '').strip()
    # remove the `// ── Name ──` line from body
    body_lines = body.split('\n')[1:]
    body = "\n".join(body_lines).strip()
    
    test_code = test_blocks.get(name, '')
    if test_code:
        # remove the `// ── Name ──` line from test_code
        test_code_lines = test_code.split('\n')[1:]
        test_code = "\n".join(test_code_lines).strip()
    
    file_content = f"""use crate::filter::ops::{{extract_vec_infallible as extract_vec, rebuild}};
use crate::image::Image;
use burn::tensor::backend::Backend;

{body}

#[cfg(test)]
mod tests {{
    use super::*;
    use crate::image::Image;
    use crate::spatial::{{Direction, Point, Spacing}};
    use burn::tensor::{{Shape, Tensor, TensorData}};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {{
        let device = Default::default();
        let td = TensorData::new(data, Shape::new(shape));
        let t = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }}

    fn vals(img: &Image<B, 3>) -> Vec<f32> {{
        img.data().clone().into_data().into_vec::<f32>().unwrap()
    }}

{test_code}
}}
"""
    with open(os.path.join(out_dir, fname), 'w', encoding='utf-8') as f:
        f.write(file_content)

os.remove(source)
print("Split successful!")
