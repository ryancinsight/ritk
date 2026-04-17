with open(r'D:\ritk\crates\ritk-cli\src\commands\filter.rs', 'r', encoding='utf-8') as f:
    c = f.read()

# 6. Add run_curvature and run_sato before run_gradient_magnitude
sep = '\u2500'
run_cur = (
    'fn run_curvature(args: &FilterArgs) -> Result<()> {\n'
    '    use ritk_core::filter::diffusion::{CurvatureAnisotropicDiffusionFilter, CurvatureConfig};\n\n'
    '    let image = read_image(&args.input)?;\n'
    '    let config = CurvatureConfig {\n'
    '        num_iterations: args.iterations,\n'
    '        time_step: args.time_step as f32,\n'
    '    };\n'
    '    let filter = CurvatureAnisotropicDiffusionFilter::new(config);\n'
    '    let filtered = filter.apply(&image)?;\n\n'
    '    write_image_inferred(&args.output, &filtered)?;\n\n'
    '    println!(\n'
    '        "Applied curvature-diffusion (iterations={}, dt={}) to {} \u2192 {}",\n'
    '        args.iterations,\n'
    '        args.time_step,\n'
    '        args.input.display(),\n'
    '        args.output.display(),\n'
    '    );\n'
    '    info!(\n'
    '        input = %args.input.display(),\n'
    '        output = %args.output.display(),\n'
    '        iterations = args.iterations,\n'
    '        time_step = args.time_step,\n'
    '        "filter: curvature complete"\n'
    '    );\n'
    '    Ok(())\n'
    '}\n\n'
)
q = chr(39)
run_sato = (
    'fn run_sato(args: &FilterArgs) -> Result<()> {\n'
    '    use ritk_core::filter::vesselness::{SatoConfig, SatoLineFilter};\n\n'
    '    let image = read_image(&args.input)?;\n\n'
    '    let scales: Vec<f64> = args\n'
    '        .scales\n'
    '        .split(' + q + ',' + q + ')\n'
    '        .filter_map(|s| s.trim().parse::<f64>().ok())\n'
    '        .collect();\n'
    '    let scales = if scales.is_empty() { vec![1.0, 2.0, 3.0] } else { scales };\n\n'
    '    let config = SatoConfig { scales: scales.clone(), alpha: args.alpha, bright_tubes: true };\n'
    '    let filter = SatoLineFilter::new(config);\n'
    '    let filtered = filter.apply(&image)?;\n\n'
    '    write_image_inferred(&args.output, &filtered)?;\n\n'
    '    println!(\n'
    '        "Applied sato (scales={:?}, \u03b1={}) to {} \u2192 {}",\n'
    '        scales, args.alpha, args.input.display(), args.output.display(),\n'
    '    );\n'
    '    info!(\n'
    '        input = %args.input.display(),\n'
    '        output = %args.output.display(),\n'
    '        alpha = args.alpha,\n'
    '        "filter: sato complete"\n'
    '    );\n'
    '    Ok(())\n'
    '}\n\n'
)
old_grad = ('// ' + sep*70 + '\n\nfn run_gradient_magnitude')
assert old_grad in c, repr(old_grad[:60])
c = c.replace(old_grad, run_cur + run_sato + old_grad, 1)

with open(r'D:\ritk\crates\ritk-cli\src\commands\filter.rs', 'w', encoding='utf-8') as f:
    f.write(c)
print("filter.rs part 2 done")
