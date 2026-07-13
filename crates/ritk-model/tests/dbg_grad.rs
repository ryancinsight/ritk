use coeus_autograd::{matmul, permute, reshape, scalar_mul, softmax, sum, transpose, Var};
use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;

type B = SequentialBackend;

fn var(shape: &[usize], data: &[f32], grad: bool) -> Var<f32, B> {
    Var::new(Tensor::from_slice_on(shape, data, &SequentialBackend), grad)
}
fn ramp(n: usize) -> Vec<f32> { (0..n).map(|i| ((i%17) as f32)/17.0 - 0.5).collect() }

fn fd_check(name: &str, shape: &[usize], base: &[f32], probes: &[usize], f: impl Fn(&Var<f32,B>)->Var<f32,B>) {
    let input = var(shape, base, true);
    let loss = sum(&f(&input));
    loss.backward();
    let grad = input.grad().unwrap();
    let grad = grad.as_slice();
    let h = 1.0f32/128.0;
    let mut maxa=0.0f32;
    for &idx in probes {
        let mut p=base.to_vec(); let mut m=base.to_vec(); p[idx]+=h; m[idx]-=h;
        let fp=sum(&f(&var(shape,&p,false))); let fm=sum(&f(&var(shape,&m,false)));
        let fd=(fp.tensor.as_slice()[0]-fm.tensor.as_slice()[0])/(2.0*h);
        let a=grad[idx]; maxa=maxa.max(a.abs());
        let d=(fd-a).abs(); let tol=2e-2*(1.0+a.abs());
        println!("{name}[{idx}] fd={fd} auto={a} d={d} tol={tol} {}", if d<=tol {"OK"} else {"FAIL"});
    }
    println!("{name} maxa={maxa}");
}

#[test]
fn dbg_branch_accum() {
    // x used twice: reshape branch add
    fd_check("branch", &[1,4], &ramp(4), &[0,1], |x| {
        let a = scalar_mul(x, 2.0);
        let b = scalar_mul(x, 3.0);
        coeus_autograd::add(&a,&b)
    });
}

#[test]
fn dbg_matmul4d() {
    // [1,2,3,4] x [1,2,4,3] -> [1,2,3,3]
    let bshape=[1,2,4,3];
    let bdata=ramp(24);
    let bv=var(&bshape,&bdata,false);
    fd_check("mm4d", &[1,2,3,4], &ramp(24), &[0,7,15], |x| matmul(x,&bv));
}

#[test]
fn dbg_transpose_mm() {
    // q[1,2,3,4], k[1,2,3,4]; attn=matmul(q, transpose(k,2,3)) -> [1,2,3,3]; grad wrt k
    let qshape=[1,2,3,4]; let qdata=ramp(24); let qv=var(&qshape,&qdata,false);
    fd_check("tmm_k", &[1,2,3,4], &ramp(24), &[0,7,15], |k| matmul(&qv, &transpose(k,2,3)));
}

#[test]
fn dbg_softmax4d() {
    fd_check("sm4d", &[1,2,3,3], &ramp(18), &[0,4,8], |x| softmax(x, -1));
}

#[test]
fn dbg_perm_reshape() {
    // reshape+permute roundtrip then sum-of-squares-ish via matmul with self? just permute+reshape
    fd_check("permresh", &[1,3,2,4], &ramp(24), &[0,10,23], |x| {
        let y = permute(&reshape(x,[1,3,2,4]), &[0,2,1,3]);
        reshape(&y,[1,2,3,4])
    });
}

use coeus_autograd::{add, index_select};
use coeus_nn::module::Module;
use coeus_nn::Linear;

fn mk_lin(inp: usize, out: usize, seed: u64) -> Linear<f32, B> {
    let mut l = Linear::new(inp, out, true);
    coeus_nn::init::kaiming_uniform_with_seed(&mut l.weight, inp, seed);
    l
}

#[test]
fn dbg_attention_stages() {
    let (n, c, nh, hd) = (8usize, 4usize, 2usize, 2usize);
    let q = mk_lin(c, c, 1);
    let k = mk_lin(c, c, 2);
    let v = mk_lin(c, c, 3);
    let proj = mk_lin(c, c, 4);
    let scale = (hd as f64).powf(-0.5) as f32;

    // bias table + index
    let m = 2usize; let ndist = (2*m-1).pow(3);
    let table = {
        let mut t = Var::new(Tensor::zeros_on([ndist, nh], &SequentialBackend), true);
        coeus_nn::init::normal_with_seed(&mut t, 0.0, 0.02, 9);
        t
    };
    let mut idxv = Vec::new();
    let mut coords=Vec::new();
    for d in 0..m { for h in 0..m { for w in 0..m { coords.push((d as i32,h as i32,w as i32)); }}}
    let range=2*m as i32-1;
    for &(d1,h1,w1) in coords.iter() { for &(d2,h2,w2) in coords.iter() {
        let rd=(d1-d2)+(m as i32-1); let rh=(h1-h2)+(m as i32-1); let rw=(w1-w2)+(m as i32-1);
        idxv.push((rd*range*range+rh*range+rw) as f32);
    }}
    let index = Var::new(Tensor::from_slice_on([idxv.len()], &idxv, &SequentialBackend), false);

    let build = |x: &Var<f32,B>, use_bias: bool, use_proj: bool| {
        let proj_fn = |lin: &Linear<f32,B>| {
            let y = lin.forward(x);
            permute(&reshape(&y,[1,n,nh,hd]), &[0,2,1,3])
        };
        let qq=proj_fn(&q); let kk=proj_fn(&k); let vv=proj_fn(&v);
        let attn = scalar_mul(&matmul(&qq,&transpose(&kk,2,3)), scale);
        let attn = if use_bias {
            let bias = index_select(&table, 0, &index);
            let bias = permute(&reshape(&bias,[n,n,nh]), &[2,0,1]);
            let bias = reshape(&bias,[1,nh,n,n]);
            add(&attn,&bias)
        } else { attn };
        let attn = softmax(&attn,-1);
        let out = matmul(&attn,&vv);
        let out = reshape(&permute(&out,&[0,2,1,3]),[1,n,c]);
        if use_proj { proj.forward(&out) } else { out }
    };

    let base = ramp(n*c);
    fd_check("attn_nobias_noproj", &[1,n,c], &base, &[0,11,27], |x| build(x,false,false));
    fd_check("attn_bias_noproj",   &[1,n,c], &base, &[0,11,27], |x| build(x,true,false));
    fd_check("attn_bias_proj",     &[1,n,c], &base, &[0,11,27], |x| build(x,true,true));
}

#[test]
fn dbg_linear_rank3() {
    let l = mk_lin(4, 4, 7);
    // rank-3 input [1,8,4], FD wrt input
    fd_check("lin_r3", &[1,8,4], &ramp(32), &[0,11,27], |x| l.forward(x));
}

#[test]
fn dbg_linear_rank2() {
    let l = mk_lin(4, 4, 7);
    fd_check("lin_r2", &[8,4], &ramp(32), &[0,11,27], |x| l.forward(x));
}

#[test]
fn dbg_linear_chain_r3() {
    let a = mk_lin(4,4,7); let b2 = mk_lin(4,4,8);
    // reshape/permute in between like attention out path
    fd_check("lin_chain", &[1,8,4], &ramp(32), &[0,11,27], |x| {
        let y = a.forward(x);
        let y = permute(&reshape(&y,[1,8,2,2]), &[0,2,1,3]);
        let y = reshape(&permute(&y,&[0,2,1,3]),[1,8,4]);
        b2.forward(&y)
    });
}
