use burn::prelude::*;
use burn::module::Ignored;

#[derive(Config, Debug, Clone, PartialEq)]
pub struct MyConfig {
    pub a: usize,
}

#[derive(Module, Debug)]
pub struct MyModule<B: Backend> {
    pub config: Ignored<MyConfig>,
}
