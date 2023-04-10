use serde::{Serialize, Deserialize};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Activation {
    Tanh,
    ReLU,
    LeakyReLU(f32),
    ParametricReLU(f32),
    ELU(f32),
    Swish(f32),
    // Add other activation functions if desired
}

impl Activation {
    pub fn compute(&self, x: f32) -> f32 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU(alpha) => x.max(alpha * x),
            Activation::ParametricReLU(alpha) => x.max(alpha * x),
            Activation::ELU(alpha) => if x > 0.0 { x } else { alpha * (x.exp() - 1.0) },
            Activation::Swish(beta) => x * (beta * x).sigmoid(),
            // Add other activation function computations if desired
        }
    }

    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::LeakyReLU(alpha) => if x > 0.0 { 1.0 } else { *alpha },
            Activation::ParametricReLU(alpha) => if x > 0.0 { 1.0 } else { *alpha },
            Activation::ELU(alpha) => if x > 0.0 { 1.0 } else { alpha * x.exp() },
            Activation::Swish(beta) => {
                let sigmoid = (beta * x).sigmoid();
                sigmoid + beta * x * (1.0 - sigmoid)
            }
            // Add other activation function derivative computations if desired
        }
    }
}
trait FloatSigmoid {
    fn sigmoid(self) -> Self;
}

impl FloatSigmoid for f32 {
    fn sigmoid(self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }
}
