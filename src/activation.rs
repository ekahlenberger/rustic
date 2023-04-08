use serde::{Serialize, Deserialize};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Activation {
    Tanh,
    ReLU
    // Add other activation functions if desired
}

impl Activation {
    pub fn compute(&self, x: f32) -> f32 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::ReLU => x.max(0.0),
            // Add other activation function computations if desired
        }
    }

    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            // Add other activation function derivative computations if desired
        }
    }
}