use crate::activation::Activation;
use serde::{Serialize, Deserialize};
use rand::{thread_rng, Rng};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Layer {
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
    pub activation: Activation,
}

impl Layer
{
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = thread_rng();
        let weights = (0..input_size)
            .map(|_| (0..output_size).map(|_| rng.gen_range(-1f32 .. 1f32)).collect())
            .collect();
        let biases = (0..output_size).map(|_| rng.gen_range(-1f32 .. 1f32)).collect();
        Self { weights, biases, activation }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; self.biases.len()];
        for (output_neuron_index, bias) in self.biases.iter().enumerate() {
            let weight_sum = input.iter().enumerate()
                .map(|(input_neuron_index, input_value)| {
                    if input_neuron_index >= self.weights.len() || 
                        output_neuron_index >= self.weights[input_neuron_index].len() {
                        let _i = input_neuron_index;
                    }
                    self.weights[input_neuron_index][output_neuron_index] * input_value
                })
                .sum::<f32>();

            result[output_neuron_index] = self.activation.compute(weight_sum + bias);
        }
        result
    }
    const L1_REGULARIZATION: f32 = 0.001;
    const L2_REGULARIZATION: f32 = 0.001;
    pub fn update_weights_and_biases(&mut self, input: &[f32], errors: &[f32], learning_rate: f32) {
        let forward_result = self.forward(input); // Call forward once and store the result
        
        let derivatives: Vec<f32> = forward_result
        .iter()
        .map(|output| self.activation.derivative(*output))
        .collect();

        let deltas: Vec<f32> = errors
            .iter()
            .zip(derivatives.iter())
            .map(|(error, derivative)| error * derivative)
            .collect();
        
        // Clip the deltas to avoid exploding gradients
        let clip_threshold = 1.0;
        let clipped_deltas: Vec<f32> = deltas
            .iter()
            .map(|delta| delta.min(clip_threshold).max(-clip_threshold))
            .collect();

        for output_node_index in 0..clipped_deltas.len(){
            let clipped_delta = clipped_deltas[output_node_index];
            if clipped_delta.abs() < 1e-6 {
                continue;
            }
            let learn_delta = clipped_delta * learning_rate;
            for weights in self.weights.iter_mut(){
                let w = weights[output_node_index];
                let l1_regularization = Self::L1_REGULARIZATION * w.signum();
                let l2_regularization = Self::L2_REGULARIZATION * w;
                
                weights[output_node_index] = w + (learn_delta - learning_rate * (l1_regularization + l2_regularization));
            }
            let bias = self.biases[output_node_index];
            let l1_regularization = Self::L1_REGULARIZATION * bias.signum();
            let l2_regularization = Self::L2_REGULARIZATION * bias;
            self.biases[output_node_index] += learn_delta - learning_rate * (l1_regularization + l2_regularization);

        }
    }
}