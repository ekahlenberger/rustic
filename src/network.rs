use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NeuralNetwork {
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = (0..input_size)
            .map(|_| vec![rand::random::<f32>(); output_size])
            .collect();
        let biases = vec![rand::random::<f32>(); output_size];
        Self { weights, biases }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; self.biases.len()];
        for (output_neuron_index, bias) in self.biases.iter().enumerate() {
            let weight_sum = self.weights.iter()
                .map(|w| w[output_neuron_index] * input[output_neuron_index])
                .sum::<f32>();
            result[output_neuron_index] = (weight_sum + bias).tanh();
        }
        result
    }
    const L1_REGULARIZATION: f32 = 0.001;
    const L2_REGULARIZATION: f32 = 0.001;
    pub fn update_weights_and_biases(&mut self, state: &[f32], target_q_values: &[f32], learning_rate: f32) {
        let forward_result = self.forward(state); // Call forward once and store the result
        
        // Calculate the deltas first
        let deltas: Vec<f32> = target_q_values
            .iter()
            .zip(forward_result.iter())
            .map(|(target, forward)| learning_rate * (target - forward))
            .collect();
        
        // Clip the deltas to avoid exploding gradients
        let clip_threshold = 1.0;
        let clipped_deltas: Vec<f32> = deltas
            .iter()
            .map(|delta| delta.min(clip_threshold).max(-clip_threshold))
            .collect();
    
        // Update weights and biases using the clipped deltas
        for (idx, (weight, bias)) in self.weights.iter_mut().zip(self.biases.iter_mut()).enumerate() {
            let delta = clipped_deltas[idx]; // Use the clipped delta from the temporary vector
            for w in weight.iter_mut() {
                let l1_regularization = Self::L1_REGULARIZATION * w.signum();
                let l2_regularization = Self::L2_REGULARIZATION * *w;
                *w += delta - learning_rate * (l1_regularization + l2_regularization);
            }
            let l1_regularization = Self::L1_REGULARIZATION * bias.signum();
            let l2_regularization = Self::L2_REGULARIZATION * *bias;
            *bias = *bias + delta - learning_rate * (l1_regularization + l2_regularization);
        }
    }
}
