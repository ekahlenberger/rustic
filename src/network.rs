use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
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
        for (i, bias) in self.biases.iter().enumerate() {
            let weight_sum = self.weights.iter()
                .map(|w| w[i] * input[i])
                .sum::<f32>();
            result[i] = weight_sum + bias;
        }
        result
    }   

    pub fn update_weights_and_biases(&mut self, state: &[f32], target_q_values: &[f32], learning_rate: f32) {
        let forward_result = self.forward(state); // Call forward once and store the result
        
        // Calculate the deltas first
        let deltas: Vec<f32> = target_q_values
            .iter()
            .zip(forward_result.iter())
            .map(|(target, forward)| learning_rate * (target - forward))
            .collect();
    
        // Update weights and biases using the deltas
        for (idx, (weight, bias)) in self.weights.iter_mut().zip(self.biases.iter_mut()).enumerate() {
            let delta = deltas[idx]; // Use the delta from the temporary vector
            for w in weight.iter_mut() {
                *w += delta;
            }
            *bias = *bias + delta;
        }
    }
}
