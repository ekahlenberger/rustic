use serde::{Serialize, Deserialize};
use crate::layer::Layer;
use crate::activation::Activation;



#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(sizes: &[usize], activations: &[Activation]) -> Self {
        assert_eq!(sizes.len() - 1, activations.len(), "Number of activations should be one less than the number of layer sizes");
    
        let layers: Vec<Layer> = sizes.windows(2)
            .zip(activations.iter())
            .map(|(window, &activation)| Layer::new(window[0], window[1], activation))
            .collect();
    
        Self { layers }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.layers.iter().fold(input.to_vec(), |input, layer| layer.forward(&input))
    }
    // returns the network output and the activations (outputs) of each layer
    pub fn forward_activations(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let mut activations = Vec::new();
        let _ = self.layers.iter().fold(input.to_vec(), |input, layer| {
            let layer_output = layer.forward(&input);
            activations.push(layer_output.clone());
            layer_output // becomes the input for the next layer
        });
        activations
    }

    pub fn backpropagate(&mut self, input: &[f32], target: &[f32], learning_rate: f32) {
        let activations = self.forward_activations(input);
        
        // Compute the errors for the output layer
        let mut errors = target.iter()
            .zip(activations.last().unwrap().iter())
            .map(|(t, o)| t - o)
            .collect::<Vec<_>>();

        // Prepend input to activations to use it as the input for the first layer
        let mut activations_with_input = vec![input.to_vec()];
        activations_with_input.extend(activations);
    
        // Iterate through each layer in reverse order
        for (layer, layer_activations) in self.layers.iter_mut().zip(activations_with_input.iter()).rev() {
            layer.update_weights_and_biases(layer_activations, &errors, learning_rate);
            
            // Compute the errors for the next layer
            errors = layer.weights
                .iter()
                .map(|neuron_weights| neuron_weights.iter().zip(errors.iter()).map(|(w, e)| w * e).sum())
                .collect();
        }
    }
}