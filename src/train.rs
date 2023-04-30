use crate::activation::Activation;
use crate::game::{Board, BOARD_SIZE, empty_board, check_winner, make_move, is_full};
use crate::network::NeuralNetwork;
use rand::{Rng};
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::{BufReader, BufWriter};
// use rayon::prelude::*;
// use std::sync::Mutex;
//use itertools::{Itertools, Either};

const DISCOUNT_FACTOR: f32 = 0.9;
const INITIAL_EPSILON: f32 = 1.0;
const FINAL_EPSILON: f32 = 0.1;
const EPSILON_DECAY: f32 = 0.9999;
const BUFFER_CAPACITY: usize = 10000;
const BATCH_SIZE: usize = 300;
const LEARNING_RATE: f32 = 0.0001;
const EPISODES: usize = 50000;

#[derive(Clone, PartialEq, Debug)]
struct Experience {
    state: Vec<f32>,
    action: usize,
    reward: f32,
    next_state: Vec<f32>,
    draw: bool,
}

// impl Experience {
//     pub fn with_reward(&self, new_reward: f32) -> Self {
//         Experience {
//             reward: new_reward,
//             next_state: self.next_state.clone(),
//             state: self.state.clone(),
//             ..*self
//         }
//     }
// }

pub fn board_to_input(board: &Board, player: char) -> Vec<f32> {
    let total_size = board.len() * board[0].len();
    let mut input = vec![0.0; total_size * 2];
    for (idx, row) in board.iter().enumerate() {
        for (jdx, cell) in row.iter().enumerate() {
            let player_index = idx * board.len() + jdx;
            let oponent_index = total_size + idx * board.len() + jdx;
            input[player_index] = match *cell {
                c if c == player => 1.0,
                _ => 0.0,
            };
            input[oponent_index] = match *cell {
                c if c != player && c != '-' => 1.0,
                _ => 0.0,
            };
        }
    }
    input
}

pub fn epsilon_greedy(network: &NeuralNetwork, state: &Vec<f32>, epsilon: f32, board: &Board, legal_only: bool) -> usize {
    let mut rng = rand::thread_rng();
    if rng.gen::<f32>() < epsilon {
        if legal_only {
            // Choose a random legal move
            let mut legal_moves: Vec<usize> = vec![];
            for i in 0..(board.len() * board[0].len()) {
                let (row, col) = (i / board.len(), i % board.len());
                if board[row][col] == '-' {
                    legal_moves.push(i);
                }
            }
            legal_moves.choose(&mut rng).cloned().unwrap()
        } else {
            rng.gen_range(0 .. board.len() * board.len())
        }
    } else {
        // Choose the move with the highest Q-value among legal moves
        let q_values = network.forward(state);
        if legal_only{
            let mut legal_q_values: Vec<(usize, f32)> = vec![];
            for i in 0..(board.len() * board.len()) {
                let (row, col) = (i / board.len(), i % board.len());
                if board[row][col] == '-' {
                    legal_q_values.push((i, q_values[i]));
                }
            }
            legal_q_values.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|&(idx, _)| idx).unwrap()
        }
        else {
            q_values.iter().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|(idx, _)| idx).unwrap()
        }
    }
}

pub fn train(mut network: NeuralNetwork) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();
    
    let mut experiences: Vec<Experience> = Vec::new();
    let mut epsilon = INITIAL_EPSILON;
    // let result_network = Mutex::new(network);
    
    for _ in 0..EPISODES {
        let mut board = empty_board();
        let mut episode_finished = false;
        let mut player = 'X';
        let mut oponent = 'O';

        let mut episode_experiences: Vec<Experience> = Vec::new();

        loop {
            let state = board_to_input(&board, player);
            let action = epsilon_greedy(&network, &state, epsilon, &board, false);
            let (row, col) = (action / BOARD_SIZE, action % BOARD_SIZE);

            let res = make_move(&mut board, player, row, col);
            let next_state = board_to_input(&board, player);

            let reward;
            if res.is_err() {
                reward = -100.0;
                episode_finished = true;
            } else if check_winner(&board).is_some() {
                reward = 10.0;
                episode_finished = true;
            } else if is_full(&board) {
                reward = -0.5;
                episode_finished = true;
            } else {
                reward = -0.1;
            }
            
            episode_experiences.push(Experience {
                state: state,
                action,
                reward,
                next_state: next_state.clone(),
                draw: is_full(&board) && check_winner(&board).is_none(),
            });

            if episode_finished {
                break;
            }

            // Swap players
            std::mem::swap(&mut player, &mut oponent);
        }

        // Add all experiences of the current episode to the main experience list
        experiences.extend(episode_experiences.into_iter());

        // Ensure the buffer does not exceed its capacity
        while experiences.len() > BUFFER_CAPACITY {
            experiences.remove(0);
        }

        // Train the neural network
        if experiences.len() >= BATCH_SIZE {
            for experience in experiences.choose_multiple(&mut rng, BATCH_SIZE){
                    // if (experience.reward -10.0).abs() < 0.0001 {
                    //     println!("win");
                    // }
                    let q_values = network.forward(&experience.state);
                    let mut target_q_values = q_values.clone();

                    let next_q_values = network.forward(&experience.next_state);
                    let max_next_q_value = if experience.draw {0.0}
                                            else { next_q_values.iter().fold(f32::NEG_INFINITY, |acc, x| acc.max(*x))};
                    let target_q_value = experience.reward + DISCOUNT_FACTOR * max_next_q_value;
                    // zero target_q_values
                    // target_q_values.iter_mut().for_each(|x| *x = 0.0);
                    target_q_values[experience.action] = target_q_value;

                    // Update the neural network using gradient descent
                    network.backpropagate(&experience.state, &target_q_values, LEARNING_RATE);

            }
        }

        // Decay epsilon after a full game (episode) as decay is optimized for episodes count
        epsilon = epsilon * EPSILON_DECAY;
        epsilon = epsilon.max(FINAL_EPSILON);
    }
    Ok(network)
}

pub fn save_network(model: &NeuralNetwork, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, model)?;
    Ok(())
}

pub fn load_network(path: &str, node_counts: &[usize], activations: &[Activation]) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let model: NeuralNetwork = serde_json::from_reader(reader)?;
    for ((wnd, layer), activation) in node_counts.windows(2).zip(&model.layers).zip(activations) {
        if wnd[0] != layer.weights.len() || wnd[1] != layer.weights[0].len() || wnd[1] != layer.biases.len() || activation != &layer.activation {
            return Err("Invalid model".into());
        }
    }
    Ok(model)
}
