use crate::game::{Board, BOARD_SIZE, empty_board, check_winner, make_move, play_random_move, is_full};
use crate::network::NeuralNetwork;
use rand::{Rng};
use rand::seq::SliceRandom;
use rulinalg::utils;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};


const DISCOUNT_FACTOR: f32 = 0.9;
const INITIAL_EPSILON: f32 = 1.0;
const FINAL_EPSILON: f32 = 0.1;
const EPSILON_DECAY: f32 = 0.999;
const BUFFER_CAPACITY: usize = 10000;
const BATCH_SIZE: usize = 32;
const LEARNING_RATE: f32 = 0.001;
const EPISODES: usize = 50000;

#[derive(Clone)]
struct Experience {
    state: Vec<f32>,
    action: usize,
    reward: f32,
    next_state: Vec<f32>,
}

pub fn board_to_input(board: &Board, player: char) -> Vec<f32> {
    let mut input = vec![0.0; BOARD_SIZE * BOARD_SIZE];
    for (idx, row) in board.iter().enumerate() {
        for (jdx, cell) in row.iter().enumerate() {
            let index = idx * BOARD_SIZE + jdx;
            input[index] = match *cell {
                c if c == player => 1.0,
                '-' => 0.0,
                _ => -1.0,
            }
        }
    }
    input
}

fn epsilon_greedy(network: &NeuralNetwork, state: &Vec<f32>, epsilon: f32) -> usize {
    let mut rng = rand::thread_rng();
    if rng.gen::<f32>() < epsilon {
        // Choose a random move
        rng.gen_range(0..BOARD_SIZE * BOARD_SIZE)
    } else {
        // Choose the move with the highest Q-value
        let q_values = network.forward(state);
        //q_values.argminmax().1
        utils::argmax(&(q_values.into_iter().map(|f| f).collect::<Vec<f32>>())).0
    }
}

pub fn train(mut network: NeuralNetwork) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();
    let mut buffer: Vec<Experience> = Vec::new();
    let mut epsilon = INITIAL_EPSILON;
    for _ in 0..EPISODES {
        let mut board = empty_board();
        while check_winner(&board).is_none() &&
              !is_full(&board) 
        {
            let state = board_to_input(&board, 'X');
            let action = epsilon_greedy(&network, &state, epsilon);
            let (row, col) = (action / BOARD_SIZE, action % BOARD_SIZE);

            let res = make_move(&mut board, 'X', row, col);
            if res.is_err() {
                continue;
            }

            let reward;
            if let Some(winner) = check_winner(&board) {
                reward = if winner == 'X' { 1.0 } else { 0.0 };
            } else {
                reward = 0.5;
            }

            let next_state = board_to_input(&board, 'X');
            buffer.push(Experience {
                state,
                action,
                reward,
                next_state: next_state.clone(),
            });

            // Ensure the buffer does not exceed its capacity
            if buffer.len() > BUFFER_CAPACITY {
                buffer.remove(0);
            }

            // Sample a mini-batch from the buffer and update the neural network
            if buffer.len() >= BATCH_SIZE {
                let mut indices: Vec<usize> = (0..buffer.len()).collect();
                indices.shuffle(&mut rng);
                indices.truncate(BATCH_SIZE);

                for &idx in indices.iter() {
                    let experience = &buffer[idx];
                    let q_values = network.forward(&experience.state);
                    let mut target_q_values = q_values.clone();

                    let next_q_values = network.forward(&experience.next_state);
                    let max_next_q_value = next_q_values.iter().fold(f32::NEG_INFINITY, |acc, x| acc.max(*x));
                    let target_q_value = experience.reward + DISCOUNT_FACTOR * max_next_q_value;
                    target_q_values[experience.action] = target_q_value;

                    // Update the neural network using gradient descent
                    network.update_weights_and_biases(&experience.state, &target_q_values, LEARNING_RATE);
                }
            }

            // Decay epsilon
            epsilon = epsilon * EPSILON_DECAY;
            epsilon = epsilon.max(FINAL_EPSILON);

            // Agent plays a random move as 'O'
            if check_winner(&board).is_none() {
                let _ = play_random_move(&mut board, 'O');
            }
        }
    }
    Ok(network)
}

pub fn save_network(model: &NeuralNetwork, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, model)?;
    Ok(())
}

pub fn load_network(path: &str) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let model = serde_json::from_reader(reader)?;
    Ok(model)
}
