use crate::activation::Activation;
use crate::game::{Board, BOARD_SIZE, empty_board, check_winner, make_move, is_full};
use crate::network::NeuralNetwork;
use rand::{Rng};
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use rayon::prelude::*;
use std::sync::Mutex;
//use itertools::{Itertools, Either};

const DISCOUNT_FACTOR: f32 = 0.9;
const INITIAL_EPSILON: f32 = 1.0;
const FINAL_EPSILON: f32 = 0.1;
const EPSILON_DECAY: f32 = 0.9999;
const BUFFER_CAPACITY: usize = 10000;
const BATCH_SIZE: usize = 300;
const LEARNING_RATE: f32 = 0.001;
const EPISODES: usize = 50000;

#[derive(Clone, PartialEq, Debug)]
struct Experience {
    state: Vec<f32>,
    action: usize,
    reward: f32,
    next_state: Vec<f32>,
    draw: bool,
}

impl Experience {
    pub fn with_reward(&self, new_reward: f32) -> Self {
        Experience {
            reward: new_reward,
            next_state: self.next_state.clone(),
            state: self.state.clone(),
            ..*self
        }
    }
}

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
            rng.gen_range(0 .. BOARD_SIZE * BOARD_SIZE)
        }
    } else {
        // Choose the move with the highest Q-value among legal moves
        let q_values = network.forward(state);
        if legal_only{
            let mut legal_q_values: Vec<(usize, f32)> = vec![];
            for i in 0..(BOARD_SIZE * BOARD_SIZE) {
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

pub fn train(network: NeuralNetwork) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();
    
    let mut experiences: Vec<Experience> = Vec::new();
    let mut epsilon = INITIAL_EPSILON;
    let result_network = Mutex::new(network);
    let mut player = 'X';
    let mut oponent = 'O';
    for _ in 0..EPISODES {
        let mut board = empty_board();
        let mut episode_finished = false;
        
        // if thread_rng().gen::<f32>() < 0.5 {
        //     let _ = play_random_move(&mut board, 'O');
        // }
        let mut last_round_experience: Option<Experience> = None;
        loop {
            let nc = result_network.lock().unwrap().clone();
            let state = board_to_input(&board, player);
            let action = match player {
                'X' =>  epsilon_greedy(&nc, &state, epsilon, &board, true),
                'O' =>  epsilon_greedy(&nc, &state, epsilon, &board, true), // Oponent always plays randomly, but valid only
                _ => panic!("Invalid player")
            };
            let (row, col) = (action / BOARD_SIZE, action % BOARD_SIZE);

            let res = make_move(&mut board, player, row, col);
            let next_state = board_to_input(&board, player);

            let reward;
            if res.is_err() { // can only be X, as O always plays randomly and valid only
                reward = -2.0;
                episode_finished = true;
            }
            else if let Some(winner) = check_winner(&board) {
                reward = if winner == player { 1.0 } else { -1.0 };
                episode_finished = true;
            } else if is_full(&board) {
                reward = -0.5;
                episode_finished = true;
            }
            else {
                reward = 0.0;
            }
            
            if episode_finished {
                experiences.push(Experience {
                    state: state,
                    action,
                    reward,
                    next_state,
                    draw: is_full(&board) && check_winner(&board).is_none(),
                });
                // if let Some(exp) = last_round_experience.take() {
                //     if res.is_ok() { // don't add experience if the move was invalid, as this is not of any value
                //         experiences.push(exp.with_reward(-reward ));
                //     }
                // }
            }
            else
            {
                last_round_experience = Some(Experience {
                    state: state,
                    action,
                    reward,
                    next_state,
                    draw: is_full(&board) && check_winner(&board).is_none(),
                });
            }
            

            // Ensure the buffer does not exceed its capacity
            if experiences.len() > BUFFER_CAPACITY {
                experiences.remove(0);
            }

            // Ensure the buffer does not exceed its capacity
            if experiences.len() > BUFFER_CAPACITY {
                experiences.remove(0);
            }

            // Train the neural network
            if experiences.len() >= BATCH_SIZE {
                experiences
                .choose_multiple(&mut rng, BATCH_SIZE)
                .collect::<Vec<_>>()
                .into_par_iter()
                .for_each(|experience| {
                    let q_values = nc.forward(&experience.state);
                    let mut target_q_values = q_values.clone();

                    let next_q_values = nc.forward(&experience.next_state);
                    let max_next_q_value = if experience.draw {0.0} // there is no next state and therefore nothing to maximize
                                                else { next_q_values.iter().fold(f32::NEG_INFINITY, |acc, x| acc.max(*x))};
                    let target_q_value = experience.reward + DISCOUNT_FACTOR * max_next_q_value;
                    target_q_values[experience.action] = target_q_value;

                    // Update the neural network using gradient descent
                    result_network.lock().unwrap().backpropagate(&experience.state, &target_q_values, LEARNING_RATE);
                });
            }                       
            
            // Agent plays a random move as 'O'
            // if check_winner(&board).is_none() {
            //     let _ = play_random_move(&mut board, 'O');
            // }
            
            // Swap players
            std::mem::swap(&mut player, &mut oponent);
            if episode_finished {
                break;
            }
        }
        // Decay epsilon after a full game (episode) as decay is optimized for episodes count
        epsilon = epsilon * EPSILON_DECAY;
        epsilon = epsilon.max(FINAL_EPSILON);
    }
    Ok(result_network.into_inner().unwrap())
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
