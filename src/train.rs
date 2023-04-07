use crate::game::{Board, BOARD_SIZE, empty_board, check_winner, make_move, is_full};
use crate::network::NeuralNetwork;
use rand::{Rng};
use rand::seq::SliceRandom;
//use rulinalg::utils;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use rayon::prelude::*;
use std::sync::Mutex;
use itertools::{Itertools, Either};

const DISCOUNT_FACTOR: f32 = 0.9;
const INITIAL_EPSILON: f32 = 1.0;
const FINAL_EPSILON: f32 = 0.1;
const EPSILON_DECAY: f32 = 0.9999;
const BUFFER_CAPACITY: usize = 5000;
const BATCH_SIZE: usize = 300;
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

pub fn epsilon_greedy(network: &NeuralNetwork, state: &Vec<f32>, epsilon: f32, board: &Board) -> usize {
    let mut rng = rand::thread_rng();
    if rng.gen::<f32>() < epsilon {
        // Choose a random legal move
        let mut legal_moves: Vec<usize> = vec![];
        for i in 0..(BOARD_SIZE * BOARD_SIZE) {
            let (row, col) = (i / BOARD_SIZE, i % BOARD_SIZE);
            if board[row][col] == '-' {
                legal_moves.push(i);
            }
        }
        legal_moves.choose(&mut rng).cloned().unwrap()
    } else {
        // Choose the move with the highest Q-value among legal moves
        let q_values = network.forward(state);
        let mut legal_q_values: Vec<(usize, f32)> = vec![];
        for i in 0..(BOARD_SIZE * BOARD_SIZE) {
            let (row, col) = (i / BOARD_SIZE, i % BOARD_SIZE);
            if board[row][col] == '-' {
                legal_q_values.push((i, q_values[i]));
            }
        }
        legal_q_values.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|&(idx, _)| idx).unwrap()
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
        let mut episodes_experiences: Vec<Experience> = Vec::new();
        let mut episode_finished = false;
        // if thread_rng().gen::<f32>() < 0.5 {
        //     let _ = play_random_move(&mut board, 'O');
        // }
        loop {
            let nc = result_network.lock().unwrap().clone();
            let state = board_to_input(&board, player);
            let action = epsilon_greedy(&nc, &state, epsilon, &board);
            let (row, col) = (action / BOARD_SIZE, action % BOARD_SIZE);

            let res = make_move(&mut board, player, row, col);

            let reward;
            if res.is_err() {
                reward = -20f32;
                let next_state = state.clone();
                experiences.push(Experience {
                    state,
                    action,
                    reward,
                    next_state,
                });
                break;
            }
            else if let Some(_) = check_winner(&board) {
                //reward = if winner == player { 10f32 } else { -10f32 };
                episode_finished = true;
            } else if is_full(&board) {
                //reward = 0.0;
                episode_finished = true;
            } else {
                //reward = 0.5;
            }

            let next_state = match res {
                Ok(_) => board_to_input(&board, player),
                Err(_) => state.clone()
            };
            episodes_experiences.push(Experience {
                state,
                action,
                reward: 0f32,
                next_state: next_state.clone(),
            });

            

            if episode_finished {
                // split the experiences into player's and opponent's
                let (mut player_experiences, mut opponent_experiences): (Vec<_>, Vec<_>) = episodes_experiences
                    .into_iter()
                    .enumerate()
                    .partition_map(|(index, experience)| {
                        if index % 2 == 0 {
                            Either::Left(experience)
                        } else {
                            Either::Right(experience)
                        }
                    });
                // Apply rewards to the player's experiences
                let player_reward = if check_winner(&board) == Some(player) { 10f32 } else { -10f32 };
                let too_many_player_moves = (player_experiences.len() - 3) as f32;
                for experience in player_experiences.iter_mut() {
                    experience.reward = player_reward - too_many_player_moves;
                }

                // Apply rewards to the opponent's experiences
                let opponent_reward = if check_winner(&board) == Some(oponent) { 10f32 } else { -10f32 };
                let too_many_opponent_moves = (opponent_experiences.len() - 3) as f32;
                for experience in opponent_experiences.iter_mut() {
                    experience.reward = opponent_reward - too_many_opponent_moves;
                }

                // fill experiences with episodes_experiences
                experiences.append(&mut player_experiences);
                experiences.append(&mut opponent_experiences);

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
                        let max_next_q_value = next_q_values.iter().fold(f32::NEG_INFINITY, |acc, x| acc.max(*x));
                        let target_q_value = experience.reward + DISCOUNT_FACTOR * max_next_q_value;
                        target_q_values[experience.action] = target_q_value;

                        // Update the neural network using gradient descent
                        result_network.lock().unwrap().update_weights_and_biases(&experience.state, &target_q_values, LEARNING_RATE);
                    });
                }
                // Decay epsilon
                epsilon = epsilon * EPSILON_DECAY;
                epsilon = epsilon.max(FINAL_EPSILON);
                break;
            }

            // Agent plays a random move as 'O'
            // if check_winner(&board).is_none() {
            //     let _ = play_random_move(&mut board, 'O');
            // }
            
            // Swap players
            std::mem::swap(&mut player, &mut oponent);
        }
    }
    Ok(result_network.into_inner().unwrap())
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
