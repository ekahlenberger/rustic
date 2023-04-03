use rulinalg::utils;

use game::{Board, BOARD_SIZE, empty_board, check_winner, make_move, play_random_move};
use train::{board_to_input, save_network, load_network};
use network::NeuralNetwork;

mod game;
mod network;
mod train;


fn main() {
    let trained_network_path = "trained_network.json";
    let network = match load_network(trained_network_path) {
        Ok(model) => model,
        Err(_) => NeuralNetwork::new(BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE),
    };

    let trained_network = match train::train(network) {
        Ok(n) => {
            println!("Training complete.");
            n
        }
        Err(e) => {
            println!("Error during training: {:?}", e);
            return
        }
    };

    // Save the trained model
    
    if let Err(e) = save_network(&trained_network, trained_network_path) {
        println!("Error saving network: {:?}", e);
    }

    // Simulate a game using the trained model
    let mut board = empty_board();
    let mut player = 'X';

    while check_winner(&board).is_none() {
        println!("Player: {}", player);
        print_board(&board);

        if player == 'X' {
            let state = board_to_input(&board, 'X');
            let q_values = trained_network.forward(&state);
            let action = utils::argmax(&(q_values.into_iter().map(|f| f).collect::<Vec<f32>>())).0;

            let (row, col) = (action / BOARD_SIZE, action % BOARD_SIZE);
            match make_move(&mut board, player, row, col) {
                Ok(_) => (),
                Err(_) => {
                    println!("Invalid move by 'X'.");
                    continue;
                }
            }
        } else {
            if let Err(_) = play_random_move(&mut board, player) {
                println!("Invalid move by 'O'.");
                continue;
            }
        }

        player = if player == 'X' { 'O' } else { 'X' };
    }

    println!("Final board state:");
    print_board(&board);
    match check_winner(&board) {
        Some(winner) => println!("Winner: {}", winner),
        None => println!("Draw"),
    }
}

fn print_board(board: &Board) {
    for row in board {
        for cell in row {
            print!("{} ", cell);
        }
        println!();
    }
}