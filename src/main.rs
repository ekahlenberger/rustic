use rulinalg::utils;

use game::{Board, BOARD_SIZE, empty_board, check_winner, make_move, play_random_move, is_full};
use train::{board_to_input, save_network, load_network, epsilon_greedy};
use network::NeuralNetwork;

mod game;
mod network;
mod train;


fn main() {
    let trained_network_path = "trained_network.json";
    let mut network = match load_network(trained_network_path) {
        Ok(model) => model,
        Err(_) => NeuralNetwork::new(BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE),
    };

    let mut no_loss_streak = -1;
    let no_loss_streak_limit = 100;
    while no_loss_streak < no_loss_streak_limit {
        if no_loss_streak == 0 {
            println!("Loss: training network...");
            
            network = match train::train(network) {
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
            if let Err(e) = save_network(&network, trained_network_path) {
                println!("Error saving network: {:?}", e);
            }
        }
        if no_loss_streak == -1 {no_loss_streak = 0;}
        
        // Simulate a game using the current network
        let mut board = empty_board();
        let mut player = 'X';

        while check_winner(&board).is_none() && !is_full(&board) {
            //print_board(&board);
            //println!("Player: {}", player);

            if player == 'X' {
                let state = board_to_input(&board, 'X');
                //let q_values = network.forward(&state);
                //let action = utils::argmax(&(q_values.into_iter().map(|f| f).collect::<Vec<f32>>())).0;
                let action = train::epsilon_greedy(&network, &state, 0.0, &board); // Use epsilon_greedy with epsilon set to 0.0


                let (row, col) = (action / BOARD_SIZE, action % BOARD_SIZE);
                match make_move(&mut board, player, row, col) {
                    Ok(_) => (),
                    Err(_) => {
                        println!("Invalid move by 'X' on {col}x{row}. Game Over!");
                        break;
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

        print_board(&board);
        match check_winner(&board) {
            Some(winner) => {
                print!("Winner: {}", winner);
                if winner == 'O' {
                    println!(" Loss streak broken at: {}/{}", no_loss_streak, no_loss_streak_limit);
                    no_loss_streak = 0;
                }
                else {
                    no_loss_streak += 1;
                    println!(" {}/{}", no_loss_streak, no_loss_streak_limit);
                    
                }
                winner == 'X'
            }
            None => {
                println!("Draw");
                no_loss_streak += 1;
                false
            }
        };
        
        
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