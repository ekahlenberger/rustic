use activation::Activation;
use game::{Board, BOARD_SIZE, empty_board, check_winner, make_move, play_random_move, is_full};
use train::{board_to_input, save_network, load_network};
use network::NeuralNetwork;

mod game;
mod network;
mod train;
mod layer;
mod activation;

fn main() {
    let trained_network_path = "trained_network.json";
    let mut network = match load_network(trained_network_path, &[18, 16, 14, 9], &[Activation::Tanh, Activation::Tanh, Activation::Tanh]) {
        Ok(model) => model,
        Err(_) => NeuralNetwork::new(&[18, 16, 14, 9], &[Activation::Tanh, Activation::Tanh, Activation::Tanh]),
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
        let mut boards = vec![];
        let mut illegal_move = false;
        while check_winner(&board).is_none() && !is_full(&board) {
            if player == 'X' {
                let state = board_to_input(&board, 'X');
                
                let action = train::epsilon_greedy(&network, &state, 0.0, &board, false); // Use epsilon_greedy with epsilon set to 0.0 (exploit)


                let (row, col) = (action / BOARD_SIZE, action % BOARD_SIZE);
                match make_move(&mut board, player, row, col) {
                    Ok(_) => (),
                    Err(_) => {
                        println!("Invalid move by 'X' on {col}x{row}. Game Over!");
                        illegal_move = true;
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
            boards.push(board.clone());
        }

        print_boards_horizontally(&boards);
        match check_winner(&board) {
            Some(winner) => {
                print!("Winner: {}", winner);
                if winner == 'O' {
                    println!(" No Loss streak broken at: {}/{}", no_loss_streak, no_loss_streak_limit);
                    no_loss_streak = 0;
                }
                else {
                    no_loss_streak += 1;
                    println!(" {}/{}", no_loss_streak, no_loss_streak_limit);
                }
            }
            None => {
                if illegal_move {
                    println!("Illegal move. No Loss streak broken at: {}/{}", no_loss_streak, no_loss_streak_limit);
                    no_loss_streak = 0;
                }
                else {
                    no_loss_streak += 1;
                    println!("Draw. {}/{}", no_loss_streak, no_loss_streak_limit);
                }
            }
        };
        
        
    }
}

fn print_boards_horizontally(boards: &[Board]) {
    for row in 0..BOARD_SIZE {
        for board in boards {
            for col in 0..BOARD_SIZE {
                print!("{} ", board[row][col]);
            }
            print!(" | ");
        }
        println!();
    }
}