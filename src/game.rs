pub const BOARD_SIZE: usize = 3;

pub type Board = Vec<Vec<char>>;

pub fn empty_board() -> Board {
    vec![vec!['-'; BOARD_SIZE]; BOARD_SIZE]
}

pub fn is_full(board: &Board) -> bool {
    !board.iter().any(|row| row.iter().any(|&cell| cell == '-'))
}

pub fn check_winner(board: &Board) -> Option<char> {
    for i in 0..BOARD_SIZE {
        if board[i][0] != '-' && board[i][0] == board[i][1] && board[i][1] == board[i][2] {
            return Some(board[i][0]);
        }
        if board[0][i] != '-' && board[0][i] == board[1][i] && board[1][i] == board[2][i] {
            return Some(board[0][i]);
        }
    }
    if board[0][0] != '-' && board[0][0] == board[1][1] && board[1][1] == board[2][2] {
        return Some(board[0][0]);
    }
    if board[0][2] != '-' && board[0][2] == board[1][1] && board[1][1] == board[2][0] {
        return Some(board[0][2]);
    }
    None
}

pub fn make_move(board: &mut Board, player: char, row: usize, col: usize) -> Result<(), &'static str> {
    if row >= BOARD_SIZE || col >= BOARD_SIZE {
        return Err("Invalid move: out of bounds");
    }
    if board[row][col] != '-' {
        return Err("Invalid move: cell already occupied");
    }

    board[row][col] = player;
    Ok(())
}

pub fn play_random_move(board: &mut Board, player: char) -> Result<(), &'static str> {
    let available_moves: Vec<(usize, usize)> = board.iter().enumerate().flat_map(|(row, r)| {
        r.iter().enumerate().filter_map(move |(col, &cell)| if cell == '-' {
            Some((row, col))
        } else {
            None
        })
    }).collect();

    if available_moves.is_empty() {
        return Err("No valid moves available");
    }

    let idx = rand::random::<usize>() % available_moves.len();
    let (row, col) = available_moves[idx];
    make_move(board, player, row, col)
}