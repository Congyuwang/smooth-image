#[derive(Debug)]
pub enum Error {
    ErrorMessage(String),
}

pub type Result<T> = std::result::Result<T, Error>;
