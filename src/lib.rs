
// for logging (debug mostly, switched at compile time in cargo.toml)




#[macro_use]
extern crate lazy_static;

pub mod prelude;
pub mod types;
pub mod scsg;
pub mod svrg;
pub mod sag;
pub mod mnist;
mod monitor;
pub mod applis;

lazy_static! {
    #[allow(dead_code)]
    static ref LOG: u64 = {
        let res = init_log();
        res
    };
}

// install a logger facility
fn init_log() -> u64 {
    env_logger::try_init().unwrap();
    println!("\n ************** initializing logger *****************\n");    
    return 1;
}

#[cfg(test)]
mod tests {
    #[test]
    // initialize once log system for tests.
    fn init_log() {
        env_logger::try_init().unwrap();
    }
}  // end of tests
