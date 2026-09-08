//! The capability other vector indexes do not have: querying the past.
use chronovec::{Index, Metric};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let index = Index::builder(4).metric(Metric::L2).nprobe(8).build()?;

    let before = index.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
    index.insert(2, &[0.0, 1.0, 0.0, 0.0])?;
    index.delete(1)?;

    let query = [1.0, 0.0, 0.0, 0.0];
    println!("now:      {:?}", index.search(&query, 2)?); // id 1 gone, id 2 remains
    println!("as of t1: {:?}", index.search_as_of(&query, 2, before)?);

    let reclaimed = index.vacuum(index.horizon(), 64);
    println!("reclaimed {reclaimed} expired versions");
    Ok(())
}
