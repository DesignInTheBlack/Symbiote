pub const LSH_BITS: usize = 16;
pub const LSH_NEIGHBOR_RADIUS: usize = 1;

const LSH_STRIDE: usize = 37;
const LSH_OFFSET: usize = 13;

pub fn embedding_signature(embedding: &[f32]) -> u32 {
    if embedding.is_empty() {
        return 0;
    }

    let mean = embedding.iter().sum::<f32>() / embedding.len() as f32;
    let mut signature = 0u32;
    let len = embedding.len();

    for bit in 0..LSH_BITS {
        let idx = (bit * LSH_STRIDE + LSH_OFFSET) % len;
        if embedding[idx] >= mean {
            signature |= 1u32 << bit;
        }
    }

    signature
}

pub fn candidate_buckets(signature: u32) -> Vec<i64> {
    let mut buckets = Vec::with_capacity(LSH_BITS + 1);
    buckets.push(signature as i64);

    if LSH_NEIGHBOR_RADIUS >= 1 {
        for bit in 0..LSH_BITS {
            buckets.push((signature ^ (1u32 << bit)) as i64);
        }
    }

    buckets.sort_unstable();
    buckets.dedup();
    buckets
}
