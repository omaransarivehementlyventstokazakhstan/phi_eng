//! Minimal FxHash hasher for hot maps (contact cache).
//!
//! `std`'s default `RandomState` (SipHash) is DOS-resistant but slow —
//! overkill for internal `(body_a, body_b)` keys that never come from the
//! network. FxHash is a single multiply-rotate per word, deterministic
//! (fixed seed, so map behavior is reproducible run-to-run), and a
//! well-known fit for small integer keys.
//!
//! Algorithm: the `rustc-hash` Fx variant (64-bit). `write_usize` folds
//! each word as `hash = (hash.rotate_left(5) ^ word) * K`. `write` consumes
//! trailing bytes in usize chunks with the remainder folded byte-wise.
//! This is NOT cryptographic and MUST NOT hash untrusted input.

use std::hash::{BuildHasher, Hasher};

const SEED: usize = 0x51_7c_c1_b7_27_22_0a_95;

// K differs by pointer width so the avalanche covers the full word.
#[cfg(target_pointer_width = "64")]
const K: usize = 0x51_7c_c1_b7_27_22_0a_95;
#[cfg(target_pointer_width = "32")]
const K: usize = 0x9e_37_79_b9_7f_4a_7c_15;

#[derive(Debug, Clone, Copy, Default)]
pub struct FxHasher {
    hash: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BuildFxHasher;

impl FxHasher {
    #[inline]
    fn add_to_hash(&mut self, word: usize) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash as u64
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Fold full words first, then the tail byte-wise. Correct for any
        // length; hot keys ((usize, usize)) always hit the word path.
        // Manual chunking (no chunk APIs): identical output on every
        // toolchain, so hash values never drift with the compiler.
        const WORD: usize = std::mem::size_of::<usize>();
        let mut i = 0;
        while i + WORD <= bytes.len() {
            let mut arr = [0u8; std::mem::size_of::<usize>()];
            arr.copy_from_slice(&bytes[i..i + WORD]);
            self.add_to_hash(usize::from_ne_bytes(arr));
            i += WORD;
        }
        for &b in &bytes[i..] {
            self.add_to_hash(b as usize);
        }
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add_to_hash(i);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add_to_hash(i as usize);
        #[cfg(target_pointer_width = "64")]
        {}
        #[cfg(target_pointer_width = "32")]
        {
            self.add_to_hash((i >> 32) as usize);
        }
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add_to_hash(i as usize);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add_to_hash(i as usize);
    }
}

impl BuildHasher for BuildFxHasher {
    type Hasher = FxHasher;

    #[inline]
    fn build_hasher(&self) -> FxHasher {
        FxHasher { hash: SEED }
    }
}

/// Hot-map alias: `HashMap` with the Fx hasher.
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildFxHasher>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fx_deterministic() {
        // Same keys hash identically across hasher instances (fixed seed).
        fn hash_pair(a: usize, b: usize) -> u64 {
            let mut h = BuildFxHasher.build_hasher();
            h.write_usize(a);
            h.write_usize(b);
            h.finish()
        }
        assert_eq!(hash_pair(1, 2), hash_pair(1, 2));
        assert_ne!(hash_pair(1, 2), hash_pair(2, 1));
        assert_ne!(hash_pair(0, 0), hash_pair(0, 1));
    }

    #[test]
    fn test_fx_map_roundtrip() {
        let mut m: FxHashMap<(usize, usize), Vec<u32>> = FxHashMap::default();
        m.entry((3, 7)).or_default().push(42);
        m.entry((3, 7)).or_default().push(43);
        assert_eq!(m[&(3, 7)], vec![42, 43]);
        assert!(!m.contains_key(&(7, 3)));
    }

    #[test]
    fn test_fx_write_any_length() {
        // write() path must accept arbitrary byte strings without panic
        // and distinguish inputs.
        fn hash_bytes(b: &[u8]) -> u64 {
            let mut h = BuildFxHasher.build_hasher();
            h.write(b);
            h.finish()
        }
        assert_eq!(hash_bytes(b"abc"), hash_bytes(b"abc"));
        assert_ne!(hash_bytes(b"abc"), hash_bytes(b"abd"));
        assert_ne!(hash_bytes(b""), hash_bytes(b"\0"));
    }
}
