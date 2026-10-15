//! Deterministic constraint islands for parallel solving.
//!
//! An island is a set of bodies connected by contacts or joints that must
//! be solved together, but independently of every other island. Islands
//! are rebuilt from scratch every step (stateless across frames), so they
//! never go stale when bodies are added, removed, or reindexed.
//!
//! Determinism contract (the whole point of this module):
//! - Union-find runs serially over canonically sorted edges, so island
//!   membership is a pure function of the constraint set.
//! - Islands are emitted in canonical order (sorted by smallest body id),
//!   and constraint indices within an island keep their input order
//!   (stable partition). Parallel execution therefore produces the exact
//!   same memory writes as the serial path, in an order that cannot
//!   affect the result (islands are body-disjoint).
//! - Warm-start cache write-back is merged canonically after joining.
//!
//! Soundness contract for the parallel solver: islands are BODY-DISJOINT
//! by construction (every body index appears in at most one island), so
//! threads may hold `&mut` access to disjoint body subsets through a raw
//! pointer. Debug builds verify disjointness with a visited bitset.

/// One island: body indices plus indices into the step's constraint list.
/// Both lists are in canonical order (see module docs).
#[derive(Debug, Clone, Default)]
pub struct Island {
    /// Sorted body indices owned by this island.
    pub bodies: Vec<usize>,
    /// Constraint indices (into the caller's constraint slice) belonging
    /// to this island, in input order (stable).
    pub constraints: Vec<usize>,
}

/// Build islands from contact pairs and joint endpoint pairs.
///
/// - `n_bodies`: body count (all ids/indices are `< n_bodies`).
/// - `contacts`: `(body_a, body_b)` contact pairs (any order; canonical
///   or not — union-find is order-sensitive, so pairs are sorted first).
/// - `joints`: `(body_a, body_b)` joint endpoint pairs (same treatment).
///
/// Static bodies participate in union-find like any other body: a static
/// floor shared by two piles merges them into one island (correct — the
/// piles interact through contact impulses... more precisely, they share
/// no velocity state, but treating shared statics as connectors keeps
/// the partitioning rule uniform and predictable; a future optimization
/// may split on statics once the solver skips static writes).
/// Self-pairs `(a, a)` are ignored (degenerate input, never generated
/// by the pipeline).
pub fn build_islands(
    n_bodies: usize,
    contacts: &[(usize, usize)],
    joints: &[(usize, usize)],
) -> Vec<Island> {
    if n_bodies == 0 {
        return Vec::new();
    }
    // Canonical edge list: sorted, deduplicated, self-pairs dropped.
    // Sorting makes the partition a pure function of the edge SET.
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(contacts.len() + joints.len());
    for &(a, b) in contacts.iter().chain(joints.iter()) {
        if a >= n_bodies || b >= n_bodies || a == b {
            continue;
        }
        edges.push(if a < b { (a, b) } else { (b, a) });
    }
    edges.sort_unstable();
    edges.dedup();

    // Union-find with path compression + union by rank.
    let mut parent: Vec<usize> = (0..n_bodies).collect();
    let mut rank: Vec<u8> = vec![0; n_bodies];

    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    fn union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb {
            return;
        }
        match rank[ra].cmp(&rank[rb]) {
            std::cmp::Ordering::Less => parent[ra] = rb,
            std::cmp::Ordering::Greater => parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                parent[rb] = ra;
                rank[ra] = rank[ra].saturating_add(1);
            }
        }
    }

    for &(a, b) in &edges {
        union(&mut parent, &mut rank, a, b);
    }

    // Group bodies by root. Only bodies that appear in some edge join an
    // island (isolated bodies need no solving).
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n_bodies];
    let mut present = vec![false; n_bodies];
    for &(a, b) in &edges {
        present[a] = true;
        present[b] = true;
    }
    for (i, &is_present) in present.iter().enumerate() {
        if is_present {
            let r = find(&mut parent, i);
            groups[r].push(i);
        }
    }

    // Canonical island order: by smallest body id. Members already ascend
    // (insertion in index order).
    let mut islands: Vec<Island> = groups
        .into_iter()
        .filter(|g| !g.is_empty())
        .map(|bodies| Island {
            bodies,
            constraints: Vec::new(),
        })
        .collect();
    islands.sort_by_key(|isl| isl.bodies[0]);
    islands
}

/// Assign each contact to its island. `contact_pairs[i]` is the
/// `(body_a, body_b)` pair that produced `constraints[i]` — parallel
/// slices. Every pair must share one island (both endpoints connected);
/// pairs violating that indicate pipeline corruption and are dropped in
/// release (debug-asserted).
///
/// Returns per-island constraint index lists in canonical island order.
pub fn assign_constraints(
    islands: &[Island],
    contact_pairs: &[(usize, usize)],
    body_to_island: &[usize],
) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); islands.len()];
    for (ci, &(a, b)) in contact_pairs.iter().enumerate() {
        if a >= body_to_island.len() || b >= body_to_island.len() {
            debug_assert!(false, "contact pair out of range");
            continue;
        }
        let ia = body_to_island[a];
        let ib = body_to_island[b];
        // Both endpoints of a union-find edge share a root by construction.
        if ia != ib || ia >= out.len() {
            debug_assert!(false, "contact endpoints split across islands");
            continue;
        }
        out[ia].push(ci);
    }
    out
}

/// Build the body-index → island-index map for [`assign_constraints`].
/// Bodies in no island map to `usize::MAX`.
pub fn island_index_map(islands: &[Island], n_bodies: usize) -> Vec<usize> {
    let mut map = vec![usize::MAX; n_bodies];
    for (ii, isl) in islands.iter().enumerate() {
        for &b in &isl.bodies {
            if b < n_bodies {
                map[b] = ii;
            }
        }
    }
    map
}

/// Debug-only disjointness verifier: every body in at most one island.
/// Called by the parallel solver before spawning threads.
#[cfg(debug_assertions)]
pub fn debug_verify_disjoint(islands: &[Island], n_bodies: usize) {
    let mut seen = vec![false; n_bodies];
    for isl in islands {
        for &b in &isl.bodies {
            assert!(b < n_bodies, "island body out of range");
            assert!(!seen[b], "body {b} in two islands (parallel UB)");
            seen[b] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_edges_no_islands() {
        assert!(build_islands(0, &[], &[]).is_empty());
        assert!(build_islands(5, &[], &[]).is_empty());
    }

    #[test]
    fn test_single_chain_single_island() {
        // A-B-C: one island, canonical order.
        let isl = build_islands(4, &[(0, 1), (1, 2)], &[]);
        assert_eq!(isl.len(), 1);
        assert_eq!(isl[0].bodies, vec![0, 1, 2]);
    }

    #[test]
    fn test_two_chains_two_islands_sorted() {
        // Input order must not matter: canonical output.
        let isl = build_islands(5, &[(3, 4), (0, 1)], &[]);
        assert_eq!(isl.len(), 2);
        assert_eq!(isl[0].bodies, vec![0, 1]);
        assert_eq!(isl[1].bodies, vec![3, 4]);
    }

    #[test]
    fn test_joints_join_islands() {
        // Contacts (0,1) + joint (1,2) merge all three.
        let isl = build_islands(4, &[(0, 1)], &[(1, 2)]);
        assert_eq!(isl.len(), 1);
        assert_eq!(isl[0].bodies, vec![0, 1, 2]);
    }

    #[test]
    fn test_self_pairs_and_oob_dropped() {
        let isl = build_islands(3, &[(0, 0), (0, 1), (9, 9)], &[]);
        assert_eq!(isl.len(), 1);
        assert_eq!(isl[0].bodies, vec![0, 1]);
    }

    #[test]
    fn test_assign_keeps_input_order() {
        let isl = build_islands(5, &[(0, 1), (3, 4), (0, 1)], &[]);
        let map = island_index_map(&isl, 5);
        let pairs = vec![(0, 1), (3, 4), (1, 0)];
        let assigned = assign_constraints(&isl, &pairs, &map);
        assert_eq!(assigned.len(), 2);
        assert_eq!(assigned[0], vec![0, 2]);
        assert_eq!(assigned[1], vec![1]);
    }

    #[test]
    fn test_determinism_same_set() {
        // Shuffled input orderings of the same edge set → identical islands.
        let a = build_islands(6, &[(0, 1), (2, 3), (4, 5)], &[(1, 2)]);
        let b = build_islands(6, &[(4, 5), (0, 1), (2, 3)], &[(1, 2)]);
        // (1,2) joint merges {0,1} and {2,3}.
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.bodies, y.bodies);
        }
        assert_eq!(a[0].bodies, vec![0, 1, 2, 3]);
        assert_eq!(a[1].bodies, vec![4, 5]);
    }
}
