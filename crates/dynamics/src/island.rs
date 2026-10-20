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

/// Sentinel for "this body is in no island".
const NO_ISLAND: u32 = u32::MAX;

/// Reusable scratch for the island partition.
///
/// The partition is rebuilt from scratch every step (it must be: bodies
/// are added, removed and reindexed), so every buffer it needs is
/// pooled and refilled rather than reallocated. Without this the
/// parallel solver path allocated O(bodies) times per step — and it
/// allocated them even when the partition turned out to hold a single
/// island and the work was handed back to the serial solver.
///
/// The grouping is done with a counting sort over the union-find
/// roots, so no per-island index `Vec` is ever created: the island
/// member lists are ranges into two flat arrays.
#[derive(Debug, Default)]
pub struct IslandScratch {
    /// Canonical (sorted, deduplicated) edge list.
    edges: Vec<(usize, usize)>,
    /// Union-find parent array.
    parent: Vec<usize>,
    /// Union-find rank array.
    rank: Vec<u8>,
    /// Whether a body takes part in some edge.
    present: Vec<bool>,
    /// Island index of each union-find root (`NO_ISLAND` = unseen).
    root_island: Vec<u32>,
    /// Island index of each present body (`NO_ISLAND` = not in an island).
    body_island: Vec<u32>,
    /// Body index -> island index (`usize::MAX` = not in an island).
    body_to_island: Vec<usize>,
    /// Per-island member counts (counting-sort buckets).
    counts: Vec<u32>,
    /// Prefix-summed offsets, one per island plus a terminator.
    starts: Vec<u32>,
    /// Fill cursors, one per island.
    cursors: Vec<u32>,
    /// Bodies grouped by island, ascending within an island.
    body_order: Vec<usize>,
    /// Constraint indices grouped by island, in input order.
    constraint_order: Vec<usize>,
    /// Joint indices grouped by island, in input order.
    joint_order: Vec<usize>,
    /// Recycled per-island index buffers. The three list kinds share
    /// the pool (they are the same type): growing an island list
    /// borrows a spare, shrinking it returns one. Without this the
    /// per-island `Vec`s were dropped and reallocated every step.
    spare_lists: Vec<Vec<usize>>,
    /// Per-island body index lists.
    pub island_bodies: Vec<Vec<usize>>,
    /// Per-island constraint index lists.
    pub island_constraints: Vec<Vec<usize>>,
    /// Per-island joint index lists.
    pub island_joints: Vec<Vec<usize>>,
    /// Number of islands produced by the last build.
    pub island_count: usize,
}

impl IslandScratch {
    /// Fresh scratch (no capacity reserved yet).
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the island partition from contact pairs and joint endpoint
    /// pairs.
    ///
    /// The island order and the within-island member order are exactly
    /// those of [`build_islands`] (canonical: islands ordered by their
    /// smallest body id, members ascending), so the two agree
    /// bit-for-bit.
    pub fn build(
        &mut self,
        n_bodies: usize,
        contacts: &[(usize, usize)],
        joints: &[(usize, usize)],
    ) {
        // Canonical edge list: sorted, deduplicated, self-pairs dropped.
        // Sorting makes the partition a pure function of the edge SET.
        self.edges.clear();
        self.edges.reserve(contacts.len() + joints.len());
        for &(a, b) in contacts.iter().chain(joints.iter()) {
            if a >= n_bodies || b >= n_bodies || a == b {
                continue;
            }
            self.edges.push(if a < b { (a, b) } else { (b, a) });
        }
        self.edges.sort_unstable();
        self.edges.dedup();

        // Union-find with path compression + union by rank.
        self.parent.clear();
        self.parent.extend(0..n_bodies);
        self.rank.clear();
        self.rank.resize(n_bodies, 0);
        for &(a, b) in &self.edges {
            union(&mut self.parent, &mut self.rank, a, b);
        }

        // Bodies that take part in some edge. Only those join an island
        // (isolated bodies need no solving).
        self.present.clear();
        self.present.resize(n_bodies, false);
        for &(a, b) in &self.edges {
            self.present[a] = true;
            self.present[b] = true;
        }

        // Island index per root, assigned in ascending body order. The
        // first present body of a root is that island's smallest member,
        // so this is the same order as sorting the groups by their first
        // member.
        self.root_island.clear();
        self.root_island.resize(n_bodies, NO_ISLAND);
        self.body_island.clear();
        self.body_island.resize(n_bodies, NO_ISLAND);
        self.counts.clear();
        self.island_count = 0;
        for i in 0..n_bodies {
            if !self.present[i] {
                continue;
            }
            let r = find(&mut self.parent, i);
            if self.root_island[r] == NO_ISLAND {
                self.root_island[r] = self.island_count as u32;
                self.counts.push(0);
                self.island_count += 1;
            }
            let isl = self.root_island[r] as usize;
            self.body_island[i] = isl as u32;
            self.counts[isl] += 1;
        }

        // Counting sort: island members in one flat array.
        self.starts.clear();
        self.starts.reserve(self.counts.len() + 1);
        let mut acc = 0u32;
        for &c in &self.counts {
            self.starts.push(acc);
            acc += c;
        }
        self.starts.push(acc);
        self.cursors.clear();
        self.cursors
            .extend_from_slice(&self.starts[..self.counts.len()]);
        self.body_order.clear();
        self.body_order.resize(acc as usize, 0);
        for i in 0..n_bodies {
            if !self.present[i] {
                continue;
            }
            let isl = self.body_island[i] as usize;
            let pos = self.cursors[isl] as usize;
            self.body_order[pos] = i;
            self.cursors[isl] += 1;
        }

        // Distribute into the per-island lists (reused: cleared, not
        // reallocated).
        resize_island_lists(
            &mut self.island_bodies,
            &mut self.spare_lists,
            self.island_count,
        );
        for isl in 0..self.island_count {
            let start = self.starts[isl] as usize;
            let end = self.starts[isl + 1] as usize;
            let list = &mut self.island_bodies[isl];
            list.clear();
            list.extend_from_slice(&self.body_order[start..end]);
        }

        // The constraint and joint lists are filled by the assign steps
        // below; size them to the island count so their per-island
        // buffers are reused too.
        resize_island_lists(
            &mut self.island_constraints,
            &mut self.spare_lists,
            self.island_count,
        );
        resize_island_lists(
            &mut self.island_joints,
            &mut self.spare_lists,
            self.island_count,
        );

        // Body -> island map.
        self.body_to_island.clear();
        self.body_to_island.resize(n_bodies, usize::MAX);
        for i in 0..n_bodies {
            if self.body_island[i] != NO_ISLAND {
                self.body_to_island[i] = self.body_island[i] as usize;
            }
        }
    }

    /// Assign contact indices to islands (input order within an island),
    /// using the map built by [`build`](Self::build).
    pub fn assign_constraints(&mut self, contact_pairs: &[(usize, usize)]) {
        let Self {
            body_to_island,
            counts,
            starts,
            cursors,
            constraint_order,
            island_constraints,
            ..
        } = self;
        counts.clear();
        counts.resize(island_constraints.len(), 0);
        assign_to_islands(
            body_to_island,
            counts,
            starts,
            cursors,
            constraint_order,
            island_constraints,
            contact_pairs,
        );
    }

    /// Assign joint indices to islands (input order within an island).
    pub fn assign_joints(&mut self, joint_pairs: &[(usize, usize)]) {
        let Self {
            body_to_island,
            counts,
            starts,
            cursors,
            joint_order,
            island_joints,
            ..
        } = self;
        counts.clear();
        counts.resize(island_joints.len(), 0);
        assign_to_islands(
            body_to_island,
            counts,
            starts,
            cursors,
            joint_order,
            island_joints,
            joint_pairs,
        );
    }
}

/// Resize a per-island list vector to `n` entries, recycling the
/// per-island index buffers through `spare`.
///
/// Growing takes a buffer from the spare pool (or a fresh one), and
/// shrinking returns the excess to it, so the per-island allocations
/// happen once and are then reused for the lifetime of the scratch.
fn resize_island_lists(lists: &mut Vec<Vec<usize>>, spare: &mut Vec<Vec<usize>>, n: usize) {
    while lists.len() > n {
        let mut v = lists.pop().expect("non-empty");
        v.clear();
        spare.push(v);
    }
    while lists.len() < n {
        let mut v = spare.pop().unwrap_or_default();
        v.clear();
        lists.push(v);
    }
}

/// Counting-sort `pairs` into the per-island lists.
///
/// `pairs[i]` is the index of the item (a contact or joint) and
/// `(a, b)` its two body endpoints. Items whose endpoints are not in
/// the same island are dropped (the same rule the serial path applies).
#[allow(clippy::too_many_arguments)]
fn assign_to_islands(
    body_to_island: &[usize],
    counts: &mut [u32],
    starts: &mut Vec<u32>,
    cursors: &mut Vec<u32>,
    order: &mut Vec<usize>,
    lists: &mut [Vec<usize>],
    pairs: &[(usize, usize)],
) {
    let n_islands = lists.len();
    debug_assert_eq!(
        counts.len(),
        n_islands,
        "count scratch must cover every island"
    );
    counts.fill(0);
    for &(a, b) in pairs {
        if a >= body_to_island.len() || b >= body_to_island.len() {
            continue;
        }
        let ia = body_to_island[a];
        let ib = body_to_island[b];
        if ia != ib || ia >= n_islands {
            continue;
        }
        counts[ia] += 1;
    }

    starts.clear();
    starts.reserve(n_islands + 1);
    let mut acc = 0u32;
    for &c in counts.iter() {
        starts.push(acc);
        acc += c;
    }
    starts.push(acc);
    cursors.clear();
    cursors.extend_from_slice(&starts[..n_islands]);

    order.clear();
    order.resize(acc as usize, 0);
    for (idx, &(a, _b)) in pairs.iter().enumerate() {
        if a >= body_to_island.len() {
            continue;
        }
        let isl = body_to_island[a];
        if isl >= n_islands {
            continue;
        }
        let pos = cursors[isl] as usize;
        order[pos] = idx;
        cursors[isl] += 1;
    }

    for isl in 0..n_islands {
        let start = starts[isl] as usize;
        let end = starts[isl + 1] as usize;
        let list = &mut lists[isl];
        list.clear();
        list.extend_from_slice(&order[start..end]);
    }
}

/// Union-find with path compression and union by rank.
fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Merge the sets of `a` and `b`.
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
    // Same assignment as the pooled path (see
    // `IslandScratch::assign_constraints`), built into fresh lists for
    // the caller.
    let mut scratch = IslandScratch::new();
    scratch.island_count = islands.len();
    scratch.island_constraints = vec![Vec::new(); islands.len()];
    scratch.body_to_island.clear();
    scratch.body_to_island.extend_from_slice(body_to_island);
    scratch.assign_constraints(contact_pairs);
    scratch.island_constraints
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

/// Debug-only disjointness verifier for the pooled path: every body in
/// at most one island. Same contract as [`debug_verify_disjoint`].
#[cfg(debug_assertions)]
pub fn debug_verify_disjoint_scratch(scratch: &IslandScratch, n_bodies: usize) {
    let mut seen = vec![false; n_bodies];
    for isl in 0..scratch.island_count {
        for &b in &scratch.island_bodies[isl] {
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
