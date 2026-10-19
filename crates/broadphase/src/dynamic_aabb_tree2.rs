use glam::Vec2;
use phieng_math::AABB2;

const NULL_NODE: usize = usize::MAX;

#[derive(Debug, Clone, Copy)]
struct TreeNode2 {
    aabb: AABB2,
    parent: usize,
    left_child: usize,
    right_child: usize,
    /// Body id for leaf nodes, [`NULL_NODE`] for internal nodes.
    body_id: usize,
    height: i32,
}

/// Dynamic AABB tree broadphase for 2D.
///
/// A binary tree of AABBs where internal nodes store the merged bounds of
/// their children. Insertion adds a leaf, attaches it beside the best
/// existing leaf (minimum surface-area growth), then refits ancestors.
///
/// Best for large scenes (5000+ bodies) and clustered distributions where
/// a spatial hash degenerates (many bodies per cell).
///
/// Pair collection walks every internal node once, testing its two child
/// subtrees against each other, then recurses into both children. Output is
/// sorted and deduped, so pair order is deterministic regardless of tree
/// shape.
///
/// [`World2`](https://docs.rs/phieng_dynamics) rebuilds the tree every step
/// (`clear` + `insert`), so [`remove`](Self::remove) is only used by direct
/// API callers; it rebuilds from the surviving leaves, which is O(n log n)
/// but keeps the index bookkeeping trivial and obviously correct.
#[derive(Debug, Default)]
pub struct DynamicAABBTree2 {
    nodes: Vec<TreeNode2>,
    root: usize,
    count: usize,
}

impl DynamicAABBTree2 {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: NULL_NODE,
            count: 0,
        }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root = NULL_NODE;
        self.count = 0;
    }

    /// Insert a body. Non-finite bounds are dropped (they cannot be
    /// ordered, and would poison every ancestor merge).
    pub fn insert(&mut self, body_id: usize, min: Vec2, max: Vec2) {
        if !min.x.is_finite() || !min.y.is_finite() || !max.x.is_finite() || !max.y.is_finite() {
            return;
        }
        self.insert_aabb(body_id, AABB2::new(min, max));
    }

    fn insert_aabb(&mut self, body_id: usize, aabb: AABB2) {
        let leaf = self.nodes.len();
        self.nodes.push(TreeNode2 {
            aabb,
            parent: NULL_NODE,
            left_child: NULL_NODE,
            right_child: NULL_NODE,
            body_id,
            height: 0,
        });
        self.count += 1;

        if self.root == NULL_NODE {
            self.root = leaf;
            return;
        }

        // Descend to the leaf whose subtree grows least by absorbing `aabb`.
        let mut index = self.root;
        while self.nodes[index].left_child != NULL_NODE {
            let left = self.nodes[index].left_child;
            let right = self.nodes[index].right_child;

            let left_growth =
                self.nodes[left].aabb.merge(&aabb).volume() - self.nodes[left].aabb.volume();
            let right_growth =
                self.nodes[right].aabb.merge(&aabb).volume() - self.nodes[right].aabb.volume();

            if left_growth < right_growth {
                index = left;
            } else if right_growth < left_growth {
                index = right;
            } else {
                // Deterministic tie-break: prefer the smaller subtree.
                index = if self.nodes[left].aabb.volume() <= self.nodes[right].aabb.volume() {
                    left
                } else {
                    right
                };
            }
        }

        let sibling = index;
        let old_parent = self.nodes[sibling].parent;

        let internal = self.nodes.len();
        self.nodes.push(TreeNode2 {
            aabb: self.nodes[sibling].aabb.merge(&aabb),
            parent: old_parent,
            left_child: sibling,
            right_child: leaf,
            body_id: NULL_NODE,
            height: self.nodes[sibling].height + 1,
        });

        self.nodes[sibling].parent = internal;
        self.nodes[leaf].parent = internal;

        if old_parent == NULL_NODE {
            self.root = internal;
        } else if self.nodes[old_parent].left_child == sibling {
            self.nodes[old_parent].left_child = internal;
        } else {
            self.nodes[old_parent].right_child = internal;
        }

        self.refit_ancestors(internal);
        self.rebalance_if_degenerate();
    }

    /// Remove a body. No-op if the body is not present.
    ///
    /// Marks the leaf and rebuilds the whole tree bottom-up from the
    /// surviving leaves, rather than splicing the node out of the middle
    /// of `nodes` (which would invalidate every index that pointed at a
    /// moved node).
    pub fn remove(&mut self, body_id: usize) {
        let present = self
            .nodes
            .iter()
            .any(|n| n.left_child == NULL_NODE && n.body_id == body_id);
        if !present {
            return;
        }
        for n in &mut self.nodes {
            if n.left_child == NULL_NODE && n.body_id == body_id {
                n.body_id = NULL_NODE;
            }
        }
        self.rebuild_bottom_up();
    }

    /// Rebuild the tree bottom-up from the current leaves.
    ///
    /// Monotonically inserted far-apart bodies (the common "spawn a stream
    /// to the right" pattern) descend the same branch on every insert, so
    /// the SAH descent alone degenerates into a linked list — O(n) queries
    /// after n inserts, and deep enough to overflow the stack in the
    /// recursive pair walk. Re-inserting the same leaves in the same order
    /// reproduces the same shape, so the rebuild has to change the
    /// *construction*: sort leaves by centroid along the widest axis, then
    /// pair them up level by level. That guarantees height ~log2(n)
    /// regardless of insertion order, and is the same approach Box2D uses.
    fn rebuild_bottom_up(&mut self) {
        let leaves: Vec<(usize, AABB2)> = self
            .nodes
            .iter()
            .filter(|n| n.left_child == NULL_NODE && n.body_id != NULL_NODE)
            .map(|n| (n.body_id, n.aabb))
            .collect();
        if leaves.is_empty() {
            self.clear();
            return;
        }

        // Sort by centroid along whichever axis the bodies spread over
        // most, so each level's merge volume stays small.
        let mut lo = Vec2::splat(f32::INFINITY);
        let mut hi = Vec2::splat(f32::NEG_INFINITY);
        for (_, aabb) in &leaves {
            let c = aabb.center();
            lo = lo.min(c);
            hi = hi.max(c);
        }
        let by_x = (hi.x - lo.x) >= (hi.y - lo.y);
        let mut leaves = leaves;
        leaves.sort_by(|a, b| {
            let ca = a.1.center();
            let cb = b.1.center();
            let ka = if by_x { ca.x } else { ca.y };
            let kb = if by_x { cb.x } else { cb.y };
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        });

        self.clear();

        // One leaf node per body.
        let base = self.nodes.len();
        for (id, aabb) in &leaves {
            self.nodes.push(TreeNode2 {
                aabb: *aabb,
                parent: NULL_NODE,
                left_child: NULL_NODE,
                right_child: NULL_NODE,
                body_id: *id,
                height: 0,
            });
        }
        self.count = leaves.len();

        // Pair adjacent nodes level by level; a lone trailing node is
        // promoted unchanged so no leaf is ever dropped.
        let mut level: Vec<usize> = (base..base + leaves.len()).collect();
        while level.len() > 1 {
            let mut next: Vec<usize> = Vec::with_capacity(level.len().div_ceil(2));
            let mut i = 0;
            while i < level.len() {
                let l = level[i];
                if i + 1 < level.len() {
                    let r = level[i + 1];
                    let internal = self.nodes.len();
                    let merged = self.nodes[l].aabb.merge(&self.nodes[r].aabb);
                    let height = self.nodes[l].height.max(self.nodes[r].height) + 1;
                    self.nodes.push(TreeNode2 {
                        aabb: merged,
                        parent: NULL_NODE,
                        left_child: l,
                        right_child: r,
                        body_id: NULL_NODE,
                        height,
                    });
                    self.nodes[l].parent = internal;
                    self.nodes[r].parent = internal;
                    next.push(internal);
                } else {
                    next.push(l);
                }
                i += 2;
            }
            level = next;
        }
        self.root = level[0];
        self.nodes[self.root].parent = NULL_NODE;
    }

    /// Rebuild when the tree is far enough past ideal depth to be worth
    /// the O(n log n) sort (see [`Self::rebuild_bottom_up`]).
    fn rebalance_if_degenerate(&mut self) {
        if self.count < 2 {
            return;
        }
        let height = self.nodes[self.root].height + 1;
        let ideal = (usize::BITS - self.count.leading_zeros()) as i32;
        if height <= 2 * ideal + 2 {
            return;
        }
        self.rebuild_bottom_up();
    }

    /// Append every overlapping AABB pair to `out` without allocating.
    ///
    /// Contract: `out` must be empty (the world passes cleared scratch).
    pub fn collect_pairs(&self, out: &mut Vec<(usize, usize)>) {
        debug_assert!(out.is_empty());
        if self.root == NULL_NODE {
            return;
        }
        self.collect_pairs_recursive(self.root, out);
        out.sort_unstable();
        out.dedup();
    }

    fn collect_pairs_recursive(&self, index: usize, out: &mut Vec<(usize, usize)>) {
        let node = self.nodes[index];
        if node.left_child == NULL_NODE {
            return;
        }

        let left = node.left_child;
        let right = node.right_child;

        // Pairs spanning the two child subtrees.
        self.find_overlaps(left, right, out);
        // Pairs entirely inside one child subtree.
        self.collect_pairs_recursive(left, out);
        self.collect_pairs_recursive(right, out);
    }

    fn find_overlaps(&self, a: usize, b: usize, out: &mut Vec<(usize, usize)>) {
        let node_a = self.nodes[a];
        let node_b = self.nodes[b];

        if !node_a.aabb.intersects(&node_b.aabb) {
            return;
        }

        let a_leaf = node_a.left_child == NULL_NODE;
        let b_leaf = node_b.left_child == NULL_NODE;

        if a_leaf && b_leaf {
            let id_a = node_a.body_id;
            let id_b = node_b.body_id;
            out.push(if id_a < id_b {
                (id_a, id_b)
            } else {
                (id_b, id_a)
            });
            return;
        }

        // Descend into the taller subtree so the walk stays shallow
        // (standard: always split the deeper side). A leaf has no
        // children, so it can never be the side we split.
        if !a_leaf && (b_leaf || node_a.height >= node_b.height) {
            self.find_overlaps(node_a.left_child, b, out);
            self.find_overlaps(node_a.right_child, b, out);
        } else {
            self.find_overlaps(a, node_b.left_child, out);
            self.find_overlaps(a, node_b.right_child, out);
        }
    }

    fn refit_ancestors(&mut self, mut index: usize) {
        while index != NULL_NODE {
            let node = self.nodes[index];
            let left = node.left_child;
            let right = node.right_child;
            if left != NULL_NODE && right != NULL_NODE {
                self.nodes[index].aabb = self.nodes[left].aabb.merge(&self.nodes[right].aabb);
                self.nodes[index].height =
                    self.nodes[left].height.max(self.nodes[right].height) + 1;
            }
            index = node.parent;
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Tree height in nodes (0 when empty). Used by tests and diagnostics
    /// to assert the structure stays balanced-ish after rebuilds.
    pub fn height(&self) -> i32 {
        if self.root == NULL_NODE {
            0
        } else {
            self.nodes[self.root].height + 1
        }
    }
}

impl Clone for DynamicAABBTree2 {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            root: self.root,
            count: self.count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brute_force2::BruteForce2;

    #[test]
    fn test_dynamic_aabb_tree2_pairs() {
        let mut tree = DynamicAABBTree2::new();
        tree.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        tree.insert(1, Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        tree.insert(2, Vec2::new(10.0, 10.0), Vec2::new(11.0, 11.0));

        let mut pairs = Vec::new();
        tree.collect_pairs(&mut pairs);
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn test_dynamic_aabb_tree2_no_collision() {
        let mut tree = DynamicAABBTree2::new();
        tree.insert(0, Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        tree.insert(1, Vec2::new(5.0, 5.0), Vec2::new(6.0, 6.0));

        let mut pairs = Vec::new();
        tree.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_dynamic_aabb_tree2_remove() {
        let mut tree = DynamicAABBTree2::new();
        tree.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        tree.insert(1, Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        tree.remove(0);

        let mut pairs = Vec::new();
        tree.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_dynamic_aabb_tree2_remove_root() {
        let mut tree = DynamicAABBTree2::new();
        tree.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        tree.remove(0);
        assert!(tree.is_empty());
        assert_eq!(tree.root, NULL_NODE);
    }

    #[test]
    fn test_dynamic_aabb_tree2_insert_after_remove() {
        let mut tree = DynamicAABBTree2::new();
        tree.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        tree.insert(1, Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        tree.remove(0);
        tree.insert(2, Vec2::new(0.5, 0.5), Vec2::new(1.5, 1.5));

        let mut pairs = Vec::new();
        tree.collect_pairs(&mut pairs);
        assert_eq!(pairs, vec![(1, 2)]);
    }

    #[test]
    fn test_dynamic_aabb_tree2_remove_middle_of_many() {
        let mut tree = DynamicAABBTree2::new();
        // 12 well-separated leaves: enough depth that the removed leaf is
        // never the last slot, exercising the rebuild path.
        for i in 0..12 {
            let x = i as f32 * 10.0;
            tree.insert(i, Vec2::new(x, 0.0), Vec2::new(x + 1.0, 1.0));
        }
        for i in 0..12 {
            tree.remove(i);
            let mut pairs = Vec::new();
            tree.collect_pairs(&mut pairs);
            // Remaining bodies are all far apart, so no pairs survive.
            assert!(
                pairs.is_empty(),
                "step {} should leave no overlapping pairs",
                i
            );
            assert_eq!(tree.len(), 11 - i);
        }
        assert!(tree.is_empty());
    }

    #[test]
    fn test_dynamic_aabb_tree2_matches_brute_force_on_cluster() {
        let mut tree = DynamicAABBTree2::new();
        let mut bf = BruteForce2::new();
        // Deterministic pseudo-cluster: 40 bodies in 5 tight groups.
        for i in 0..40usize {
            let g = (i % 5) as f32;
            let k = (i / 5) as f32;
            let min = Vec2::new(g * 50.0 + k * 0.5, k * 0.5);
            let max = Vec2::new(g * 50.0 + k * 0.5 + 2.0, k * 0.5 + 2.0);
            tree.insert(i, min, max);
            bf.insert(i, min, max);
        }

        let mut a = Vec::new();
        tree.collect_pairs(&mut a);
        let mut b = Vec::new();
        bf.collect_pairs(&mut b);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }

    #[test]
    fn test_dynamic_aabb_tree2_rejects_non_finite() {
        let mut tree = DynamicAABBTree2::new();
        tree.insert(0, Vec2::new(f32::NAN, 0.0), Vec2::new(1.0, 1.0));
        assert!(tree.is_empty());
        let mut pairs = Vec::new();
        tree.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_dynamic_aabb_tree2_stays_balanced() {
        let mut tree = DynamicAABBTree2::new();
        // Insert far-apart bodies: a naive build would degenerate to a
        // linked list; the SAH descent keeps it shallow.
        for i in 0..64usize {
            let x = i as f32 * 100.0;
            tree.insert(i, Vec2::new(x, 0.0), Vec2::new(x + 1.0, 1.0));
        }
        assert_eq!(tree.len(), 64);
        assert!(
            tree.height() <= 16,
            "tree degenerate: height {} for 64 leaves",
            tree.height()
        );
    }
}
