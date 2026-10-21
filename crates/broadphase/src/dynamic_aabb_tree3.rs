use glam::Vec3;
use phieng_math::AABB3;

const NULL_NODE: usize = usize::MAX;

#[derive(Debug, Clone, Copy)]
struct TreeNode3 {
    aabb: AABB3,
    parent: usize,
    left_child: usize,
    right_child: usize,
    /// Body id for leaf nodes, [`NULL_NODE`] for internal nodes.
    body_id: usize,
    height: i32,
}

/// Dynamic AABB tree broadphase for 3D.
///
/// Same algorithm as the 2D tree with 3D volumes (`minimize the volume
/// growth when attaching a new leaf`). Best for large scenes and clustered
/// distributions. Pair output is sorted and deduped for determinism.
#[derive(Debug, Default)]
pub struct DynamicAABBTree3 {
    nodes: Vec<TreeNode3>,
    root: usize,
    count: usize,
}

impl DynamicAABBTree3 {
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
    pub fn insert(&mut self, body_id: usize, min: Vec3, max: Vec3) {
        if !min.x.is_finite()
            || !min.y.is_finite()
            || !min.z.is_finite()
            || !max.x.is_finite()
            || !max.y.is_finite()
            || !max.z.is_finite()
        {
            return;
        }
        self.insert_aabb(body_id, AABB3::new(min, max));
    }

    fn insert_aabb(&mut self, body_id: usize, aabb: AABB3) {
        let leaf = self.nodes.len();
        self.nodes.push(TreeNode3 {
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
        self.nodes.push(TreeNode3 {
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

    /// Remove a body. No-op if the body is not present. Rebuilds the
    /// whole tree bottom-up from the surviving leaves (which keeps all
    /// node indices valid without any splice bookkeeping).
    pub fn remove(&mut self, body_id: usize) {
        let present = self
            .nodes
            .iter()
            .any(|n| n.left_child == NULL_NODE && n.body_id == body_id);
        if !present {
            return;
        }
        // Drop the leaf's AABB by marking the node, then rebuild from
        // everything that is still a leaf. Avoids index-splice bookkeeping.
        for n in &mut self.nodes {
            if n.left_child == NULL_NODE && n.body_id == body_id {
                n.body_id = NULL_NODE;
            }
        }
        self.rebuild_bottom_up();
    }

    /// Append every overlapping AABB pair to `out` without allocating.
    /// Contract: `out` must be empty (the world passes cleared scratch).
    /// Append the ids of the bodies whose AABB overlaps `aabb`.
    ///
    /// The bodies whose AABB overlaps the query box, in ascending id
    /// order and without duplicates. The tree is walked from the
    /// root, skipping every subtree whose box is disjoint from the
    /// query box.
    pub fn query_aabb(&self, aabb: &AABB3, out: &mut Vec<usize>) {
        self.query_aabb_recursive(self.root, aabb, out);
        out.sort_unstable();
        out.dedup();
    }

    /// Walk the tree, collecting the leaves whose box overlaps the
    /// query box.
    fn query_aabb_recursive(&self, index: usize, aabb: &AABB3, out: &mut Vec<usize>) {
        if index == NULL_NODE {
            return;
        }
        let node = &self.nodes[index];
        if !node.aabb.intersects(aabb) {
            return;
        }
        if node.body_id != NULL_NODE {
            out.push(node.body_id);
        }
        self.query_aabb_recursive(node.left_child, aabb, out);
        self.query_aabb_recursive(node.right_child, aabb, out);
    }

    pub fn collect_pairs(&self, out: &mut Vec<(usize, usize)>) {
        debug_assert!(out.is_empty());
        if self.root == NULL_NODE {
            return;
        }
        self.collect_pairs_recursive(self.root, out);
        out.sort_unstable();
        out.dedup();
    }

    /// Same as [`Self::collect_pairs`]: the tree only ever emits pairs
    /// whose leaf AABBs overlap (every descent is guarded by an AABB
    /// overlap test), so its output is exactly the AABB-overlapping
    /// subsequence by construction. Provided so the narrowphase fast
    /// path is uniform across backends.
    pub fn collect_overlapping_pairs(&self, out: &mut Vec<(usize, usize)>) {
        self.collect_pairs(out);
    }

    fn collect_pairs_recursive(&self, index: usize, out: &mut Vec<(usize, usize)>) {
        let node = self.nodes[index];
        if node.left_child == NULL_NODE {
            return;
        }

        let left = node.left_child;
        let right = node.right_child;

        self.find_overlaps(left, right, out);
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

        // Descend into the taller subtree so the walk stays shallow. A
        // leaf has no children, so it can never be the side we split.
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

    /// Rebuild the tree bottom-up from the current leaves (see the 2D tree
    /// for the full rationale: monotone insertion degenerates the SAH
    /// descent, and re-inserting in the same order just reproduces the
    /// degenerate shape — the construction itself has to change).
    fn rebuild_bottom_up(&mut self) {
        let leaves: Vec<(usize, AABB3)> = self
            .nodes
            .iter()
            .filter(|n| n.left_child == NULL_NODE && n.body_id != NULL_NODE)
            .map(|n| (n.body_id, n.aabb))
            .collect();
        if leaves.is_empty() {
            self.clear();
            return;
        }

        // Sort by centroid along the axis of greatest spread.
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for (_, aabb) in &leaves {
            let c = aabb.center();
            lo = lo.min(c);
            hi = hi.max(c);
        }
        let spread = hi - lo;
        let axis = if spread.x >= spread.y && spread.x >= spread.z {
            0
        } else if spread.y >= spread.z {
            1
        } else {
            2
        };
        let mut leaves = leaves;
        leaves.sort_by(|a, b| {
            let ca = a.1.center();
            let cb = b.1.center();
            let ka = match axis {
                0 => ca.x,
                1 => ca.y,
                _ => ca.z,
            };
            let kb = match axis {
                0 => cb.x,
                1 => cb.y,
                _ => cb.z,
            };
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        });

        self.clear();

        let base = self.nodes.len();
        for (id, aabb) in &leaves {
            self.nodes.push(TreeNode3 {
                aabb: *aabb,
                parent: NULL_NODE,
                left_child: NULL_NODE,
                right_child: NULL_NODE,
                body_id: *id,
                height: 0,
            });
        }
        self.count = leaves.len();

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
                    self.nodes.push(TreeNode3 {
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
    /// the O(n log n) sort.
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

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Tree height in nodes (0 when empty). Used by tests and diagnostics.
    pub fn height(&self) -> i32 {
        if self.root == NULL_NODE {
            0
        } else {
            self.nodes[self.root].height + 1
        }
    }
}

impl Clone for DynamicAABBTree3 {
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
    use crate::brute_force3::BruteForce3;

    #[test]
    fn test_dynamic_aabb_tree3_pairs() {
        let mut tree = DynamicAABBTree3::new();
        tree.insert(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        tree.insert(1, Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));
        tree.insert(2, Vec3::new(10.0, 10.0, 10.0), Vec3::new(11.0, 11.0, 11.0));

        let mut pairs = Vec::new();
        tree.collect_pairs(&mut pairs);
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn test_dynamic_aabb_tree3_no_collision() {
        let mut tree = DynamicAABBTree3::new();
        tree.insert(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        tree.insert(1, Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));

        let mut pairs = Vec::new();
        tree.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_dynamic_aabb_tree3_remove() {
        let mut tree = DynamicAABBTree3::new();
        tree.insert(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        tree.insert(1, Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));
        tree.remove(0);

        let mut pairs = Vec::new();
        tree.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_dynamic_aabb_tree3_remove_root() {
        let mut tree = DynamicAABBTree3::new();
        tree.insert(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        tree.remove(0);
        assert!(tree.is_empty());
        assert_eq!(tree.root, NULL_NODE);
    }

    #[test]
    fn test_dynamic_aabb_tree3_matches_brute_force_on_cluster() {
        let mut tree = DynamicAABBTree3::new();
        let mut bf = BruteForce3::new();
        for i in 0..40usize {
            let g = (i % 5) as f32;
            let k = (i / 5) as f32;
            let min = Vec3::new(g * 50.0 + k * 0.5, k * 0.5, k * 0.25);
            let max = Vec3::new(g * 50.0 + k * 0.5 + 2.0, k * 0.5 + 2.0, k * 0.25 + 2.0);
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
    fn test_dynamic_aabb_tree3_rejects_non_finite() {
        let mut tree = DynamicAABBTree3::new();
        tree.insert(
            0,
            Vec3::new(0.0, f32::INFINITY, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        assert!(tree.is_empty());
    }

    #[test]
    fn test_dynamic_aabb_tree3_stays_balanced() {
        let mut tree = DynamicAABBTree3::new();
        // Monotone stream to the right: the degenerate-insertion pattern.
        for i in 0..64usize {
            let x = i as f32 * 100.0;
            tree.insert(i, Vec3::new(x, 0.0, 0.0), Vec3::new(x + 1.0, 1.0, 1.0));
        }
        assert_eq!(tree.len(), 64);
        assert!(
            tree.height() <= 16,
            "tree degenerate: height {} for 64 leaves",
            tree.height()
        );
    }

    #[test]
    fn test_dynamic_aabb_tree3_still_exact_after_balancing() {
        // Rotation changes tree shape, never the reported pair set.
        let mut tree = DynamicAABBTree3::new();
        let mut bf = BruteForce3::new();
        for i in 0..64usize {
            let x = i as f32 * 100.0;
            let min = Vec3::new(x, 0.0, 0.0);
            let max = Vec3::new(x + 150.0, 1.0, 1.0);
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
}
