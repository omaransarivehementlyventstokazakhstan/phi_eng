use glam::{Vec2, Vec3};

/// Axis-Aligned Bounding Box in 2D
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB2 {
    pub min: Vec2,
    pub max: Vec2,
}

impl AABB2 {
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    pub fn from_center_half_extents(center: Vec2, half_extents: Vec2) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    pub fn half_extents(&self) -> Vec2 {
        (self.max - self.min) * 0.5
    }

    pub fn size(&self) -> Vec2 {
        self.max - self.min
    }

    pub fn volume(&self) -> f32 {
        let size = self.size();
        size.x * size.y
    }

    pub fn contains(&self, point: Vec2) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }

    pub fn intersects(&self, other: &AABB2) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    pub fn merge(&self, other: &AABB2) -> AABB2 {
        AABB2 {
            min: Vec2::min(self.min, other.min),
            max: Vec2::max(self.max, other.max),
        }
    }

    pub fn expand(&self, amount: Vec2) -> AABB2 {
        AABB2 {
            min: self.min - amount,
            max: self.max + amount,
        }
    }
}

/// Axis-Aligned Bounding Box in 3D
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB3 {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB3 {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn from_center_half_extents(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn volume(&self) -> f32 {
        let size = self.size();
        size.x * size.y * size.z
    }

    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    pub fn intersects(&self, other: &AABB3) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    pub fn merge(&self, other: &AABB3) -> AABB3 {
        AABB3 {
            min: Vec3::min(self.min, other.min),
            max: Vec3::max(self.max, other.max),
        }
    }

    pub fn expand(&self, amount: Vec3) -> AABB3 {
        AABB3 {
            min: self.min - amount,
            max: self.max + amount,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    #[test]
    fn test_aabb2_contains() {
        let aabb = AABB2::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        assert!(aabb.contains(Vec2::new(0.5, 0.5)));
        assert!(!aabb.contains(Vec2::new(1.5, 0.5)));
    }

    #[test]
    fn test_aabb2_intersects() {
        let aabb1 = AABB2::new(Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        let aabb2 = AABB2::new(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        assert!(aabb1.intersects(&aabb2));
    }

    #[test]
    fn test_aabb2_merge() {
        let aabb1 = AABB2::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        let aabb2 = AABB2::new(Vec2::new(2.0, 2.0), Vec2::new(3.0, 3.0));
        let merged = aabb1.merge(&aabb2);
        assert_eq!(merged.min, Vec2::new(0.0, 0.0));
        assert_eq!(merged.max, Vec2::new(3.0, 3.0));
    }
}
