//! A 3D kinematic character controller: move and slide.
//!
//! The character is a sphere that the caller places (kinematic): the
//! controller advances it by a desired translation, stops at the first
//! surface, and slides the remaining motion along that surface, up to a
//! fixed number of contacts per call. It never integrates gravity — it
//! reports `grounded` and the ground normal, and the caller decides
//! whether to apply gravity, jump, or walk.
//!
//! Contacts come from the closest-point query, exactly as in 2D: the
//! nearest surface point's outward normal is the separating direction,
//! both outside a surface and deep inside a shape. The sphere body is
//! deliberate: sphere-versus-shape is what the query covers analytically
//! (sphere, box, capsule). A capsule body needs a capsule-versus-shape
//! distance query, which does not exist yet — see `LIMITATIONS.md`.
//!
//! The convex hull and convex mesh report no closest point, so they are
//! walked through (the same documented gap as the ray cast). Give a
//! level's collision geometry a primitive shape to make it solid here.

use glam::Vec3;

use super::body::QueryFilter;
use super::shape_query::closest_point3;
use super::world::World3;

/// The character's size and feel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterConfig3 {
    /// The sphere's radius.
    pub radius: f32,
    /// The cosine of the steepest walkable slope.
    pub max_slope_cos: f32,
    /// How far below the feet to probe for ground.
    pub ground_probe: f32,
    /// The maximum number of contacts resolved per move.
    pub max_slides: u32,
}

impl Default for CharacterConfig3 {
    fn default() -> Self {
        Self {
            radius: 0.4,
            // 50 degrees.
            max_slope_cos: 0.642_787_6,
            ground_probe: 0.02,
            max_slides: 4,
        }
    }
}

/// What one move did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterMove3 {
    /// The translation actually applied.
    pub translation: Vec3,
    /// Whether a walkable surface is under the feet.
    pub grounded: bool,
    /// The ground normal (up when not grounded).
    pub ground_normal: Vec3,
    /// The number of contacts the move resolved.
    pub contacts: u32,
}

/// The blocking contact of a sphere at `center` moving by `delta`.
///
/// A contact only blocks motion that pushes *into* the surface
/// (`delta · normal < 0`), which is what lets a character resting on a
/// surface move: standing on a closed box counts as "inside" it.
/// Picks the deepest blocking contact; ties keep the earlier body.
fn blocking_contact(
    world: &World3,
    center: Vec3,
    radius: f32,
    delta: Vec3,
    filter: &QueryFilter,
) -> Option<(Vec3, f32)> {
    let mut best: Option<(Vec3, f32)> = None;
    for body in &world.bodies {
        if !filter.accepts(body.collision_group, body.collision_mask) {
            continue;
        }
        let Some(shape) = body.shape.as_ref() else {
            continue;
        };
        let Some(hit) = closest_point3(
            shape,
            body.transform.position,
            body.transform.rotation,
            center,
        ) else {
            continue;
        };
        let slack = radius * 1e-4;
        let penetration = if hit.inside {
            radius + hit.distance
        } else if hit.distance < radius - slack {
            radius - hit.distance
        } else {
            continue;
        };
        if delta.dot(hit.normal) >= 0.0 {
            continue;
        }
        if best.map(|(_, d)| penetration > d).unwrap_or(true) {
            best = Some((hit.normal, penetration));
        }
    }
    best
}

/// Move a kinematic sphere character by `desired`, sliding along
/// contacts. See the module docs for the contract.
pub fn move_character3(
    world: &World3,
    start: Vec3,
    config: &CharacterConfig3,
    desired: Vec3,
    filter: &QueryFilter,
) -> CharacterMove3 {
    let radius = config.radius.max(1e-4);
    let mut position = start;
    let mut translation = Vec3::ZERO;
    let mut contacts = 0u32;
    let slides = config.max_slides.max(1);

    // Substep so a fast move cannot tunnel a thin wall.
    let length = desired.length();
    let steps = if length <= radius * 0.5 {
        1
    } else {
        ((length / (radius * 0.5)).ceil() as u32).clamp(2, 64)
    };
    let step = desired / steps as f32;

    for _ in 0..steps {
        let mut remaining = step;
        for _ in 0..slides {
            if remaining.length_squared() <= 1e-12 {
                break;
            }
            if blocking_contact(world, position + remaining, radius, remaining, filter).is_none() {
                position += remaining;
                translation += remaining;
                break;
            }
            let mut lo = 0.0f32;
            let mut hi = 1.0f32;
            for _ in 0..10 {
                let mid = 0.5 * lo + 0.5 * hi;
                if blocking_contact(world, position + remaining * mid, radius, remaining, filter)
                    .is_none()
                {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let advance = remaining * lo;
            position += advance;
            translation += advance;
            // The advanced fraction is spent: slide only the remainder.
            remaining -= advance;
            contacts += 1;

            let probe = remaining * (lo + 1e-3);
            let Some((normal, _)) =
                blocking_contact(world, position + probe, radius, remaining, filter)
            else {
                break;
            };
            let into = remaining.dot(normal);
            if into >= 0.0 {
                continue;
            }
            remaining -= normal * into;
            position += normal * 1e-4;
            translation += normal * 1e-4;
        }
    }

    // Ground probe: a contact just below the feet that opposes downward
    // motion (its normal points up), walkable if steep enough.
    let mut grounded = false;
    let mut ground_normal = Vec3::Y;
    if let Some((normal, _)) = blocking_contact(
        world,
        position - Vec3::Y * config.ground_probe,
        radius,
        -Vec3::Y,
        filter,
    ) {
        if normal.y >= config.max_slope_cos {
            grounded = true;
            ground_normal = normal;
        }
    }

    CharacterMove3 {
        translation,
        grounded,
        ground_normal,
        contacts,
    }
}
