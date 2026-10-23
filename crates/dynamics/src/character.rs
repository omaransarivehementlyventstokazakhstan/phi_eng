//! A 2D kinematic character controller: move and slide.
//!
//! The character is a circle that the caller places (kinematic): the
//! controller advances it by a desired translation, stops at the first
//! surface, and slides the remaining motion along that surface, up to a
//! fixed number of contacts per call. It never integrates gravity — it
//! reports `grounded` and the ground normal, and the caller decides
//! whether to apply gravity, jump, or walk.
//!
//! Contacts come from the closest-point query: the nearest surface
//! point's outward normal is exactly the direction that separates the
//! circle from the body, both when the circle overlaps a surface from
//! outside and when its centre is inside a shape (where the query
//! returns the least-penetration direction). That keeps one code path
//! for circles, boxes and polygons, and the managed mirror agrees bit
//! for bit because it calls the same query.
//!
//! The advance is a bisection, not a sweep: the controller finds the
//! largest fraction of the step that does not overlap, then slides. With
//! ten iterations that is 1/1024 of a step of precision, which is far
//! below the skin a caller should use.

use glam::Vec2;

use super::body::QueryFilter;
use super::shape_query::closest_point2;
use super::world::World2;

/// The character's size and feel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterConfig2 {
    /// The capsule's (circle's) radius.
    pub radius: f32,
    /// The cosine of the steepest walkable slope. A contact whose normal
    /// has a larger upward component counts as ground.
    pub max_slope_cos: f32,
    /// How far below the feet to probe for ground.
    pub ground_probe: f32,
    /// The maximum number of contacts resolved per move.
    pub max_slides: u32,
}

impl Default for CharacterConfig2 {
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
pub struct CharacterMove2 {
    /// The translation actually applied (never longer than `desired`
    /// along any direction; sliding shortens it).
    pub translation: Vec2,
    /// Whether a walkable surface is under the feet.
    pub grounded: bool,
    /// The ground normal (up when not grounded).
    pub ground_normal: Vec2,
    /// The number of contacts the move resolved.
    pub contacts: u32,
}

/// The blocking contact of a circle at `center` moving by `delta`.
///
/// A contact only blocks motion that pushes *into* the surface
/// (`delta · normal < 0`). That distinction is what lets a character
/// resting on the floor walk: standing on a closed box counts as
/// "inside" the box (the surface is part of the shape), and a naive
/// "inside means blocked" test would freeze it. With the sign test the
/// floor's escape normal is straight up and a horizontal step is free,
/// while a wall's escape normal opposes the step and blocks it.
///
/// Picks the deepest blocking contact, so the normal that wins is the
/// one most in the way. Ties keep the earlier body (deterministic).
fn blocking_contact(
    world: &World2,
    center: Vec2,
    radius: f32,
    delta: Vec2,
    filter: &QueryFilter,
) -> Option<(Vec2, f32)> {
    let mut best: Option<(Vec2, f32)> = None;
    for body in &world.bodies {
        if !filter.accepts(body.collision_group, body.collision_mask) {
            continue;
        }
        let Some(shape) = body.shape.as_ref() else {
            continue;
        };
        let Some(hit) = closest_point2(
            shape,
            body.transform.position,
            body.transform.rotation,
            center,
        ) else {
            continue;
        };
        // Inside always touches; outside touches within the radius. The
        // slack keeps a character resting exactly on a surface free to
        // move instead of deadlocking against a zero-depth contact.
        let slack = radius * 1e-4;
        let penetration = if hit.inside {
            radius + hit.distance
        } else if hit.distance < radius - slack {
            radius - hit.distance
        } else {
            continue;
        };
        if delta.dot(hit.normal) >= 0.0 {
            // Moving along or away from this surface.
            continue;
        }
        if best.map(|(_, d)| penetration > d).unwrap_or(true) {
            best = Some((hit.normal, penetration));
        }
    }
    best
}

/// Move a kinematic circle character by `desired`, sliding along
/// contacts. See the module docs for the contract.
///
/// `filter` selects which bodies are solid for the character: a body the
/// filter rejects is walked through, and sensors are solid unless the
/// filter excludes them (the caller can build a filter for that).
pub fn move_character2(
    world: &World2,
    start: Vec2,
    config: &CharacterConfig2,
    desired: Vec2,
    filter: &QueryFilter,
) -> CharacterMove2 {
    let radius = config.radius.max(1e-4);
    let mut position = start;
    let mut translation = Vec2::ZERO;
    let mut contacts = 0u32;
    let slides = config.max_slides.max(1);

    // Substep so a fast move cannot tunnel a thin wall: the endpoint
    // test alone would step clean over one. Half a radius per substep
    // means the midpoint of every substep is still inside any wall the
    // move crosses, which the bisection then finds.
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
            // Bisect for the largest free fraction of the step.
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

            // Read the blocking normal just past the free point (where
            // the shapes already overlap) and slide the rest along it.
            let probe = remaining * (lo + 1e-3);
            let Some((normal, _)) =
                blocking_contact(world, position + probe, radius, remaining, filter)
            else {
                break;
            };
            let into = remaining.dot(normal);
            if into >= 0.0 {
                // The remaining motion already points away: keep it.
                continue;
            }
            remaining -= normal * into;
            // A hair of separation keeps the next bisection from
            // starting in contact with the same surface.
            position += normal * 1e-4;
            translation += normal * 1e-4;
        }
    }

    // Ground probe: a contact just below the feet that opposes downward
    // motion (its normal points up), walkable if steep enough.
    let mut grounded = false;
    let mut ground_normal = Vec2::Y;
    if let Some((normal, _)) = blocking_contact(
        world,
        position - Vec2::Y * config.ground_probe,
        radius,
        -Vec2::Y,
        filter,
    ) {
        if normal.y >= config.max_slope_cos {
            grounded = true;
            ground_normal = normal;
        }
    }

    CharacterMove2 {
        translation,
        grounded,
        ground_normal,
        contacts,
    }
}
