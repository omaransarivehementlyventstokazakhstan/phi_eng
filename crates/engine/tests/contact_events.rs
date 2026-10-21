//! Contact-event and sensor tests (Z01–Z08, mirrored Rust↔C#).
//!
//! A contact event says a pair began or stopped touching. A sensor
//! reports its contacts as events and never pushes: that is what a
//! trigger volume is. These tests pin:
//!   1. a pair that starts touching produces a *began* event, and one
//!      that stops touching produces an *ended* event,
//!   2. a pair that keeps touching produces no further events,
//!   3. a sensor falls through what it touches, and reports it,
//!   4. the events are off by default, and a scene that never reads
//!      them behaves exactly as before.

use phieng_dynamics::body::ContactEventKind;
use phieng_dynamics::{Shape2, Shape3, World2, World3};

const DT: f32 = 1.0 / 60.0;

/// A ball above a floor, with the events on.
fn scene_2d() -> (World2, usize) {
    let mut w = World2::new();
    w.gravity = glam::Vec2::new(0.0, -9.81);
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    w.collect_contact_events = true;
    w.add_static_body(
        glam::Vec2::new(0.0, -1.0),
        Shape2::Box {
            half_extents: glam::Vec2::new(10.0, 0.5),
        },
    )
    .expect("valid shape");
    let ball = w
        .add_dynamic_body(
            glam::Vec2::new(0.0, 2.0),
            Shape2::Circle { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");
    // No bounce: the ball comes to rest, so a test can tell "still
    // touching" from "touched again". The floor's restitution counts too
    // (the pair takes the higher one), so both are set.
    w.bodies[ball].material.restitution = 0.0;
    w.bodies[0].material.restitution = 0.0;
    (w, ball)
}

/// Z01: a pair that starts touching produces a *began* event.
#[test]
fn z01_began_event_when_a_pair_starts_touching() {
    let (mut w, ball) = scene_2d();
    // The ball is still in the air.
    w.step(DT);
    assert!(
        w.contact_events().is_empty(),
        "a pair in the air produces no event"
    );

    // Step until it lands.
    let mut landed = false;
    for _ in 0..240 {
        w.step(DT);
        if w.contact_events()
            .iter()
            .any(|e| e.body_a == 0 && e.body_b == ball as u32 && e.kind == ContactEventKind::Began)
        {
            landed = true;
            break;
        }
    }
    assert!(landed, "the ball must produce a began event when it lands");
}

/// Z02: a pair that stops touching produces an *ended* event.
#[test]
fn z02_ended_event_when_a_pair_stops_touching() {
    let (mut w, ball) = scene_2d();
    // Let it land.
    for _ in 0..240 {
        w.step(DT);
    }
    // Lift it out of the floor's reach.
    w.bodies[ball].transform.position = glam::Vec2::new(0.0, 50.0);
    w.bodies[ball].velocity = glam::Vec2::ZERO;

    let mut ended = false;
    for _ in 0..60 {
        w.step(DT);
        if w.contact_events()
            .iter()
            .any(|e| e.body_a == 0 && e.body_b == ball as u32 && e.kind == ContactEventKind::Ended)
        {
            ended = true;
            break;
        }
    }
    assert!(
        ended,
        "the pair must produce an ended event when it separates"
    );
}

/// Z03: a pair that keeps touching produces no further events.
#[test]
fn z03_a_resting_pair_produces_no_repeat_events() {
    let (mut w, _ball) = scene_2d();
    for _ in 0..240 {
        w.step(DT);
    }
    // The ball is resting on the floor now: the pair is still touching,
    // so no event fires.
    let mut repeats = 0;
    for _ in 0..60 {
        w.step(DT);
        repeats += w.contact_events().len();
    }
    assert_eq!(repeats, 0, "a resting pair must not fire events again");
}

/// Z04: a sensor reports its contacts and never pushes.
#[test]
fn z04_a_sensor_reports_without_pushing() {
    let (mut w, ball) = scene_2d();
    // The ball is a sensor: it falls through the floor.
    w.bodies[ball].is_sensor = true;

    let mut began = false;
    for _ in 0..240 {
        w.step(DT);
        if w.contact_events()
            .iter()
            .any(|e| e.body_a == 0 && e.body_b == ball as u32 && e.kind == ContactEventKind::Began)
        {
            began = true;
        }
    }
    assert!(began, "the sensor must report the contact it had");
    assert!(
        w.bodies[ball].transform.position.y < -1.0,
        "a sensor must fall through the floor (y={})",
        w.bodies[ball].transform.position.y
    );
}

/// Z05: a sensor's contact ends when it leaves.
#[test]
fn z05_a_sensor_reports_the_contact_ending() {
    let (mut w, ball) = scene_2d();
    w.bodies[ball].is_sensor = true;
    // The sensor falls through the floor; the contact ends on the way
    // down, and the event says so.
    let mut ended = false;
    for _ in 0..240 {
        w.step(DT);
        if w.contact_events()
            .iter()
            .any(|e| e.body_a == 0 && e.body_b == ball as u32 && e.kind == ContactEventKind::Ended)
        {
            ended = true;
        }
    }
    assert!(
        ended,
        "the sensor's contact must end when the pair separates"
    );
}

/// Z06: the same in 3D.
#[test]
fn z06_events_and_sensors_in_3d() {
    let mut w = World3::new();
    w.gravity = glam::Vec3::new(0.0, -9.81, 0.0);
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    w.collect_contact_events = true;
    w.add_static_body(
        glam::Vec3::new(0.0, -1.0, 0.0),
        Shape3::Box {
            half_extents: glam::Vec3::new(10.0, 0.5, 10.0),
        },
    )
    .expect("valid shape");
    let ball = w
        .add_dynamic_body(
            glam::Vec3::new(0.0, 2.0, 0.0),
            Shape3::Sphere { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");

    let mut began = false;
    for _ in 0..240 {
        w.step(DT);
        if w.contact_events()
            .iter()
            .any(|e| e.body_a == 0 && e.body_b == ball as u32 && e.kind == ContactEventKind::Began)
        {
            began = true;
        }
    }
    assert!(began, "the ball must produce a began event when it lands");

    // As a sensor it falls through.
    w.bodies[ball].is_sensor = true;
    for _ in 0..240 {
        w.step(DT);
    }
    assert!(
        w.bodies[ball].transform.position.y < -1.0,
        "a sensor must fall through the floor (y={})",
        w.bodies[ball].transform.position.y
    );
}

/// Z07: the events are off by default, and the scene is unchanged.
#[test]
fn z07_events_are_off_by_default() {
    let mut w = World2::new();
    w.gravity = glam::Vec2::new(0.0, -9.81);
    w.add_static_body(
        glam::Vec2::new(0.0, -1.0),
        Shape2::Box {
            half_extents: glam::Vec2::new(10.0, 0.5),
        },
    )
    .expect("valid shape");
    w.add_dynamic_body(
        glam::Vec2::new(0.0, 2.0),
        Shape2::Circle { radius: 0.5 },
        1.0,
    )
    .expect("valid shape");
    assert!(!w.collect_contact_events, "the events are off by default");
    for _ in 0..60 {
        w.step(DT);
    }
    assert!(
        w.contact_events().is_empty(),
        "no events are collected when they are off"
    );

    // A scene with the events on behaves the same: the events do not
    // change the simulation.
    let mut with_events = World2::new();
    with_events.gravity = glam::Vec2::new(0.0, -9.81);
    with_events.collect_contact_events = true;
    with_events
        .add_static_body(
            glam::Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: glam::Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    with_events
        .add_dynamic_body(
            glam::Vec2::new(0.0, 2.0),
            Shape2::Circle { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");
    for _ in 0..60 {
        with_events.step(DT);
    }
    assert_eq!(
        w.bodies[1].transform.position, with_events.bodies[1].transform.position,
        "collecting the events must not change the simulation"
    );
}

/// Z08: the native bridge exposes the sensor flag and the events.
#[test]
fn z08_ffi_sensor_flag_and_events() {
    use phieng::ffi::{
        phieng_world_add_body, phieng_world_free, phieng_world_new, phieng_world_set_body_sensor,
        phieng_world_step,
    };

    unsafe {
        let world = phieng_world_new();
        assert!(!world.is_null());
        let id = phieng_world_add_body(world, 0.0, 2.0, 0, 0.5, 0.0, 1.0, 0, 0.5, 0.0);
        assert!(id >= 0);
        assert_eq!(phieng_world_set_body_sensor(world, id as u32, 1), 1);
        assert_eq!(
            phieng_world_set_body_sensor(world, 9999, 1),
            0,
            "an unknown body must be refused"
        );
        phieng_world_step(world, DT);
        phieng_world_free(world);
    }
}
