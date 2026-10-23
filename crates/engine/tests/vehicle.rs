//! Suspension and arcade vehicles (I01–I08, mirrored Rust↔C#).
//!
//! These tests pin the stateless vehicle models: the 3D suspension
//! settles at its analytic compression (I01), full throttle drives
//! straight (I02) and steering turns right toward +X (I03), the brake
//! stops the car (I04), airborne wheels push nothing (I05), the 2D
//! arcade car drives (I06) with grip that the handbrake steals (I07),
//! and a drop from height bottoms the bump stops and recovers (I08).

use glam::{Vec2, Vec3};
use phieng_dynamics::{
    Shape2, Shape3, VehicleConfig2, VehicleConfig3, VehicleInput2, VehicleInput3, WheelMount2,
    WheelMount3, WheelTelemetry2, WheelTelemetry3, World2, World3,
};

fn bits(v: f32) -> u32 {
    v.to_bits()
}

const DT: f32 = 1.0 / 60.0;

/// A 1200 kg 3D car over a long plate, sleeping disabled.
fn car3() -> (World3, usize, VehicleConfig3) {
    let mut w = World3::new();
    w.gravity = Vec3::new(0.0, -9.81, 0.0);
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    w.collect_contact_events = true;
    w.add_static_body(
        Vec3::new(0.0, -0.5, 0.0),
        Shape3::Box {
            half_extents: Vec3::new(500.0, 0.5, 500.0),
        },
    )
    .expect("ground");
    let chassis = w
        .add_dynamic_body(
            Vec3::new(0.0, 1.0, 0.0),
            Shape3::Box {
                half_extents: Vec3::new(0.9, 0.35, 2.0),
            },
            240.0,
        )
        .expect("chassis");
    w.bodies[chassis].material.restitution = 0.0;
    w.bodies[0].material.restitution = 0.0;
    let mut cfg = VehicleConfig3::default();
    for (k, (x, z, steer)) in [
        (0.8, 1.2, true),
        (-0.8, 1.2, true),
        (0.8, -1.2, false),
        (-0.8, -1.2, false),
    ]
    .iter()
    .enumerate()
    {
        cfg.wheels[k] = WheelMount3 {
            offset: Vec3::new(*x, -0.35, *z),
            steer: *steer,
            drive: true,
        };
    }
    cfg.wheel_count = 4;
    (w, chassis, cfg)
}

/// A 1200 kg 2D car in a gravity-free top-down world.
fn car2() -> (World2, usize, VehicleConfig2) {
    let mut w = World2::new();
    w.gravity = Vec2::new(0.0, 0.0);
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    let chassis = w
        .add_dynamic_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(1.0, 0.5),
            },
            600.0,
        )
        .expect("chassis");
    let mut cfg = VehicleConfig2::default();
    for (k, (x, y, steer, drive)) in [
        (1.2, 0.5, true, false),
        (1.2, -0.5, true, false),
        (-1.2, 0.5, false, true),
        (-1.2, -0.5, false, true),
    ]
    .iter()
    .enumerate()
    {
        cfg.wheels[k] = WheelMount2 {
            offset: Vec2::new(*x, *y),
            steer: *steer,
            drive: *drive,
        };
    }
    cfg.wheel_count = 4;
    (w, chassis, cfg)
}

/// I01: the suspension settles at the analytic static compression.
#[test]
fn i01_the_suspension_settles_at_its_static_compression() {
    let (mut w, chassis, cfg) = car3();
    let input = VehicleInput3::default();
    let mut tele = [WheelTelemetry3::default(); 8];
    for _ in 0..600 {
        assert_eq!(w.step_vehicle3(chassis, &cfg, &input, DT, &mut tele), 4);
        w.step(DT);
    }
    // Every wheel shares the weight: x0 = m g / (4 k), as a fraction
    // of the rest length.
    let m = w.bodies[chassis].mass;
    let x0 = m * 9.81 / (4.0 * cfg.spring_k * cfg.rest_length);
    assert_eq!(bits(tele[0].compression01), 0x3ec5_41a8);
    assert!((tele[0].compression01 - x0).abs() / x0 < 1e-3);
    for t in tele[..4].iter() {
        assert!(t.in_contact);
        assert!((t.compression01 - x0).abs() / x0 < 1e-3);
    }
}

/// I02: full throttle drives the car straight up to speed.
#[test]
fn i02_full_throttle_drives_straight() {
    let (mut w, chassis, cfg) = car3();
    let input = VehicleInput3 {
        throttle: 1.0,
        ..Default::default()
    };
    let mut tele = [WheelTelemetry3::default(); 8];
    for _ in 0..600 {
        w.step_vehicle3(chassis, &cfg, &input, DT, &mut tele);
        w.step(DT);
    }
    let v = w.bodies[chassis].velocity;
    let p = w.bodies[chassis].transform.position;
    assert_eq!(bits(v.length()), 0x422e_9cdd);
    assert!(v.length() > 30.0, "full throttle must reach speed");
    // Straight: the velocity is the forward speed, nothing sideways.
    assert!((v.z - v.length()).abs() < 1e-3);
    assert!(p.x.abs() < 0.5, "no steering means no sideways drift");
    assert!(p.y > 0.5, "the car stays on its wheels");
}

/// I03: steering turns the car right, toward +X.
#[test]
fn i03_steering_turns_right() {
    let (mut w, chassis, cfg) = car3();
    let input = VehicleInput3 {
        throttle: 0.6,
        steering: 1.0,
        ..Default::default()
    };
    let mut tele = [WheelTelemetry3::default(); 8];
    for _ in 0..120 {
        w.step_vehicle3(chassis, &cfg, &input, DT, &mut tele);
        w.step(DT);
    }
    let fwd: Vec3 = w.bodies[chassis].transform.rotation * Vec3::Z;
    let yaw = w.bodies[chassis].angular_velocity.y;
    assert_eq!(bits(fwd.x), 0x3f74_6fcb);
    assert_eq!(bits(yaw), 0x3f8b_0642);
    assert!(fwd.x > 0.5, "full lock must turn hard right");
    assert!(yaw > 0.5, "still yawing right");
}

/// I04: the brake stops the car.
#[test]
fn i04_the_brake_stops_the_car() {
    let (mut w, chassis, cfg) = car3();
    w.collect_contact_events = true;
    let go = VehicleInput3 {
        throttle: 1.0,
        ..Default::default()
    };
    let mut tele = [WheelTelemetry3::default(); 8];
    for _ in 0..400 {
        w.step_vehicle3(chassis, &cfg, &go, DT, &mut tele);
        w.step(DT);
    }
    assert!(w.bodies[chassis].velocity.length() > 10.0);
    // Half brake: full lock dives the nose onto the ground,
    // and the test is about braking, not belly grazes.
    let stop = VehicleInput3 {
        brake: 0.5,
        ..Default::default()
    };
    let mut ev_total = 0usize;
    for _ in 0..400 {
        w.step_vehicle3(chassis, &cfg, &stop, DT, &mut tele);
        w.step(DT);
        ev_total += w.contact_events().len();
    }
    let v1 = w.bodies[chassis].velocity.length();
    assert_eq!(bits(v1), 0x3d98_7e50);
    // Contact-free: a belly graze would hand the stop to the contact
    // solver, which resolves sub-penetration touches last-bit
    // differently across mirrors (documented follow-up).
    assert_eq!(ev_total, 0);
    assert!(v1 < 0.6, "the brake must (nearly) stop the car, got {v1}");
}

/// I05: airborne wheels push nothing and report the miss.
#[test]
fn i05_airborne_wheels_push_nothing() {
    let (mut w, chassis, cfg) = car3();
    w.bodies[chassis].transform.position.y = 50.0;
    let input = VehicleInput3 {
        throttle: 1.0,
        ..Default::default()
    };
    let mut tele = [WheelTelemetry3::default(); 8];
    let before = w.bodies[chassis].velocity;
    assert_eq!(w.step_vehicle3(chassis, &cfg, &input, DT, &mut tele), 4);
    assert_eq!(w.bodies[chassis].velocity, before);
    for t in tele[..4].iter() {
        assert!(!t.in_contact);
        assert_eq!(t.compression01, 0.0);
        assert_eq!(t.normal_load, 0.0);
        assert_eq!(t.force_long, 0.0);
        assert_eq!(t.force_lat, 0.0);
    }
}

/// I06: the 2D arcade car drives along its heading.
#[test]
fn i06_the_arcade_car_drives() {
    let (mut w, chassis, cfg) = car2();
    let input = VehicleInput2 {
        throttle: 1.0,
        ..Default::default()
    };
    let mut tele = [WheelTelemetry2::default(); 8];
    for _ in 0..600 {
        assert_eq!(w.step_vehicle(chassis, &cfg, &input, DT, &mut tele), 4);
        w.step(DT);
    }
    let v = w.bodies[chassis].velocity;
    assert_eq!(bits(v.length()), 0x4228_2963);
    assert!(v.length() > 30.0);
    assert!((v.x - v.length()).abs() < 1e-3, "straight along +X");
}

/// I07: lateral grip kills a slide; the handbrake lets it live while
/// stopping the car.
#[test]
fn i07_grip_kills_slides_and_the_handbrake_steals_grip() {
    let shove = |handbrake: f32| {
        let (mut w, chassis, cfg) = car2();
        w.bodies[chassis].velocity = Vec2::new(6.0, 4.0);
        let input = VehicleInput2 {
            handbrake,
            ..Default::default()
        };
        let mut tele = [WheelTelemetry2::default(); 8];
        for _ in 0..120 {
            w.step_vehicle(chassis, &cfg, &input, DT, &mut tele);
            w.step(DT);
        }
        let b = &w.bodies[chassis];
        let fwd = b.transform.rotation * Vec2::X;
        let lat = Vec2::new(-fwd.y, fwd.x);
        (b.velocity.dot(lat), b.velocity.length())
    };
    let (lat_plain, speed_plain) = shove(0.0);
    let (lat_hb, speed_hb) = shove(1.0);
    assert_eq!(bits(lat_plain), 0x3e86_6a1c);
    assert_eq!(bits(lat_hb), 0x3f87_8691);
    assert!(lat_plain.abs() < 0.5, "grip must kill the slide");
    assert!(
        lat_hb.abs() > lat_plain.abs(),
        "no rear grip means more slide"
    );
    assert!(speed_hb < speed_plain, "the handbrake must slow the car");
}

/// I08: a drop from height bottoms the bump stops and recovers.
#[test]
fn i08_bottoming_caps_the_landing() {
    let (mut w, chassis, cfg) = car3();
    w.bodies[chassis].transform.position.y = 5.0;
    let input = VehicleInput3::default();
    let mut tele = [WheelTelemetry3::default(); 8];
    let mut max_comp = 0.0f32;
    let mut max_force = 0.0f32;
    let mut ev_total = 0usize;
    for _ in 0..600 {
        w.step_vehicle3(chassis, &cfg, &input, DT, &mut tele);
        w.step(DT);
        ev_total += w.contact_events().len();
        for t in tele[..4].iter() {
            max_comp = max_comp.max(t.compression01);
            max_force = max_force.max(t.normal_load);
        }
    }
    let y = w.bodies[chassis].transform.position.y;
    assert_eq!(bits(y), 0x3f65_2900);
    assert_eq!(bits(max_force), 0x46bc_8629);
    // Contact-free, like I04: the pins cover suspension only.
    assert_eq!(ev_total, 0);
    assert_eq!(bits(y), 0x3f65_2900);
    assert_eq!(bits(max_force), 0x46bc_8629);
    assert_eq!(max_comp, 1.0, "the landing must bottom the suspension");
    assert!(
        max_force <= cfg.max_suspension_force,
        "the bump stop caps the force"
    );
    assert!(y > 0.5 && y < 1.5, "recovered, not bounced to orbit");
}
