using System.Numerics;
using Xunit;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Contact-event and sensor tests (Z01–Z12): a contact event says a
/// pair began or stopped touching, and a sensor reports its contacts as
/// events and never pushes. Mirrors
/// <c>crates/engine/tests/contact_events.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class ContactEventTests
{
    private const float Dt = 1.0f / 60.0f;

    /// <summary>A ball above a floor, with the events on.</summary>
    private static PhiNativeWorld Scene()
    {
        var w = new PhiNativeWorld();
        w.SetGravity(new Vector2(0.0f, -9.81f));
        w.SetSleepThresholds(0.0f, 0.0f);
        w.SetCollectContactEvents(true);
        w.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        w.AddDynamicBall(new Vector2(0.0f, 2.0f), 0.5f, 1.0f);
        return w;
    }

    private static bool HasEvent(
        (uint BodyA, uint BodyB, int Kind, Vector2 Point, Vector2 Normal)[] events,
        uint a, uint b, int kind)
        => Array.Exists(events, e => e.BodyA == a && e.BodyB == b && e.Kind == kind);

    /// <summary>Whether the world holds a contact between the two bodies.</summary>
    private static bool Touches(Dynamics.World2 w, int a, int b)
        => w.ContactCache.Keys.Any(k =>
            (k.Item1 == a && k.Item2 == b) || (k.Item1 == b && k.Item2 == a));

    /// <summary>Z01: a pair that starts touching produces a *began* event.</summary>
    [Fact]
    public void Z01_BeganEventWhenAPairStartsTouching()
    {
        using var w = Scene();
        w.Step(Dt);
        Assert.Empty(w.ContactEvents());

        bool landed = false;
        for (int i = 0; i < 240 && !landed; i++)
        {
            w.Step(Dt);
            landed = HasEvent(w.ContactEvents(), 0, 1, 0);
        }
        Assert.True(landed, "the ball must produce a began event when it lands");
    }

    /// <summary>Z02: a pair that stops touching produces an *ended* event.</summary>
    [Fact]
    public void Z02_EndedEventWhenAPairStopsTouching()
    {
        using var w = Scene();
        for (int i = 0; i < 240; i++) w.Step(Dt);
        w.SetBodyTransform(1, new Vector2(0.0f, 50.0f), 0.0f);

        bool ended = false;
        for (int i = 0; i < 60 && !ended; i++)
        {
            w.Step(Dt);
            ended = HasEvent(w.ContactEvents(), 0, 1, 1);
        }
        Assert.True(ended, "the pair must produce an ended event when it separates");
    }

    /// <summary>Z03: a pair that keeps touching produces no further events.</summary>
    [Fact]
    public void Z03_ARestingPairProducesNoRepeatEvents()
    {
        using var w = Scene();
        for (int i = 0; i < 240; i++) w.Step(Dt);
        int repeats = 0;
        for (int i = 0; i < 60; i++)
        {
            w.Step(Dt);
            repeats += w.ContactEvents().Length;
        }
        Assert.Equal(0, repeats);
    }

    /// <summary>Z04: a sensor reports its contacts and never pushes.</summary>
    [Fact]
    public void Z04_ASensorReportsWithoutPushing()
    {
        using var w = Scene();
        w.SetBodySensor(1, true);

        bool began = false;
        for (int i = 0; i < 240; i++)
        {
            w.Step(Dt);
            began |= HasEvent(w.ContactEvents(), 0, 1, 0);
        }
        Assert.True(began, "the sensor must report the contact it had");
        Assert.True(
            w.GetBodyTransform(1).Position.Y < -1.0f,
            $"a sensor must fall through the floor (y={w.GetBodyTransform(1).Position.Y})");
    }

    /// <summary>Z05: a sensor's contact ends when it leaves.</summary>
    [Fact]
    public void Z05_ASensorReportsTheContactEnding()
    {
        using var w = Scene();
        w.SetBodySensor(1, true);
        bool ended = false;
        for (int i = 0; i < 240; i++)
        {
            w.Step(Dt);
            ended |= HasEvent(w.ContactEvents(), 0, 1, 1);
        }
        Assert.True(ended, "the sensor's contact must end when the pair separates");
    }

    /// <summary>Z06: the same in 3D.</summary>
    [Fact]
    public void Z06_EventsAndSensorsIn3D()
    {
        using var w = new PhiNativeWorld3();
        w.SetGravity(new Vector3(0.0f, -9.81f, 0.0f));
        w.SetSleepThresholds(0.0f, 0.0f);
        w.SetCollectContactEvents(true);
        w.AddStaticBody(new Vector3(0.0f, -1.0f, 0.0f), new Vector3(10.0f, 0.5f, 10.0f));
        w.AddDynamicBall(new Vector3(0.0f, 2.0f, 0.0f), 0.5f, 1.0f);

        bool began = false;
        for (int i = 0; i < 240; i++)
        {
            w.Step(Dt);
            began |= Array.Exists(w.ContactEvents(),
                e => e.BodyA == 0 && e.BodyB == 1 && e.Kind == 0);
        }
        Assert.True(began, "the ball must produce a began event when it lands");

        w.SetBodySensor(1, true);
        for (int i = 0; i < 240; i++) w.Step(Dt);
        Assert.True(
            w.GetBodyTransform(1).Position.Y < -1.0f,
            $"a sensor must fall through the floor (y={w.GetBodyTransform(1).Position.Y})");
    }

    /// <summary>Z07: the events are off by default, and the scene is unchanged.</summary>
    [Fact]
    public void Z07_EventsAreOffByDefault()
    {
        using var w = new PhiNativeWorld();
        w.SetGravity(new Vector2(0.0f, -9.81f));
        w.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        w.AddDynamicBall(new Vector2(0.0f, 2.0f), 0.5f, 1.0f);
        for (int i = 0; i < 60; i++) w.Step(Dt);
        Assert.Empty(w.ContactEvents());
    }

    /// <summary>Z08: the native bridge exposes the sensor flag and the events.</summary>
    [Fact]
    public void Z08_FfiSensorFlagAndEvents()
    {
        using var w = new PhiNativeWorld();
        int id = w.AddDynamicBall(new Vector2(0.0f, 2.0f), 0.5f, 1.0f);
        w.SetBodySensor(id, true);
        Assert.Throws<ArgumentOutOfRangeException>(() => w.SetBodySensor(9999, true));
        w.Step(Dt);
    }

    /// <summary>
    /// Z09: a 3D sensor reports its contacts.
    ///
    /// The 3D sensor path used to be dead: the guard that decided
    /// whether the diff ran read a buffer that had already been emptied.
    /// </summary>
    [Fact]
    public void Z09_A3DSensorReportsItsContacts()
    {
        var w = new Dynamics.World3();
        w.Gravity = new Vector3(0.0f, -9.81f, 0.0f);
        w.LinearSleepThreshold = 0.0f;
        w.AngularSleepThreshold = 0.0f;
        w.CollectContactEvents = true;
        w.AddStaticBody(
            new Vector3(0.0f, -1.0f, 0.0f), new Dynamics.BoxShape3(new Vector3(10.0f, 0.5f, 10.0f)));
        int sensor = w.AddDynamicBody(
            new Vector3(0.0f, 2.0f, 0.0f), new Dynamics.SphereShape3(0.5f), 1.0f);
        w.Bodies[sensor].IsSensor = true;

        bool began = false;
        for (int i = 0; i < 240; i++)
        {
            w.Step(Dt);
            began |= w.ContactEvents.Any(e =>
                e.BodyA == 0 && e.BodyB == (uint)sensor && e.Kind == Dynamics.ContactEventKind.Began);
        }
        Assert.True(began, "the 3D sensor must produce a began event when it lands");
    }

    /// <summary>
    /// Z10: one <em>began</em> event per pair, even with two contact points.
    ///
    /// A box resting on a box touches at two points: it began touching
    /// once, so there is one event.
    /// </summary>
    [Fact]
    public void Z10_OneBeganEventPerPair()
    {
        var w = new Dynamics.World2();
        w.Gravity = new Vector2(0.0f, -9.81f);
        w.LinearSleepThreshold = 0.0f;
        w.AngularSleepThreshold = 0.0f;
        w.CollectContactEvents = true;
        w.AddStaticBody(
            new Vector2(0.0f, -1.0f), new Dynamics.BoxShape2(new Vector2(10.0f, 0.5f)));
        int box = w.AddDynamicBody(
            new Vector2(0.0f, -0.4f), new Dynamics.BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);

        int beganInAStep = 0;
        for (int i = 0; i < 60; i++)
        {
            w.Step(Dt);
            int began = w.ContactEvents.Count(e =>
                e.BodyA == 0 && e.BodyB == (uint)box && e.Kind == Dynamics.ContactEventKind.Began);
            beganInAStep = System.Math.Max(beganInAStep, began);
        }
        Assert.Equal(1, beganInAStep);
    }

    /// <summary>
    /// Z11: the events are the events of the last step, and of no other.
    ///
    /// The list is cleared at the start of every step: a scene that
    /// keeps producing events must not pile them up.
    /// </summary>
    [Fact]
    public void Z11_TheEventsAreOfTheLastStepOnly()
    {
        var w = new Dynamics.World2();
        w.Gravity = new Vector2(0.0f, -9.81f);
        w.LinearSleepThreshold = 0.0f;
        w.AngularSleepThreshold = 0.0f;
        w.CollectContactEvents = true;
        w.AddStaticBody(
            new Vector2(0.0f, -1.0f), new Dynamics.BoxShape2(new Vector2(10.0f, 0.5f)));
        w.AddDynamicBody(
            new Vector2(0.0f, 2.0f), new Dynamics.CircleShape(0.5f), 1.0f);

        int maxEvents = 0;
        for (int i = 0; i < 240; i++)
        {
            w.Step(Dt);
            maxEvents = System.Math.Max(maxEvents, w.ContactEvents.Count);
        }
        Assert.True(maxEvents <= 2, $"the events must not pile up (max={maxEvents})");
    }

    /// <summary>
    /// Z12: removing a body ends the contacts it had.
    ///
    /// The pair is gone, so the contact ended. The Rust core says so
    /// with an <em>ended</em> event; the managed mirror drops the
    /// contact silently (it fires no event for a removal), so the
    /// mirror asserts the contact itself is gone.
    /// </summary>
    [Fact]
    public void Z12_RemovingABodyEndsItsContacts()
    {
        var w = new Dynamics.World2();
        w.Gravity = new Vector2(0.0f, -9.81f);
        w.LinearSleepThreshold = 0.0f;
        w.AngularSleepThreshold = 0.0f;
        w.CollectContactEvents = true;
        int floor = w.AddStaticBody(
            new Vector2(0.0f, -1.0f), new Dynamics.BoxShape2(new Vector2(10.0f, 0.5f)));
        int ball = w.AddDynamicBody(
            new Vector2(0.0f, -0.4f), new Dynamics.CircleShape(0.5f), 1.0f);
        for (int i = 0; i < 60; i++)
        {
            w.Step(Dt);
        }
        Assert.Empty(w.ContactEvents);

        w.RemoveBody(ball);
        Assert.False(
            Touches(w, floor, ball),
            "removing a body must end the contacts it had");

        // And it stays gone.
        w.Step(Dt);
        Assert.False(
            Touches(w, floor, ball),
            "a removed body must not keep its contacts");
    }

    /// <summary>
    /// The managed mirror collects the same events as the native core
    /// (the managed world has its own narrowphase loop).
    /// </summary>
    [Fact]
    public void ManagedMirrorCollectsTheSameEvents()
    {
        var w = new Dynamics.World2();
        w.Gravity = new Vector2(0.0f, -9.81f);
        w.LinearSleepThreshold = 0.0f;
        w.AngularSleepThreshold = 0.0f;
        w.CollectContactEvents = true;
        w.AddStaticBody(new Vector2(0.0f, -1.0f), new Dynamics.BoxShape2(new Vector2(10.0f, 0.5f)));
        w.AddDynamicBody(new Vector2(0.0f, 2.0f), new Dynamics.CircleShape(0.5f), 1.0f);

        bool began = false;
        for (int i = 0; i < 240; i++)
        {
            w.Step(Dt);
            began |= Array.Exists(w.ContactEvents.ToArray(),
                e => e.BodyA == 0 && e.BodyB == 1 && e.Kind == Dynamics.ContactEventKind.Began);
        }
        Assert.True(began, "the managed mirror must report the began event too");
    }
}
