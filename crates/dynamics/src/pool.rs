//! Object pooling for zero-allocation physics simulation.
//!
//! Provides reusable pools for frequently allocated types to eliminate
//' heap allocations during the simulation step after warmup.

use std::cell::RefCell;
use std::rc::Rc;

/// A generic object pool for reusing allocations.
///
/// Uses `RefCell` for interior mutability, allowing the pool to be shared
/// via `Rc<Pool<T>>` without requiring mutable references.
pub struct Pool<T> {
    items: RefCell<Vec<T>>,
    factory: Box<dyn Fn() -> T>,
}

impl<T> Pool<T> {
    /// Create a new pool with the given factory function.
    pub fn new(factory: impl Fn() -> T + 'static) -> Self {
        Self {
            items: RefCell::new(Vec::new()),
            factory: Box::new(factory),
        }
    }

    /// Create a new pool with pre-allocated capacity.
    pub fn with_capacity(capacity: usize, factory: impl Fn() -> T + 'static) -> Self {
        let mut items = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            items.push(factory());
        }
        Self {
            items: RefCell::new(items),
            factory: Box::new(factory),
        }
    }

    /// Get an object from the pool, creating a new one if empty.
    pub fn get(&self) -> T {
        self.items
            .borrow_mut()
            .pop()
            .unwrap_or_else(|| (self.factory)())
    }

    /// Return an object to the pool.
    pub fn put(&self, item: T) {
        self.items.borrow_mut().push(item);
    }

    /// Get the number of objects currently in the pool.
    pub fn len(&self) -> usize {
        self.items.borrow().len()
    }

    /// Check if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.items.borrow().is_empty()
    }

    /// Clear the pool, dropping all items.
    pub fn clear(&self) {
        self.items.borrow_mut().clear();
    }

    /// Shrink the pool's capacity to fit its length.
    pub fn shrink_to_fit(&self) {
        self.items.borrow_mut().shrink_to_fit();
    }
}

/// A pooled object that automatically returns to the pool when dropped.
///
/// Use `Pool::get_pooled()` to get a `Pooled<T>` that automatically
/// returns to the pool on drop.
pub struct Pooled<T> {
    item: Option<T>,
    pool: Rc<Pool<T>>,
}

impl<T> Pooled<T> {
    pub fn new(item: T, pool: Rc<Pool<T>>) -> Self {
        Self {
            item: Some(item),
            pool,
        }
    }

    pub fn get(&self) -> &T {
        self.item.as_ref().expect("Pooled item already taken")
    }

    pub fn get_mut(&mut self) -> &mut T {
        self.item.as_mut().expect("Pooled item already taken")
    }

    /// Take the item out of the pooled wrapper, preventing return to pool.
    pub fn take(mut self) -> T {
        self.item.take().expect("Pooled item already taken")
    }
}

impl<T> Drop for Pooled<T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            self.pool.put(item);
        }
    }
}

impl<T> std::ops::Deref for Pooled<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T> std::ops::DerefMut for Pooled<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

impl<T> Pool<T> {
    /// Get a pooled object that automatically returns to the pool on drop.
    pub fn get_pooled(self: &Rc<Self>) -> Pooled<T> {
        let item = self.get();
        Pooled::new(item, Rc::clone(self))
    }
}

/// Macro for creating a pool with a default factory.
#[macro_export]
macro_rules! pool {
    ($ty:ty, $factory:expr) => {
        std::rc::Rc::new($crate::pool::Pool::new($factory))
    };
    ($ty:ty, $factory:expr, $capacity:expr) => {
        std::rc::Rc::new($crate::pool::Pool::with_capacity($capacity, $factory))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_basic() {
        let pool = Pool::new(|| vec![0u32; 4]);
        let mut v = pool.get();
        v.push(42);
        pool.put(v);

        let v2 = pool.get();
        assert_eq!(v2.len(), 5);
        assert_eq!(v2[4], 42);
    }

    #[test]
    fn pooled_auto_return() {
        let pool = Rc::new(Pool::new(|| String::from("hello")));
        {
            let pooled = pool.get_pooled();
            assert_eq!(*pooled, "hello");
        } // dropped here, returns to pool

        let pooled2 = pool.get_pooled();
        assert_eq!(*pooled2, "hello");
    }

    #[test]
    fn pool_capacity() {
        let pool = Pool::with_capacity(10, || 0u32);
        assert_eq!(pool.len(), 10);

        for _ in 0..5 {
            pool.get();
        }
        assert_eq!(pool.len(), 5);

        for _ in 0..5 {
            pool.put(0);
        }
        assert_eq!(pool.len(), 10);
    }
}
