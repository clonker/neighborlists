#pragma once

#include <atomic>

namespace neighborlists {
template<typename T>
class CopyableAtomic {
    std::atomic<T> _a;
public:
    /**
     * the contained value
     */
    using value_type = T;

    /**
     * Creates a new copyable atomic of the specified type
     */
    CopyableAtomic() : _a() {}

    /**
     * Instantiate this by an atomic. Not an atomic operation.
     * @param a the other atomic
     */
    explicit CopyableAtomic(const std::atomic<T> &a) : _a(a.load()) {}

    /**
     * Copy constructor. Not an atomic CAS operation.
     * @param other the other copyable atomic
     */
    CopyableAtomic(const CopyableAtomic &other) : _a(other._a.load()) {}

    /**
     * Copy assign. Not an atomic CAS operation.
     * @param other the other copyable atomic
     * @return this
     */
    CopyableAtomic &operator=(const CopyableAtomic &other) {
        _a.store(other._a.load());
        return *this;
    }

    /**
     * Const dereference operator, yielding the underlying std atomic.
     * @return the underlying std atomic
     */
    const std::atomic<T> &operator*() const {
        return _a;
    }

    /**
     * Nonconst dereference operator, yielding the underlying std atomic.
     * @return the underlying std atomic
     */
    std::atomic<T> &operator*() {
        return _a;
    }

    /**
     * Const pointer-dereference operator, yielding a pointer to the underlying std atomic.
     * @return a pointer to the underlying std_atomic
     */
    const std::atomic<T> *operator->() const {
        return &_a;
    }

    /**
     * Nonconst pointer-dereference operator, yielding a pointer to the underlying std atomic.
     * @return a pointer to the underlying std_atomic
     */
    std::atomic<T> *operator->() {
        return &_a;
    }
};
}
