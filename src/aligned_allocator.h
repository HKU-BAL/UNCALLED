/**
 Aligned memory allocator, for use with std::vector and AVX (and such).
 Use it to allocate a std::vector for AVX floats like this:
	std::vector<float, aligned_allocator<float,32> >   myVec;

 Allocator part taken directly from "aligned_allocator", by the Visual C++ team:
    http://blogs.msdn.com/b/vcblog/archive/2008/08/28/the-aligned_allocator.aspx
 Dr. Orion Lawlor, lawlor@alaska.edu, 2018-03-20 (Public Domain)
*/
#pragma once
#ifndef ALIGNED_ALLOCATOR_H
#define ALIGNED_ALLOCATOR_H

// The following headers are required for all allocators.
#include <stddef.h>  // Required for size_t and ptrdiff_t and NULL
#include <new>       // Required for placement new and std::bad_alloc
#include <stdexcept> // Required for std::length_error

// The following headers contain stuff that aligned_allocator uses.
#include <malloc.h>  // For _mm_malloc() and _mm_free()
#include <mm_malloc.h>

template <typename T,int alignment> class aligned_allocator {
public:

    // The following will be the same for virtually all allocators.
    typedef T * pointer;
    typedef const T * const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    T * address(T& r) const {
        return &r;
    }

    const T * address(const T& s) const {
        return &s;
    }

    size_t max_size() const {
        // The following has been carefully written to be independent of
        // the definition of size_t and to avoid signed/unsigned warnings.
        return (static_cast<size_t>(0) - static_cast<size_t>(1)) / sizeof(T);
    }


    // The following must be the same for all allocators.
    template <typename U> struct rebind {
        typedef aligned_allocator<U,alignment> other;
    };

    bool operator!=(const aligned_allocator& other) const {
        return !(*this == other);
    }

    void construct(T * const p, const T& t) const {
        void * const pv = static_cast<void *>(p);

        new (pv) T(t);
    }

    void destroy(T * const p) const {
        p->~T();
    }


    // Returns true if and only if storage allocated from *this
    // can be deallocated from other, and vice versa.
    // Always returns true for stateless allocators.
    bool operator==(const aligned_allocator& other) const {
        return true;
    }


    // Default constructor, copy constructor, rebinding constructor, and destructor.
    // Empty for stateless allocators.
    aligned_allocator() { }

    aligned_allocator(const aligned_allocator&) { }

    template <typename U> aligned_allocator(const aligned_allocator<U,alignment>&) { }

    ~aligned_allocator() { }

    // The following will be the same for all allocators that ignore hints.
    template <typename U> T * allocate(const size_t n, const U * /* const hint */) const {
        return allocate(n);
    }

    // The following will be different for each allocator.
    T * allocate(const size_t n) const {
        // The return value of allocate(0) is unspecified.
        // aligned_allocator returns NULL in order to avoid depending
        // on malloc(0)'s implementation-defined behavior
        // (the implementation can define malloc(0) to return NULL,
        // in which case the bad_alloc check below would fire).
        // All allocators can return NULL in this case.
        if (n == 0) {
            return NULL;
        }

        // All allocators should contain an integer overflow check.
        // The Standardization Committee recommends that std::length_error
        // be thrown in the case of integer overflow.
        if (n > max_size()) {
            throw std::length_error("aligned_allocator<T>::allocate() - Integer overflow.");
        }

        // aligned_allocator wraps _mm_malloc().
        void * const pv = _mm_malloc(n * sizeof(T),alignment);

        // Allocators should throw std::bad_alloc in the case of memory allocation failure.
        if (pv == NULL) {
            throw std::bad_alloc();
        }

        return static_cast<T *>(pv);
    }

    void deallocate(T * const p, const size_t n) const {
        // aligned_allocator wraps free().
        _mm_free(p);
    }

    // Allocators are not required to be assignable, so
    // all allocators should have a private unimplemented
    // assignment operator.
private:
    aligned_allocator& operator=(const aligned_allocator&);
};

#endif