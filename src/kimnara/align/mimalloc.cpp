#include <cstdlib>

#include <mimalloc.h>
#include <nanobind/nanobind.h>

namespace {
// malloc() guarantees 16-byte alignment on Linux, macOS and Windows
// https://sourceware.org/glibc/manual/latest/html_node/Aligned-Memory-Blocks.html
// https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/malloc.3.html
// https://learn.microsoft.com/en-us/windows/win32/api/heapapi/nf-heapapi-heapalloc#remarks
constexpr size_t k32 = 32;
constexpr size_t k64 = 64;
constexpr size_t k128 = 128;

constexpr decltype(std::calloc) *calloc_funcs[] {
    [](size_t num, size_t size) noexcept {
        return mi_calloc_aligned(num, size, k32);
    },
    [](size_t num, size_t size) noexcept {
        return mi_calloc_aligned(num, size, k64);
    },
};

constexpr decltype(std::free) *free_funcs[] {
    mi_free,
};

constexpr decltype(std::malloc) *malloc_funcs[] {
    [](size_t size) noexcept { return mi_malloc_aligned(size, k32); },
    [](size_t size) noexcept { return mi_malloc_aligned(size, k64); },
};

constexpr decltype(std::realloc) *realloc_funcs[] {
    [](void *ptr, size_t new_size) noexcept {
        return mi_realloc_aligned(ptr, new_size, k32);
    },
    [](void *ptr, size_t new_size) noexcept {
        return mi_realloc_aligned(ptr, new_size, k64);
    },
};

// Remove the following lines causing segmentation fault at exit:
// https://github.com/inaccel/numpy-allocator/blob/v1.2.1/numpy_allocator.c#L219-L225
void Destructor(PyObject *handler) {
    if (
        void *mem_handler = PyCapsule_GetPointer(handler, "mem_handler")
    ) [[likely]] {
        // Avoid including NumPy as a build-time dependency.
        // The data structure is documented at
        // https://numpy.org/doc/stable/reference/c-api/data_memory.html#c.PyDataMem_Handler
        uintptr_t allocator = reinterpret_cast<uintptr_t>(mem_handler) + k128;
        // NOLINTBEGIN
        std::free(*reinterpret_cast<void **>(allocator));
        std::free(mem_handler);
        // NOLINTEND
        return;
    }
    PyErr_WriteUnraisable(PyErr_Occurred());
}
}  // namespace

NB_MODULE(_mimalloc, m)  // NOLINT
{
    m.attr("calloc_funcs") = reinterpret_cast<uintptr_t>(calloc_funcs);
    m.attr("free_funcs") = reinterpret_cast<uintptr_t>(free_funcs);
    m.attr("malloc_funcs") = reinterpret_cast<uintptr_t>(malloc_funcs);
    m.attr("realloc_funcs") = reinterpret_cast<uintptr_t>(realloc_funcs);
    m.def("set_destructor", [](const nanobind::handle &capsule) {
        if (PyCapsule_SetDestructor(capsule.ptr(), Destructor)) [[unlikely]] {
            throw nanobind::python_error();
        }
    });
}
