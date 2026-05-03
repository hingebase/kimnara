#define _CRT_SECURE_NO_WARNINGS

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <omp.h>
#if defined __linux || defined __linux__
#  include <strings.h>
#endif  // defined __linux || defined __linux__

#if (defined __linux || defined __linux__) && defined __x86_64
#  include <cpuid.h>
#  define KIMNARA_LINUX64 1
#else
#  define KIMNARA_LINUX64 0
#endif  // (defined __linux || defined __linux__) && defined __x86_64

#if defined _WIN32 && (defined _M_AMD64 || defined __x86_64)
#  define strncasecmp _strnicmp
#  define KIMNARA_WIN64 1
#else
#  define KIMNARA_WIN64 0
#endif  // defined _WIN32 && (defined _M_AMD64 || defined __x86_64)

#if KIMNARA_LINUX64 || KIMNARA_WIN64
#  define MAYBE_CONSTEXPR_1
#else
#  define MAYBE_CONSTEXPR_1 constexpr
#endif  // KIMNARA_LINUX64 || KIMNARA_WIN64

#if defined __APPLE__ && defined __MACH__
#  define MAYBE_CONSTEXPR_2 constexpr
#else
#  define MAYBE_CONSTEXPR_2
#endif  // defined __APPLE__ && defined __MACH__

#define KMP_DEFAULTS(verbose, hybrid) ( \
    "OMP_DYNAMIC=false" \
    "|OMP_MAX_ACTIVE_LEVELS=1" \
    "|KMP_AFFINITY=" verbose "none" \
    "|KMP_DETERMINISTIC_REDUCTION=false" \
    "|KMP_HW_SUBSET=" hybrid "1t" \
    "|KMP_TOPOLOGY_METHOD=all" \
)

namespace {
MAYBE_CONSTEXPR_1 bool OpenMPHybridCores() {
    // Reference implementation:
    // - llvm-project/openmp/runtime/src/kmp_platform.h
    // - llvm-project/openmp/runtime/src/kmp_utility.cpp:__kmp_query_cpuid
    constexpr unsigned kInitialEAX = 7;
    constexpr unsigned kHybridBit = 0x8000;
#if KIMNARA_LINUX64
    unsigned eax {};
    unsigned ebx {};
    unsigned ecx {};
    unsigned edx {};
    __cpuid(0, eax, ebx, ecx, edx);
    if (eax < kInitialEAX) {
        return false;
    }
    __cpuid(kInitialEAX, eax, ebx, ecx, edx);
    return static_cast<bool>(edx & kHybridBit);
#elif KIMNARA_WIN64
    unsigned cpui[4];
    __cpuid(reinterpret_cast<int *>(cpui), 0);
    if (cpui[0] < kInitialEAX) {
        return false;
    }
    __cpuid(reinterpret_cast<int *>(cpui), kInitialEAX);
    return static_cast<bool>(cpui[3] & kHybridBit);
// https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
#elif defined __APPLE__ && defined __MACH__ && defined __arm64__
    return true;
#else  // linux-aarch64 or osx-64
    return false;
#endif
}

void OpenMPLaunchThreads([[maybe_unused]] int count) {
}

void OpenMPNumPyVectorize(
    const std::function<void(intptr_t)> &func,
    intptr_t n,
    nanobind::list &excs
) {
    #pragma omp parallel default(none) shared(func, n, excs)
    {
        // https://github.com/cython/cython/pull/6562#issuecomment-2925653931
        nanobind::gil_scoped_acquire gil;
        auto nogil = std::make_optional<nanobind::gil_scoped_release>();
        #pragma omp for nowait schedule(guided)
        for (intptr_t i = 0; i < n; i++)
        {
            nogil.reset();
            try {
                func(i);
            } catch (const nanobind::python_error &e) {
                excs.append(e.value());
            }
        }
    }
}

// A modified version of numba/np/ufunc/omppool.cpp:parallel_for
void OpenMPParallelFor(
    void (*func)(char **, size_t *, size_t *, void *),
    char **args,
    size_t * __restrict dimensions,
    size_t *steps,
    void *data,
    size_t inner_ndim,
    size_t array_count,
    [[maybe_unused]] int num_threads
) {
    const size_t arg_len = inner_ndim + 1;
    const auto size = static_cast<ptrdiff_t>(dimensions[0]);
    #pragma omp parallel default(none) shared(func, args, dimensions, steps, data, array_count, arg_len, size)
    {
        auto *count_space = reinterpret_cast<size_t *>(
            kmp_malloc((arg_len+array_count) * sizeof(size_t))
        );
        auto *array_arg_space = reinterpret_cast<char **>(count_space) + arg_len;
        std::copy_n(dimensions, arg_len, count_space);
        count_space[0] = 1;
        #pragma omp for nowait schedule(guided)
        for (ptrdiff_t i = 0; i < size; i++) {
            for (size_t j = 0; j < array_count; j++) {
                char *base = args[j];
                size_t step = steps[j];
                size_t offset = step * i;
                array_arg_space[j] = base + offset;
            }
            func(array_arg_space, count_space, steps, data);
        }
        kmp_free(count_space);
    }
}

MAYBE_CONSTEXPR_2 bool OpenMPVerbose() {
// https://github.com/llvm/llvm-project/issues/62554#issuecomment-1708642843
#if !(defined __APPLE__ && defined __MACH__)
    if (const char *first = std::getenv("KMP_AFFINITY")) {
        first += std::strspn(first, "\t ,");
        if (strncasecmp(first, "verbose", sizeof("verbose") - 1) == 0) {
            return true;
        }
        while (const char *last = std::strchr(first, ',')) {
            if (strncasecmp(first, "noverbose", sizeof("noverbose") - 1) == 0) {
                return false;
            }
            first = last + std::strspn(last, "\t ,");
            if (strncasecmp(first, "verbose", sizeof("verbose") - 1) == 0) {
                return true;
            }
        }
    }
#endif  // !(defined __APPLE__ && defined __MACH__)
    return false;
}
}  // namespace

// NOLINTBEGIN
NB_MODULE(_openmp, m)
// NOLINTEND
{
    if MAYBE_CONSTEXPR_2 (OpenMPVerbose()) {
        if MAYBE_CONSTEXPR_1 (OpenMPHybridCores()) {
            // Higher priority than environment variables
            kmp_set_defaults(
                KMP_DEFAULTS(
                    "verbose,",
                    // Assume there are exactly 2 efficiency levels, which is
                    // the truth for most processors
                    // Some recent Intel processors have 3 types of cores:
                    // P-cores, E-cores and LP E-cores. It can be confirmed
                    // that the E-cores and LP E-cores share the same efficiency
                    // level.
                    "c:eff1,"
                )
            );
        } else {
            kmp_set_defaults(KMP_DEFAULTS("verbose,", ""));
        }
    } else if MAYBE_CONSTEXPR_1 (OpenMPHybridCores()) {
        kmp_set_defaults(KMP_DEFAULTS("", "c:eff1,"));
    } else {
        kmp_set_defaults(KMP_DEFAULTS("", ""));
    }

    m.attr("get_num_threads") = reinterpret_cast<uintptr_t>(omp_get_max_threads);
    m.attr("get_thread_id") = reinterpret_cast<uintptr_t>(omp_get_thread_num);
    m.attr("launch_threads") = reinterpret_cast<uintptr_t>(OpenMPLaunchThreads);
    m.attr("parallel_for") = reinterpret_cast<uintptr_t>(OpenMPParallelFor);
    m.def(
        "np_vectorize",
        OpenMPNumPyVectorize,
        nanobind::call_guard<nanobind::gil_scoped_release>()
    );
}
