#include <algorithm>
#include <functional>
#include <new>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>

#define TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION 1
#include <oneapi/tbb.h>

namespace {
#if defined __cpp_lib_hardware_interference_size && __cpp_lib_hardware_interference_size >= 201703L
using std::hardware_destructive_interference_size;
#else
inline constexpr size_t hardware_destructive_interference_size = 64;
#endif  // defined __cpp_lib_hardware_interference_size && __cpp_lib_hardware_interference_size >= 201703L

class alignas(hardware_destructive_interference_size) TBBScratchSpace {
  public:
    TBBScratchSpace(
        const size_t * __restrict dimensions,
        size_t arg_len,
        size_t array_count
    ) : count_space_ {
        reinterpret_cast<size_t *>(
            tbb::detail::r1::cache_aligned_allocate(
                ~(hardware_destructive_interference_size - 1) & (
                    hardware_destructive_interference_size - 1
                    + ((arg_len+array_count) * sizeof(size_t))
                )
            )
        )
    }, array_arg_space_ {
        reinterpret_cast<char **>(count_space_) + arg_len
    } {
        std::copy_n(dimensions, arg_len, count_space_);
    }

    NB_NONCOPYABLE(TBBScratchSpace);
    TBBScratchSpace(TBBScratchSpace &&) = default;
    TBBScratchSpace &operator=(TBBScratchSpace &&) = default;

    ~TBBScratchSpace() {
        tbb::detail::r1::cache_aligned_deallocate(count_space_);
    }

    size_t *count_space() { return count_space_; }
    char **array_arg_space() { return array_arg_space_; }

  private:
    size_t *count_space_;
    char **array_arg_space_;
};

const auto &TBBConstraints() {
    static const auto constraints = [] {
        auto core_types = tbb::info::core_types();
        tbb::task_arena::constraints constraint {
            .numa_id = tbb::task_arena::automatic,
            .max_concurrency = tbb::task_arena::automatic,
            .core_type = core_types.back(),
            .max_threads_per_core = 1,
        };
        constraint.max_concurrency = tbb::info::default_concurrency(constraint);
        return constraint;
    }();
    return constraints;
}

int TBBGetNumThreads() {
    return TBBConstraints().max_concurrency;
}

void TBBNumPyVectorize(const std::function<void(intptr_t)> &func, intptr_t n) {
    tbb::task_arena limited {
        TBBConstraints(),
        /*reserved_for_masters=*/1,
        tbb::task_arena::priority::normal,
    };
    limited.execute([&, n] {
        using range_t = tbb::blocked_range<intptr_t>;
        tbb::parallel_for(
            range_t(0, n, 1),
            [&](const range_t &range) {
                /*
                There is no such optimization like the OpenMP implementation
                next to this file.
                Maybe https://github.com/uxlfoundation/oneTBB/pull/995 can help.
                */
                nanobind::gil_scoped_acquire gil;
                for (intptr_t i = range.begin(); i < range.end(); i++) {
                    func(i);
                }
            }
        );
    });
}

// A modified version of numba/np/ufunc/tbbpool.cpp:parallel_for
void TBBParallelFor(
    void (*func)(char **, size_t *, size_t *, void *),
    char **args,
    size_t *dimensions,
    size_t *steps,
    void *data,
    size_t inner_ndim,
    size_t array_count,
    [[maybe_unused]] int num_threads
) {
    tbb::enumerable_thread_specific<TBBScratchSpace> ets {
        dimensions, inner_ndim+1, array_count,
    };
    tbb::task_arena limited {
        TBBConstraints(),
        /*reserved_for_masters=*/1,
        tbb::task_arena::priority::normal,
    };
    limited.execute([=, &ets] {
        using range_t = tbb::blocked_range<size_t>;
        tbb::parallel_for(
            range_t(0, dimensions[0], 1),
            [=, &ets](const range_t &range) {
                auto &space = ets.local();
                size_t *count_space = space.count_space();
                char **array_arg_space = space.array_arg_space();
                count_space[0] = range.size();
                for (size_t j = 0; j < array_count; j++) {
                    char *base = args[j];
                    size_t step = steps[j];
                    size_t offset = step * range.begin();
                    array_arg_space[j] = base + offset;
                }
                func(array_arg_space, count_space, steps, data);
            }
        );
    });
}
}  // namespace

// NOLINTBEGIN
NB_MODULE(_tbb, m)
// NOLINTEND
{
    m.attr("get_num_threads") = reinterpret_cast<uintptr_t>(TBBGetNumThreads);
    m.attr("parallel_for") = reinterpret_cast<uintptr_t>(TBBParallelFor);
    m.def(
        "np_vectorize",
        TBBNumPyVectorize,
        nanobind::call_guard<nanobind::gil_scoped_release>()
    );
}
