#include <cfenv>
#include <cmath>
#include <complex>
#include <concepts>
#include <limits>
#include <type_traits>

#include <nanobind/nanobind.h>

namespace {
using PyUFuncGenericFunction
    = void (*)(char **, const intptr_t *, const intptr_t *, void *);

constexpr double kMax = std::numeric_limits<float>::max();
constexpr double kMin = std::numeric_limits<float>::lowest();

template<typename T>
constexpr T kPlaceholder;

template<std::floating_point T>
constexpr T kPlaceholder<T> = std::numeric_limits<T>::quiet_NaN();

template<std::floating_point T>
constexpr std::complex<T> kPlaceholder<std::complex<T>> {
    kPlaceholder<T>,
    kPlaceholder<T>,
};

template<std::signed_integral T>
constexpr T kPlaceholder<T> = std::numeric_limits<T>::lowest();

template<std::unsigned_integral T>
constexpr T kPlaceholder<T> = std::numeric_limits<T>::max();

template<std::integral T>
constexpr double kLowerBound = std::numeric_limits<T>::lowest() - .5;

template<>
constexpr double kLowerBound<int64_t>
    = static_cast<double>(std::numeric_limits<int64_t>::lowest()) - 2048;

template<std::integral T>
constexpr double kUpperBound = std::numeric_limits<T>::max() + .5;

template<typename To, typename From>
To Cast(From from) noexcept;

// This is an overload rather than a specialization, but it works
template<std::integral To>
To Cast(double from) noexcept {
    if (from > kLowerBound<To> && from < kUpperBound<To>) [[likely]] {
        return std::llround(from);
    }
    std::feraiseexcept(FE_INVALID);
    return kPlaceholder<To>;
}

// This is a specialization of the overload above
template<>
uint64_t Cast(double from) noexcept {
    constexpr auto k52 = 1LLU << (std::numeric_limits<double>::digits - 1LLU);

    if (from < k52) [[likely]] {
        if (from > kLowerBound<uint64_t>) [[likely]] {
            return std::llround(from);
        }
    } else if (from < kUpperBound<uint64_t>) [[likely]] {
        return static_cast<uint64_t>(from);
    }
    std::feraiseexcept(FE_INVALID);
    return kPlaceholder<uint64_t>;
}

template<>
float Cast(double from) noexcept {
    if (kMin <= from && from <= kMax) [[likely]] {
        return static_cast<float>(from);
    }
    std::feraiseexcept(FE_INVALID);
    return kPlaceholder<float>;
}

template<>
double Cast(double from) noexcept {
    if (std::isfinite(from)) [[likely]] {
        return from;
    }
    return kPlaceholder<double>;
}

template<>
std::complex<float> Cast(double from) noexcept {
    if (kMin <= from && from <= kMax) [[likely]] {
        return { static_cast<float>(from), 0 };
    }
    std::feraiseexcept(FE_INVALID);
    return kPlaceholder<std::complex<float>>;
}

template<>
std::complex<double> Cast(double from) noexcept {
    if (std::isfinite(from)) [[likely]] {
        return { from, 0 };
    }
    return kPlaceholder<std::complex<double>>;
}

template<>
std::complex<float> Cast(std::complex<double> from) noexcept {
    if (
        kMin <= from.real() && kMin <= from.imag()
        && from.real() <= kMax && from.imag() <= kMax
    ) [[likely]] {
        return {
            static_cast<float>(from.real()),
            static_cast<float>(from.imag()),
        };
    }
    std::feraiseexcept(FE_INVALID);
    return kPlaceholder<std::complex<float>>;
}

template<>
std::complex<double> Cast(std::complex<double> from) noexcept {
    if (std::isfinite(from.real()) && std::isfinite(from.imag())) [[likely]] {
        return from;
    }
    return kPlaceholder<std::complex<double>>;
}

template<typename To, typename From>
void Round(
    char **args,
    const intptr_t *dimensions,
    const intptr_t *steps,
    [[maybe_unused]] void *data
) noexcept {
    // NOLINTBEGIN
    intptr_t n = dimensions[0];
    char *in = args[0];
    // NOLINTEND
    char *out = args[1];
    intptr_t in_step = steps[0];
    intptr_t out_step = steps[1];

    for (intptr_t i = 0; i < n; ++i) {
        auto from = reinterpret_cast<const From *>(in)[0];
        reinterpret_cast<To *>(out)[0] = Cast<To>(from);
        in += in_step;
        out += out_step;
    }
}

template<typename To, typename From>
void Scale(
    char **args,
    const intptr_t *dimensions,
    const intptr_t *steps,
    [[maybe_unused]] void *data
) noexcept {
    intptr_t n = dimensions[0];  // NOLINT(readability-identifier-length)
    char *out = args[2];
    intptr_t out_step = steps[2];
    if (steps[1]) [[unlikely]] {
        std::feraiseexcept(FE_INVALID);
        for (intptr_t i = 0; i < n; ++i) {
            reinterpret_cast<To *>(out)[0] = kPlaceholder<To>;
            out += out_step;
        }
        return;
    }
    intptr_t in_step = steps[0];

    char *base = args[0];
    double scale = reinterpret_cast<const double *>(args[1])[0];
    for (intptr_t i = 0; i < n; ++i) {
        auto from = reinterpret_cast<const From *>(base)[0];
        if constexpr (std::is_same_v<From, std::complex<float>>) {
            reinterpret_cast<To *>(out)[0] = Cast<To, std::complex<double>>({
                from.real() * scale,
                from.imag() * scale,
            });
        } else {
            reinterpret_cast<To *>(out)[0] = Cast<To>(from * scale);
        }
        base += in_step;
        out += out_step;
    }
}

constexpr PyUFuncGenericFunction round_funcs[] {
    Round<int8_t, double>,
    Round<int16_t, double>,
    Round<int32_t, double>,
    Round<int64_t, double>,
    Round<uint8_t, double>,
    Round<uint16_t, double>,
    Round<uint32_t, double>,
    Round<uint64_t, double>,
    Round<float, double>,
    Round<double, double>,

    Round<std::complex<float>, std::complex<double>>,
    Round<std::complex<double>, std::complex<double>>,
};

constexpr PyUFuncGenericFunction scale_funcs[] {
    Scale<int8_t, int8_t>,
    Scale<int8_t, int16_t>,
    Scale<int8_t, int32_t>,
    Scale<int8_t, int64_t>,
    Scale<int8_t, uint8_t>,
    Scale<int8_t, uint16_t>,
    Scale<int8_t, uint32_t>,
    Scale<int8_t, uint64_t>,
    Scale<int8_t, float>,
    Scale<int8_t, double>,

    Scale<int16_t, int8_t>,
    Scale<int16_t, int16_t>,
    Scale<int16_t, int32_t>,
    Scale<int16_t, int64_t>,
    Scale<int16_t, uint8_t>,
    Scale<int16_t, uint16_t>,
    Scale<int16_t, uint32_t>,
    Scale<int16_t, uint64_t>,
    Scale<int16_t, float>,
    Scale<int16_t, double>,

    Scale<int32_t, int8_t>,
    Scale<int32_t, int16_t>,
    Scale<int32_t, int32_t>,
    Scale<int32_t, int64_t>,
    Scale<int32_t, uint8_t>,
    Scale<int32_t, uint16_t>,
    Scale<int32_t, uint32_t>,
    Scale<int32_t, uint64_t>,
    Scale<int32_t, float>,
    Scale<int32_t, double>,

    Scale<int64_t, int8_t>,
    Scale<int64_t, int16_t>,
    Scale<int64_t, int32_t>,
    Scale<int64_t, int64_t>,
    Scale<int64_t, uint8_t>,
    Scale<int64_t, uint16_t>,
    Scale<int64_t, uint32_t>,
    Scale<int64_t, uint64_t>,
    Scale<int64_t, float>,
    Scale<int64_t, double>,

    Scale<uint8_t, int8_t>,
    Scale<uint8_t, int16_t>,
    Scale<uint8_t, int32_t>,
    Scale<uint8_t, int64_t>,
    Scale<uint8_t, uint8_t>,
    Scale<uint8_t, uint16_t>,
    Scale<uint8_t, uint32_t>,
    Scale<uint8_t, uint64_t>,
    Scale<uint8_t, float>,
    Scale<uint8_t, double>,

    Scale<uint16_t, int8_t>,
    Scale<uint16_t, int16_t>,
    Scale<uint16_t, int32_t>,
    Scale<uint16_t, int64_t>,
    Scale<uint16_t, uint8_t>,
    Scale<uint16_t, uint16_t>,
    Scale<uint16_t, uint32_t>,
    Scale<uint16_t, uint64_t>,
    Scale<uint16_t, float>,
    Scale<uint16_t, double>,

    Scale<uint32_t, int8_t>,
    Scale<uint32_t, int16_t>,
    Scale<uint32_t, int32_t>,
    Scale<uint32_t, int64_t>,
    Scale<uint32_t, uint8_t>,
    Scale<uint32_t, uint16_t>,
    Scale<uint32_t, uint32_t>,
    Scale<uint32_t, uint64_t>,
    Scale<uint32_t, float>,
    Scale<uint32_t, double>,

    Scale<uint64_t, int8_t>,
    Scale<uint64_t, int16_t>,
    Scale<uint64_t, int32_t>,
    Scale<uint64_t, int64_t>,
    Scale<uint64_t, uint8_t>,
    Scale<uint64_t, uint16_t>,
    Scale<uint64_t, uint32_t>,
    Scale<uint64_t, uint64_t>,
    Scale<uint64_t, float>,
    Scale<uint64_t, double>,

    Scale<float, int8_t>,
    Scale<float, int16_t>,
    Scale<float, int32_t>,
    Scale<float, int64_t>,
    Scale<float, uint8_t>,
    Scale<float, uint16_t>,
    Scale<float, uint32_t>,
    Scale<float, uint64_t>,
    Scale<float, float>,
    Scale<float, double>,

    Scale<double, int8_t>,
    Scale<double, int16_t>,
    Scale<double, int32_t>,
    Scale<double, int64_t>,
    Scale<double, uint8_t>,
    Scale<double, uint16_t>,
    Scale<double, uint32_t>,
    Scale<double, uint64_t>,
    Scale<double, float>,
    Scale<double, double>,

    Scale<std::complex<float>, int8_t>,
    Scale<std::complex<float>, int16_t>,
    Scale<std::complex<float>, int32_t>,
    Scale<std::complex<float>, int64_t>,
    Scale<std::complex<float>, uint8_t>,
    Scale<std::complex<float>, uint16_t>,
    Scale<std::complex<float>, uint32_t>,
    Scale<std::complex<float>, uint64_t>,
    Scale<std::complex<float>, float>,
    Scale<std::complex<float>, double>,
    Scale<std::complex<float>, std::complex<float>>,
    Scale<std::complex<float>, std::complex<double>>,

    Scale<std::complex<double>, int8_t>,
    Scale<std::complex<double>, int16_t>,
    Scale<std::complex<double>, int32_t>,
    Scale<std::complex<double>, int64_t>,
    Scale<std::complex<double>, uint8_t>,
    Scale<std::complex<double>, uint16_t>,
    Scale<std::complex<double>, uint32_t>,
    Scale<std::complex<double>, uint64_t>,
    Scale<std::complex<double>, float>,
    Scale<std::complex<double>, double>,
    Scale<std::complex<double>, std::complex<float>>,
    Scale<std::complex<double>, std::complex<double>>,
};
}  // namespace

NB_MODULE(_accel, m)  // NOLINT
{
    m.attr("round_funcs") = reinterpret_cast<uintptr_t>(round_funcs);
    m.attr("scale_funcs") = reinterpret_cast<uintptr_t>(scale_funcs);
}
