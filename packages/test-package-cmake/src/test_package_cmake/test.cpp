#include <kimnara.h>

extern "C" {
Py_EXPORTED_SYMBOL void TestLoggingCAPI(void *logger) {
    KimnaraTrace(logger, "Trace message", 0);
    KimnaraDebug(logger, "Debug message", sizeof("Debug message") - 1);
    KimnaraInfo(logger, "Info message", sizeof("Info message") - 2);
    KimnaraWarning(logger, "Warning message", 0);
    KimnaraError(logger, "Error message", sizeof("Error message") - 1);
    KimnaraCritical(logger, "Critical message", sizeof("Critical message") - 2);
}
}
