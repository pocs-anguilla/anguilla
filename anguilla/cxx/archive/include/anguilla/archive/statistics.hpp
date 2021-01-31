#pragma once

#ifndef ANGUILLA_ARCHIVE_STATISTICS_HPP
#define ANGUILLA_ARCHIVE_STATISTICS_HPP

#include <cinttypes>

namespace archive {

struct Statistics {
   public:
    Statistics() : inserts(0.0), insertAttempts(0.0) {}

    double getInsertSucessRatio() const {
        if (insertAttempts > 0.0) {
            return inserts / insertAttempts;
        }
        return 0.0;
    }

    double inserts;
    double insertAttempts;
};

}  // namespace archive

#endif  // ANGUILLA_ARCHIVE_STATISTICS_HPP
