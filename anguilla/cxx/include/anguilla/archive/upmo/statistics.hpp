#pragma once

#ifndef ANGUILLA_UPMO_STATISTICS_HPP
#define ANGUILLA_UPMO_STATISTICS_HPP

#include <cinttypes>

namespace anguilla {
namespace upmo {

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

} // namespace upmo
} // namespace anguilla

#endif // ANGUILLA_UPMO_STATISTICS_HPP
