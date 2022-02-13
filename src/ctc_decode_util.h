#ifndef CTC_DECODER_UTIL_H
#define CTC_DECODER_UTIL_H
#include <string>
#include <vector>
#include <utility>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <valarray>
#include <algorithm>

const float NUM_FLT_INF = std::numeric_limits<float>::max();
const float NUM_FLT_MIN = std::numeric_limits<float>::min();

// inline function for validation check
inline void check(
    bool x, const char *expr, const char *file, int line, const char *err) {
  if (!x) {
    std::cout << "[" << file << ":" << line << "] ";
    std::string err_message = "\"" + std::string(expr) + "\" check filed. " + std::string(err);
    throw std::runtime_error(err_message);
  }
}

#define VALID_CHECK(x, info) \
  check(static_cast<bool>(x), #x, __FILE__, __LINE__, info)
#define VALID_CHECK_EQ(x, y, info) VALID_CHECK((x) == (y), info)
#define VALID_CHECK_GT(x, y, info) VALID_CHECK((x) > (y), info)
#define VALID_CHECK_LT(x, y, info) VALID_CHECK((x) < (y), info)


// Function template for comparing two pairs
template <typename T1, typename T2>
bool pair_comp_first_rev(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) {
  return a.first > b.first;
}

// Function template for comparing two pairs
template <typename T1, typename T2>
bool pair_comp_second_rev(const std::pair<T1, T2> &a,
                          const std::pair<T1, T2> &b) {
  return a.second > b.second;
}

// Return the sum of two probabilities in log scale
template <typename T>
T log_sum_exp(const T &x, const T &y) {
  static T num_min = -std::numeric_limits<T>::max();
  if (x <= num_min) return y;
  if (y <= num_min) return x;
  T xmax = std::max(x, y);
  return std::log(std::exp(x - xmax) + std::exp(y - xmax)) + xmax;
}

#endif //CTC_DECODER_UTIL_H