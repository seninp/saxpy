#include <algorithm>
#include <numeric>
#include <vector>

std::vector<double> subseries(std::vector<double> *ts, int start, int end) {
  std::vector<double>::const_iterator first = ts->begin() + start;
  std::vector<double>::const_iterator last = ts->begin() + end;
  std::vector<double> res(first, last);
  return res;
}
