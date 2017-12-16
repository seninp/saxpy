#include <saxpy.hpp>
#include <iostream>
#include <numeric>

std::vector<double> znorm(const std::vector<double> ts, double threshold) {


    double sum = std::accumulate(std::begin(ts), std::end(ts), 0.0);
    double mean =  sum / ts.size();
    // std::cout << " mean2 " << mean << "\n";

    std::vector<double> diff(ts.size());
    std::transform(ts.begin(), ts.end(), diff.begin(), std::bind2nd(std::minus<double>(), mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / (ts.size()-1));
    // Rcout << " stdev2 " << stdev << "\n";

    if (stdev < threshold){
      std::vector<double> res(ts);
      return res;
    }

    std::vector<double> res(ts.size());
    for(int i=0; i<ts.size(); i++){
      res[i] = (ts[i]-mean)/stdev;
    }
    return res;

}
