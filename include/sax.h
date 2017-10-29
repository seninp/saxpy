const char LETTERS[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
   'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',  'v', 'w', 'x', 'y', 'z'};

// SAX stack
//
std::vector<double> znorm(std::vector<double> *ts, double threshold);
//
std::vector<double> subseries(std::vector<double> *ts, int start, int end);
//
std::vector<double> alphabet_to_cuts(int a_size);
