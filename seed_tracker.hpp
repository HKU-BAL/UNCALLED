#ifndef SEED_TRACKER_HPP
#define SEED_TRACKER_HPP

#include "seed_graph.hpp"
#include <set>
#include <vector>

class ReadAln {
    public:
    Range ref_st_;
    int evt_st_,
        ref_en_, evt_en_,
        total_len_;

    ReadAln(Range ref_en, int evt_en);
    ReadAln(const ReadAln &r);
    ReadAln();
    int ref_start_base() const;
    void update_next(ReadAln &new_loc) const;
    void print(bool print_all) const;

    friend bool operator< (const ReadAln &q1, const ReadAln &q2);
};

bool operator< (const ReadAln &q1, const ReadAln &q2);

class SeedTracker {

    public:


    std::set<ReadAln> locations;

    SeedTracker();
    int add_seed(Result seed);
    int add_seeds(const std::vector<Result> &seeds);

    std::vector<ReadAln> get_alignments(int min_len);

    void print(std::string &strand);
};


#endif