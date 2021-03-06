#include <cmath>
#include "normalizer.hpp"

Normalizer::Normalizer() 
    : signal_(6000), //TODO need to set
      mean_(0),
      varsum_(0),
      n_(0),
      rd_(0),
      wr_(0),
      is_full_(false),
      is_empty_(true) {
}

Normalizer::Normalizer(float tgt_mean, float tgt_stdv) :
    tgt_mean_(tgt_mean),
    tgt_stdv_(tgt_stdv) {}

void Normalizer::set_target(float tgt_mean, float tgt_stdv) {
    tgt_mean_ = tgt_mean;
    tgt_stdv_ = tgt_stdv; 
}

void Normalizer::set_signal(const std::vector<float> &signal) {
    signal_ = signal;
    n_ = signal_.size();
    rd_ = wr_ = 0;
    is_full_ = true;
    is_empty_ = false;

    mean_ = 0;
    for (float e : signal_) mean_ += e;
    mean_ /= n_;

    varsum_ = 0;
    for (auto e : signal_) varsum_ += pow(e - mean_, 2);
}

bool Normalizer::push(float newevt) {
    if (is_full_) {
        return false;
    }

    double oldevt = signal_[wr_];
    signal_[wr_] = newevt;

    //Based on https://stackoverflow.com/questions/5147378/rolling-variance-algorithm
    if (n_ == signal_.size()) {
        double oldmean = mean_;
        mean_ += (newevt - oldevt) / signal_.size();
        varsum_ += (newevt + oldevt - oldmean - mean_) * (newevt - oldevt);

    //Based on https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
    } else {
        n_++;
        double dt1 = newevt - mean_;
        mean_ += dt1 / n_;
        double dt2 = newevt - mean_;
        varsum_ += dt1*dt2;
    }

    wr_ = (wr_ + 1) % signal_.size();

    is_empty_ = false;
    is_full_ = wr_ == rd_;

    return true;
}

void Normalizer::reset(u32 buffer_size) {
    n_ = 0;
    rd_ = 0;
    wr_ = 0;
    mean_ = varsum_ = 0;
    is_full_ = false;
    is_empty_ = true;

    if (buffer_size != 0 && buffer_size != signal_.size()) {
        signal_.resize(buffer_size);
    }

    signal_[0] = 0;
}

float Normalizer::get_scale() const {
    return tgt_stdv_ / sqrt(varsum_ / n_);
}

float Normalizer::get_shift(float scale) const {
    if (scale == 0) scale = get_scale();
    return tgt_mean_ - scale * mean_;
}

float Normalizer::at(u32 i) const {
    float scale = tgt_stdv_ / sqrt(varsum_ / n_);
    float shift = tgt_mean_ - scale * mean_;
    return scale * signal_[i] + shift;
}

float Normalizer::pop() {
    float e = at(rd_);

    rd_ = (rd_+1) % signal_.size();
    is_empty_ = rd_ == wr_;
    is_full_ = false;

    return e;
}

u32 Normalizer::unread_size() const {
    if (rd_ < wr_) return wr_ - rd_;
    else return (n_ - rd_) + wr_;
}

u32 Normalizer::skip_unread(u32 nkeep) {
    if (nkeep >= unread_size()) return 0;

    is_full_ = false;
    is_empty_ = nkeep == 0;

    u32 new_rd;
    if (nkeep <= wr_) new_rd = wr_ - nkeep;
    else new_rd = n_ - (nkeep - wr_);

    u32 nskip;
    if (new_rd > rd_) nskip = new_rd - rd_;
    else nskip = (n_ - rd_) + new_rd;

    rd_ = new_rd;
    return nskip;
}

bool Normalizer::empty() const {
    return is_empty_;
}

bool Normalizer::full() const {
    return is_full_;
}
