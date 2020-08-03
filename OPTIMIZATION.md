Editted 3st Aug, 2020

## Note

A `CMake` build script that builds off-line mapping code (`uncalled_map.cpp`) was written and is used throughout this optimization work for IDE, testing mapping speed, etc.. The portion of `BWA` library that is used by `UNCALLED`, with modifications applied, is incorporated into `src` folder. The real building script has not been updated to the changes made.

## Optimizations implemented

#### Vectorized k-mer probability calculation

The calculation of the probability that current event can match to each k-mer:

```c++
for (u16 kmer = 0; kmer < kmer_probs_.size(); kmer++) {
    kmer_probs_[kmer] = (-pow(event - lv_means_[kmer], 2) / lv_vars_x2_[kmer]) - lognorm_denoms_[kmer];
}
```
 
 is easily vectorizable and this optimization results in around 1-2% overall performance gain.

```c++
void calc_event_match_prob(const float &event, std::vector<float, aligned_allocator<float, 32>> &result) const {
    __m256 vec_event = _mm256_broadcast_ss(&event);
    static const __m256 vec_zero = _mm256_set1_ps(0.0f);
    for (u16 kmer = 0; kmer < kmer_count_; kmer += 8) {
        __m256 vec_means = _mm256_load_ps(&lv_means_[kmer]);
        __m256 vec_vars = _mm256_load_ps(&lv_vars_x2_[kmer]);
        __m256 vec_lognorms = _mm256_load_ps(&lognorm_denoms_[kmer]);
        __m256 vec_value = _mm256_sub_ps(vec_event, vec_means);
        vec_value = _mm256_mul_ps(vec_value, vec_value);
        vec_value = _mm256_div_ps(vec_value, vec_vars);
        vec_value = _mm256_add_ps(vec_value, vec_lognorms);
        _mm256_store_ps(&result[kmer], _mm256_sub_ps(vec_zero, vec_value));
    }
}
```

#### Use `POPCNT` instruction to speed up BWT `Occ` calculation

The following bit twiddling code in BWA library is for counting the number of ones in the binary representation of variable `y`:

```c
    y = (y & 0x3333333333333333ull) + (y >> 2 & 0x3333333333333333ull);
    return ((y + (y >> 4)) & 0xf0f0f0f0f0f0f0full) * 0x101010101010101ull >> 56;
```

Modern x86 and ARM processors come up with a _population count_ instruction that performs the exact function within one instruction rather than a sequence of bitwise operations. The above code can be replaced by GCC intrinsic

```c
    return __builtin_popcountll(y);
```

Target architecture needs to be stated explicitly when compiling (eg. `-march=broadwell` compiler flag) for the compiler to emit the `POPCNT` instruction, or the compiler will, by default, generate assembly code that does the work in the old way so as to maintain compatibility on old processors.

#### Optimized Path Buffer comparison

The comparison between path buffers (`Mapper::PathBuffer::operator<()`) is one of the hot spots of the program. It is called very frequently upon sorting path buffers when a new event is being mapped. The original code was written in the way that it first compares the FM-index ranges (calls `Range::operator<()`) as primary key, and, when they are equal, seed probabilities are compared as secondary key:

```c++
bool operator< (const Mapper::PathBuffer &p1, const Mapper::PathBuffer &p2) {
     return p1.fm_range_ < p2.fm_range_ ||
            (p1.fm_range_ == p2.fm_range_ && 
             p1.seed_prob_ < p2.seed_prob_);
}
```

`Range` comparator compares `start_` component as primary key and `end_` component as secondary key:

```c++
bool operator< (const Range &q1, const Range &q2) {
    return q1.start_ < q2.start_ || (q1.start_ == q2.start_ && q1.end_ < q2.end_);
}
bool operator== (const Range &q1, const Range &q2) {
    return q1.start_ == q2.start_ && q1.end_ == q2.end_;
}
```

where `start_` and `end_` are two 64-bit unsigned integers. Inspecting the disassembly suggests that the compiler generates a long sequence of comparison, logic and branching instructions for `Range` comparators. The first attempt is to rewrite `Range` comparators in a branchless manner. The above comparison is equivalent to comparing 128-bit integers where `start_` is regarded as the upper part and `end_` be the lower part. Therefore `Range`s can be loaded into 128-bit SSE registers and be compared with SSE comparisons. It is found that both `Range::operator<()` and `Range::operator==()` gain up to 30% speed up when benchmarking these two comparison functions alone, but when integrated into the entire program, there is no observable performance improvement.

The second attempt is to incorporate comparisons of `Range` back into `PathBuffer` comparisons directly:

```c++
bool operator< (const Mapper::PathBuffer &p1, const Mapper::PathBuffer &p2) {
    static const __m128i sign_bits = _mm_set1_epi8((char)0x80);
    const __m128i a = _mm_xor_si128(_mm_set_epi64((__m64)p1.fm_range_.start_, (__m64)p1.fm_range_.end_), sign_bits);
    const __m128i b = _mm_xor_si128(_mm_set_epi64((__m64)p2.fm_range_.start_, (__m64)p2.fm_range_.end_), sign_bits);
    const int lt = _mm_movemask_epi8(_mm_cmplt_epi8(a, b)) - _mm_movemask_epi8(_mm_cmpgt_epi8(a, b));
    const unsigned int less_than = (lt > 0), equal = (lt == 0), prob_lt = (p1.seed_prob_ < p2.seed_prob_);
    return less_than | (equal & prob_lt);
}
```

and this implementation improved the overall performance indeed. It is also tried to load both two 64-bit integers of `Range` (primary key) and the 32-bit floating number seed probability (secondary key) into 256-bit AVX registers and perform AVX comparisons, but this does not bring any considerable gain.

If FM-index range does not exceed 32-bit, the comparison between `PathBuffer`s can be even faster by directly doing 64-bit integer comparisons without using SSE registers and bring extra overall performance improvement around 3%:

```c++
    const long long i1 = p1.fm_range_.start_ << 32 | p1.fm_range_.end_;
    const long long i2 = p2.fm_range_.start_ << 32 | p2.fm_range_.end_;
    const unsigned int less_than = (i1 < i2), equal = (i1 == i2), prob_lt = (p1.seed_prob_ < p2.seed_prob_);
```

## Other ideas

#### Vectorization of BWT `Occ`

FM-index spends around a quarter of the program running time. In particular, `bwt_occ` is intensively called. There is a loop inside the said function that can be vectorized:

```c
    for (; p < end; p += 2) n += __occ_aux(*(uint64_t*)p, c);
```

where `__occ_aux` is an expression that takes constant time to evaluate and has been accelerated by `POPCNT` instruction mentioned above. One natural idea is to further vectorize this loop for more performance improvement. However, this does not bring any observable mapping speed up. A closer examination finds out that for most of the time, this loop only repeats for fewer than 4 times.

With `POPCNT` optimization applied, `bwt_occ` only spends 9-10ns on average for each call. There seems to be little room for further improvement inside `bwt_occ`.

The same situation also applies to `bwt_2occ`: similar simple loops exist but the actual number of repetitions is at most of the time too low to gain from vectorization on loop.

There is a paper [Vectorized Character Counting for Faster Pattern Matching](https://arxiv.org/abs/1811.06127v2) on SIMD implementation of `Occ` calculation. This is not something can be immediately used into `BWA` library code, but can be heuristic on accelerating FM-index part in `UNCALLED`.

#### Optimization on seed tracker

When adding a new seed to the seed tracker, there is a loop that finds the longest previously existing alignments whose starting position is before the new one, while satisfying a few other conditions.

```c++
void SeedTracker::add_seed(u64 ref_en, u32 ref_len, u32 evt_st) {
    SeedGroup new_aln(Range(ref_en-ref_len+1, ref_en), evt_st);
    //Locations sorted by decreasing ref_en_.start
    //Find the largest aln s.t. aln->ref_en_.start <= new_aln.ref_en_.start
    //AKA r1 <= r2
    auto aln = alignments_.lower_bound(new_aln), aln_match = alignments_.end();

    u64 e2 = new_aln.evt_en_, //new event aln
        r2 = new_aln.ref_en_.start_; //new ref aln

    while (aln != alignments_.end()) {
        u64 e1 = aln->evt_en_, //old event aln
            r1 = aln->ref_en_.start_; //old ref aln

        bool higher_sup = aln_match == alignments_.end() 
                       || aln_match->total_len_ < aln->total_len_,
             
             in_range = e1 <= e2 && //event aln must increase
                        r2 - r1 <= e2 - e1 && //evt increases more than ref (+ skip)
                        (r2 - r1) >= (e2 - e1) / 12; //evt doesn't increase too much
             
        if (higher_sup && in_range) {
            aln_match = aln;
        } else if (r2 - r1 >= e2) {
            break;
        }
        aln++;
    }
```

I have been contemplating a data structure that can locate the desired result, or at least the longest existing alignment prior to the new seed, in `O(1)` time, but I have not come up with an solution yet.

#### Speeding up sorting algorithm

Sorting path buffers on every new event also takes around a quarter of overall program running time. Comparison between path buffers has been optimized as mentioned above. 

There are papers, eg. [Fast Sorting Algorithms using AVX-512 on Intel Knights Landing](https://hal.inria.fr/hal-01512970v1/document), on SIMD vectorized sorting algorithms. These papers solve the problem of sorting an array of scalar values (eg. integers, floating point numbers). I am thinking how to accommodate the case in `UNCALLED` of having an 128-bit integer primary key and a floating number secondary key for comparison, and structures are being sorted instead of scalar values.

#### FM-index on GPU

There are a number of papers, eg. [Boosting the FM-Index on the GPU: Effective Techniques to Mitigate Random Memory Access](https://ieeexplore.ieee.org/document/6975110), on the topic of GPU acceleration of FM-index. They demonstrates the increase in numbers of queries can be done per second. This can be an idea for accelerating FM-index portion in `UNCALLED`.

#### Parallelization of main mapping loop

For each new event, besides a pass of sorting, there are two loops whose number of repetitions vary from a few times to a thousand times. This can probably be parallelized. But the entire process of handling a new event only takes a fraction of a millisecond. Is GPU suitable for parallelizing these loops?