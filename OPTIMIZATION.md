## Overview

This is an attempt to accelerate UNCALLED with SSE and AVX instruction sets on x86 CPUs. An acceleration of 15-25% is achieved.

Building with GCC on Linux is tested. Building on macOS and Windows has not been tried yet.

The build script targets for `native` architecture. Macro checks are added to check for instruction availability at compile time. It is fine to build the program on a computer without these instruction sets. In that circumstance the original scalar path will be taken. But run-time CPUID check and path selection is not implemented, so do not build the binary on a modern processor and put that executable on an old computer or it will crash miserably.

## Hot spots of the program

Top ten time consuming functions, as reported by GProf, are the following:

```
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  Ks/call  Ks/call  name    
 24.22    268.25   268.25 2831711712   0.00     0.00  operator<(Mapper::PathBuffer const&, Mapper::PathBuffer const&)
  8.54    362.79    94.54 7598691684   0.00     0.00  Mapper::PathBuffer::operator=(Mapper::PathBuffer const&)
  4.22    409.48    46.69    2559737   0.00     0.00  Mapper::map_next()
  3.46    447.82    38.34  954365446   0.00     0.00  Mapper::PathBuffer::make_child(Mapper::PathBuffer&, Range&, unsigned short, float, Mapper::EventType)
  2.51    475.60    27.79 26133361385  0.00     0.00  __gnu_cxx::__normal_iterator<Mapper::PathBuffer*, std::vector<Mapper::PathBuffer, std::allocator<Mapper::PathBuffer> > >::operator*() const
  2.32    501.25    25.65   63764583   0.00     0.00  std::pair<__gnu_cxx::__normal_iterator<Mapper::PathBuffer*, std::vector<Mapper::PathBuffer, std::allocator<Mapper::PathBuffer> > >, bool> pdqsort_detail::partition_right_branchless<__gnu_cxx::__normal_iterator<Mapper::PathBuffer*, std::vector<Mapper::PathBuffer, std::allocator<Mapper::PathBuffer> > >, std::less<Mapper::PathBuffer> >(__gnu_cxx::__normal_iterator<Mapper::PathBuffer*, std::vector<Mapper::PathBuffer, std::allocator<Mapper::PathBuffer> > >, __gnu_cxx::__normal_iterator<Mapper::PathBuffer*, std::vector<Mapper::PathBuffer, std::allocator<Mapper::PathBuffer> > >, std::less<Mapper::PathBuffer>)
  2.31    526.85    25.60 1650204529   0.00     0.00  bwt_occ
  2.09    550.03    23.18 1138613867   0.00     0.00  bwt_invPsi
  1.72    569.12    19.09 5450431722   0.00     0.00  std::vector<Mapper::PathBuffer, std::allocator<Mapper::PathBuffer> >::end()
  1.69    587.83    18.72 8612562513   0.00     0.00  __gnu_cxx::__normal_iterator<Mapper::PathBuffer*, std::vector<Mapper::PathBuffer, std::allocator<Mapper::PathBuffer> > >::__normal_iterator(Mapper::PathBuffer* const&)
```

They can be categorized as

* Sorting path buffers (where comparison between path buffers is massively called)
* Branching new path
* FM-index

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

and this implementation improved the overall performance indeed. It is also tried to load both two 64-bit integers of `Range` (primary key) and the 32-bit floating number seed probability (secondary key) into 256-bit AVX registers and perform AVX comparisons, but this does not bring any considerable performance gain.

If FM-index range does not exceed 32-bit, the comparison between `PathBuffer`s can be even faster by directly doing 64-bit integer comparisons without using SSE registers and bring extra overall performance improvement around 3%:

```c++
    const long long i1 = p1.fm_range_.start_ << 32 | p1.fm_range_.end_;
    const long long i2 = p2.fm_range_.start_ << 32 | p2.fm_range_.end_;
    const unsigned int less_than = (i1 < i2), equal = (i1 == i2), prob_lt = (p1.seed_prob_ < p2.seed_prob_);
```

## Abandoned ideas

#### Porting main loop to GPU

For each new event, besides a pass of sorting, there are two loops whose number of repetitions vary from a few times to a thousand times. This is potentially parallelizable.

However, by timing the loops, it turns out that the entire loop only takes a little time in the magnitude of hundreds microseconds, and each iteration spends just a few microseconds. The idea of parallelizing these two loops on GPU was ruled out, with consideration that the overhead of data transmission between host and GPU could probably overweigh the gain brought by parallelization.

#### FM-index query on GPU

There are a number of papers, eg. [Boosting the FM-Index on the GPU: Effective Techniques to Mitigate Random Memory Access](https://ieeexplore.ieee.org/document/6975110), on the topic of GPU acceleration of FM-index. They demonstrates the increase in numbers of queries can be done per second.

The idea of putting FM-index on GPU may help a lot in the situation of massive amount of asynchronous parallel FM-index queries, but in the case of UNCALLED, for processing the signal data from a single nanopore, FM-index queries are steps in the algorithm in serial with other parts of the program. Without making major changes to the logic flow of UNCALLED, whether porting FM-index to GPU can help accelerate UNCALLED is questionable.

## Failed attempts

#### Vectorization of BWT `Occ`

FM-index spends around a quarter of the program running time. In particular, `bwt_occ` is intensively called. There is a loop inside the said function that can be vectorized:

```c
    for (; p < end; p += 2) n += __occ_aux(*(uint64_t*)p, c);
```

where `__occ_aux` is an expression that takes constant time to evaluate and has been accelerated by `POPCNT` instruction mentioned above. One natural idea is to further vectorize this loop for more performance improvement. However, this does not bring any observable mapping speed up. A closer examination finds out that for most of the time, this loop only repeats for fewer than 4 times.

With `POPCNT` optimization applied, `bwt_occ` only spends 9-10ns on average for each call. There seems to be little room for further improvement inside `bwt_occ`.

The same situation also applies to `bwt_2occ`: similar simple loops exist, but the actual number of repetitions, at most of the time, is too low to gain from vectorization on loop.

#### Speeding up sorting

Sorting path buffers on every new event also takes around a quarter of overall program running time. Comparison between path buffers has been optimized as mentioned above. With that applied, though the time spent on comparison indeed decreased, the time it costs is still considerable. And it is not because a comparison spends a lot of time, but from a very large number of comparisons called.

With little room for further cutting down the time cost by a single comparison, the next idea is to bring the number of comparisons down. It is observed that within sorting, there is often a process of finding the first element less than or greater than the pivot element, searching starting from the first, or reversely from the last element, one by one, in a sub array.

With further detailed observation, it turns out that the desired element is often far away from the starting point. Then it came the idea to do vectorized comparison, in hope of being able to skip through undesired elements faster.

The first attempt is to put the 64-bit FM-index starting position at higher part and 64-bit FM-index ending position at lower part to form a 128-bit comparison key for each element. Then, in a 256-bit AVX register, two elements can be compared with the pivot element simultaneously. When sorting, in an ideal condition, the iterator will skim the sub array in the step of 2, until the comparison reports negative and the desired element is then found. Here the third key, seed probability, is ignored. That is because a strictly equivalent comparison is not required here. What is wanted is to skip undesired elements as fast as possible. And in practice, it is observed that by only comparing the first two keys is indeed sufficient to identify two elements as undesired in most of the time.

Although this does reduce the number of comparator function calls, there is no evident over all performance improvement of the program. More aggressive vectorized comparison was tried. Assuming FM-index range does not exceed 32-bit integers, it was tried to pick the 32-bit FM-index starting position as the only comparison key for an element, and compare 8 elements to the pivot element in a 256-bit AVX register at a time. At this configuration, in some test runs, there is as much as 6% over all performance improvement of the entire program. However, in other runs, with the exact same input, there is no acceleration at all. This seems to be extremely sensitive to system status, and is not a stable optimization.

Apart from the bottom-up approach discussed above, a top-down approach of adopting a vectorized sorting algorithm is also considered. There are papers on the topic of sorting arrays of scalar elements using SIMD instructions. But it is difficult to do SIMD sorting in the case of UNCALLED, where we have two 64-bit integers and a floating point number together as three comparison keys.

## What haven't been done

Improving cache locality has not been explored.
