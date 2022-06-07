#pragma once

#include <bkup/hash.h>
#include <bkup/interval_tree.hpp>
#include <iostream>
#include <map>
#include <mlir/IR/Operation.h>
#include <vector>

typedef struct MemRegion {
  size_t offset;
  size_t size;

  size_t nextOffset() const { return offset + size; }

} MemRegion;

inline bool operator==(const MemRegion &lhs, const MemRegion &rhs) {
  return lhs.offset == rhs.offset && lhs.size == rhs.size;
}

struct regionOffsetCmp {
  bool operator()(const MemRegion &reg1, const MemRegion &reg2) const {
    return reg1.offset == reg2.offset ? reg1.size < reg2.size
                                      : reg1.offset < reg2.offset;
  }
};

typedef struct LiveRange {
  size_t begin;
  size_t end;

} LiveRange;

inline bool operator==(const LiveRange &lhs, const LiveRange &rhs) {
  return lhs.begin == rhs.begin && lhs.end == rhs.end;
}

typedef mlir::Operation *LiveRangeId;
struct UniqueLiveRange {
  LiveRange lvr;
  LiveRangeId id;
  std::string caller;
};

inline bool operator==(const UniqueLiveRange lhs, const UniqueLiveRange rhs) {
  return lhs.lvr == rhs.lvr && lhs.id == rhs.id;
}

struct liveRangeStartCmp {
  bool operator()(const UniqueLiveRange &u1, const UniqueLiveRange &u2) const {
    return u1.lvr.begin == u2.lvr.begin
               ? (u1.lvr.end == u2.lvr.end ? u1.id < u2.id
                                           : u1.lvr.end < u2.lvr.end)
               : u1.lvr.begin < u2.lvr.begin;
  }
};

template <typename T>
using SortedUniqueLiveRangeMap =
    std::map<UniqueLiveRange, T, liveRangeStartCmp>;

struct PlannedAlloc {
  UniqueLiveRange ulvr;
  MemRegion reg;
};

struct memAllocOffsetCmp {
  regionOffsetCmp cmp;
  bool operator()(const PlannedAlloc &u1, const PlannedAlloc &u2) const {
    return cmp(u1.reg, u2.reg);
  }
};

inline bool operator==(const PlannedAlloc lhs, const PlannedAlloc rhs) {
  return lhs.ulvr == rhs.ulvr && lhs.reg == rhs.reg;
}

inline bool valid_add(size_t a, size_t b) {
#if defined(_MSC_VER)
  return a + b >= a;
#else
  size_t _carry = 0;
  return !__builtin_add_overflow(a, b, &_carry);
#endif
}

inline bool valid_sub(size_t a, size_t b) {
#if defined(_MSC_VER)
  return a >= b;
#else
  size_t _carry = 0;
  return !__builtin_sub_overflow(a, b, &_carry);
#endif
}

std::vector<PlannedAlloc> parsePlannedAllocsCsv(std::string fp);

static const std::string RETURN_TO_USER = "RETURN_TO_USER";

bool validateAllocations(std::vector<PlannedAlloc> allocations,
                         SortedUniqueLiveRangeMap<size_t> managed_live_ranges,
                         size_t total_size);

enum GAP_PRIORITY { FIRST, SMALLEST };

std::vector<PlannedAlloc> greedyBySizeWithSmallestGap(
    const SortedUniqueLiveRangeMap<size_t> &live_ranges);

namespace std {

template <> struct hash<MemRegion> {
  size_t operator()(const MemRegion &reg) const {
    return mem_hash::get_hash(reg.offset, reg.size);
  }
};

template <> struct hash<UniqueLiveRange> {
  size_t operator()(const UniqueLiveRange &ulvr) const {
    return mem_hash::get_hash(ulvr.lvr, ulvr.id);
  }
};

template <> struct hash<PlannedAlloc> {
  size_t operator()(const PlannedAlloc &mem) const {
    return mem_hash::get_hash(mem.reg, mem.ulvr);
  }
};

template <> struct hash<LiveRange> {
  size_t operator()(LiveRange const &range) const {
    // shift so that single point ranges don't have hash zero (xor cancels)
    return mem_hash::get_hash(range.begin, range.end);
  }
};

using namespace lib_interval_tree;

template <> struct hash<interval_t<size_t>> {
  std::size_t operator()(const lib_interval_tree::interval_t<size_t> &p) const {
    auto h1 = std::hash<size_t>{}(p.low());
    auto h2 = std::hash<size_t>{}(p.high());

    return mem_hash::get_hash(h1, h2);
  }
};

struct tuple_hash {
  std::size_t operator()(const std::tuple<size_t, size_t, size_t> &p) const {
    auto h1 = std::hash<size_t>{}(std::get<0>(p));
    auto h2 = std::hash<size_t>{}(std::get<1>(p));
    auto h3 = std::hash<size_t>{}(std::get<2>(p));

    return mem_hash::get_hash(mem_hash::get_hash(h1, h2), h3);
  }
};

} // namespace std
