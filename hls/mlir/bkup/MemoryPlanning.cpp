#include "MemoryPlanning.h"
#include <algorithm>
#include <bkup/interval_tree.hpp>
#include <limits>
#include <unordered_map>

using namespace lib_interval_tree;
bool lenCmp(std::pair<UniqueLiveRange, size_t> p1,
            std::pair<UniqueLiveRange, size_t> p2) {
  auto ulvr1 = p1.first;
  auto size1 = p1.second;
  auto ulvr2 = p2.first;
  auto size2 = p2.second;
  auto cmp = liveRangeStartCmp();

  auto len1 = ulvr1.lvr.end - ulvr1.lvr.begin;
  auto len2 = ulvr2.lvr.end - ulvr2.lvr.begin;
  return len1 == len2 ? (size1 == size2 ? cmp(ulvr1, ulvr2) : size1 > size2)
                      : len1 > len2;
}

// sort tensor usage records in non-increasing order of size (breaking ties by
// comparing live range starts)
bool sizeCmp(std::pair<UniqueLiveRange, size_t> p1,
             std::pair<UniqueLiveRange, size_t> p2) {
  auto ulvr1 = p1.first;
  auto size1 = p1.second;
  auto ulvr2 = p2.first;
  auto size2 = p2.second;
  auto cmp = liveRangeStartCmp();

  return size1 == size2 ? cmp(ulvr1, ulvr2) : size1 > size2;
}

using Cmp = bool((std::pair<UniqueLiveRange, size_t> p1,
                  std::pair<UniqueLiveRange, size_t> p2));

size_t findGapOffset(
    UniqueLiveRange unalloced_ulvr, size_t size,
    std::unordered_map<interval_t<size_t>, PlannedAlloc> current_allocations,
    interval_tree_t<size_t> current_allocation_tree,
    GAP_PRIORITY gap_priority) {
  auto size_max = std::numeric_limits<size_t>::max();
  size_t best_gap = size_max;
  size_t best_offset = size_max;
  size_t prev_offset = 0;

  // find overlaps
  std::vector<PlannedAlloc> sorted_overlaps;
  current_allocation_tree.overlap_find_all(
      {unalloced_ulvr.lvr.begin, unalloced_ulvr.lvr.end},
      [&current_allocations, &sorted_overlaps](auto iter) {
        auto alloc = current_allocations[*iter];
        sorted_overlaps.emplace_back(alloc);
        return true;
      });
  // sort by offset
  std::sort(sorted_overlaps.begin(), sorted_overlaps.end(),
            memAllocOffsetCmp());

  for (auto &alloc : sorted_overlaps) {
    // this happens because prev_offset is updated to the end of the
    // allocated region (i.e. offset+size), while the sort
    // is on the beginning
    if (alloc.reg.offset >= prev_offset) {
      auto gap = alloc.reg.offset - prev_offset;
      if (size <= gap && gap < best_gap) {
        best_offset = prev_offset;
        best_gap = gap;
        if (gap_priority == GAP_PRIORITY::FIRST)
          break;
      }
    }
    prev_offset = std::max(prev_offset, alloc.reg.nextOffset());
  }

  if (best_offset == size_max) {
    best_offset = prev_offset;
  }
  return best_offset;
}

std::vector<PlannedAlloc> orderAllocations(
    std::unordered_map<interval_t<size_t>, PlannedAlloc> current_allocations) {
  std::vector<PlannedAlloc> ordered_allocations;
  ordered_allocations.reserve(current_allocations.size());
  for (auto &item : current_allocations) {
    ordered_allocations.emplace_back(item.second);
  }

  auto final_order_cmp = liveRangeStartCmp();
  std::sort(ordered_allocations.begin(), ordered_allocations.end(),
            [&final_order_cmp](auto m1, auto m2) {
              return final_order_cmp(m1.ulvr, m2.ulvr);
            });

  return ordered_allocations;
}

constexpr size_t gAlignment = 64;
inline size_t computeAlignedTensorSize(size_t nbytes) {
  // Note: everything below is size_t
  return nbytes;
  //  return (nbytes + gAlignment - 1) & (~(gAlignment - 1));
}

std::vector<PlannedAlloc>
greedyBy(Cmp cmp, GAP_PRIORITY gap_priority,
         const SortedUniqueLiveRangeMap<size_t> &sorted_reqs) {
  std::vector<std::pair<UniqueLiveRange, size_t>> sorted_size_live_ranges(
      sorted_reqs.begin(), sorted_reqs.end());
  std::sort(sorted_size_live_ranges.begin(), sorted_size_live_ranges.end(),
            cmp);

  std::unordered_map<interval_t<size_t>, PlannedAlloc> current_allocations;
  interval_tree_t<size_t> current_allocations_tree;
  for (auto &item : sorted_size_live_ranges) {
    auto ulvr = item.first;
    if (ulvr.caller == RETURN_TO_USER) {
      current_allocations.insert({{ulvr.lvr.begin, ulvr.lvr.end},
                                  {{ulvr},
                                   {std::numeric_limits<size_t>::max(),
                                    std::numeric_limits<size_t>::max()}}});
      continue;
    }
    auto size = item.second;
    auto aligned_size = computeAlignedTensorSize(size);
    auto offset = findGapOffset(ulvr, aligned_size, current_allocations,
                                current_allocations_tree, gap_priority);
    current_allocations.insert(
        {{ulvr.lvr.begin, ulvr.lvr.end}, {{ulvr}, {offset, aligned_size}}});
    current_allocations_tree.insert({ulvr.lvr.begin, ulvr.lvr.end});
  }

  return orderAllocations(current_allocations);
}

std::vector<PlannedAlloc> greedyBySizeWithSmallestGap(
    const SortedUniqueLiveRangeMap<size_t> &live_ranges) {
  return greedyBy(sizeCmp, GAP_PRIORITY::SMALLEST, live_ranges);
}
