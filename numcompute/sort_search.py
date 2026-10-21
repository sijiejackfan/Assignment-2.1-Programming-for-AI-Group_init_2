"""
Sorts, argpartition/Top-k, quickselect, binary_search
- **Sorting**:
  - Stable sort wrapper (`np.sort(kind='stable')`)
  - Multi-key sort (sort by multiple columns)
- **Top-k / Partial Sort**:
  - `topk(values, k, largest=True, return_indices=True)` using `np.argpartition`
  - Implement **quickselect** for educational purposes
- **Searching**:
  - `binary_search(sorted_array, x)` returning insertion index and existence boolean
"""