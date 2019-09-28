import operator

class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation
    
    def reduce(self, start = 0, end = None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce(start, end, 1, 0, self._capacity-1)
    
    def _reduce(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end)//2
        if end <= mid:
            return self._reduce(start, end, node*2, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce(start, end, node*2+1, mid+1, node_end)
            else:
                return self._operation(
                    self._reduce(start, mid, node*2, node_start, mid),
                    self._reduce(mid+1, end, node*2+1, mid+1, node_end)
                )

    def __setitem__(self, idx, val):
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[idx*2],
                self._value[idx*2+1]
            )
            idx //= 2
    
    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity = capacity,
            operation = operator.add,
            neutral_element = 0.0
        )
    
    def sum(self, start=0, end = None):
        return super(SumSegmentTree, self).reduce(start, end)
    
    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:
            if self._value[2*idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity
    
class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity = capacity,
            operatoion = min,
            neutral_element = float('inf')
        )
    
    def min(self, start = 0, end = None):
        return super(MinSegmentTree, self).reduce(start, end)

if __name__ == '__main__':
    a = SumSegmentTree(4)
    for i in range(4):
        a[i] = i + 1
    pass
    b = a.find_prefixsum_idx(5)
    pass