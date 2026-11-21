from collections import defaultdict
import heapq


class ScopeManager:
    def __init__(self):
        # Maps node name -> the Scratch list index assigned to it
        self.assignments = {}
        # Keeps track of which Scratch lists are currently "not in use"
        self.free_pool = []
        # Total number of Scratch lists we ended up needing
        self.peak_lists = 0
        # Tracks how many times a node's output will be used
        self.ref_counts = defaultdict(int)
        self._aliases = {}

    def analyze_lifetimes(self, nodes, skip=None, aliases=None):
        # Step 1: Count how many times each node is used as an input.
        # Skipped (aliased) nodes don't get counted, and every use is charged
        # to the alias root that owns the data, so a list stays live until
        # the last read through any of its views. Graph outputs are pinned.
        skip = skip or set()
        self._aliases = dict(aliases or {})
        for node in nodes:
            if node.name in skip:
                continue
            for root in self._roots(node):
                self.ref_counts[root] += 1
                if node.op == "output":
                    # Never released: output lists are read after the run.
                    self.ref_counts[root] += 1

    def _roots(self, node):
        roots = []
        for value in node.all_input_nodes:
            name = value.name
            visited = set()
            while name in self._aliases and name not in visited:
                visited.add(name)
                name = self._aliases[name]
            roots.append(name)
        return roots

    def get_list_for_node(self, node):
        # Step 2: Assign a list from the pool or create a new one
        assigned_id = self.acquire_list()

        self.assignments[node.name] = assigned_id
        return f"T{assigned_id}"

    def acquire_list(self) -> int:
        """Return the lowest reusable list ID without repeatedly sorting the pool."""
        if self.free_pool:
            return heapq.heappop(self.free_pool)
        self.peak_lists += 1
        return self.peak_lists

    def release_list(self, list_id: int) -> None:
        """Make a temporary list ID available for deterministic lowest-ID reuse."""
        heapq.heappush(self.free_pool, list_id)

    def release_dependencies(self, node):
        # Step 3: Check if inputs are finished. If so, recycle their lists.
        for name in self._roots(node):
            if name in self.assignments:
                self.ref_counts[name] -= 1
                if self.ref_counts[name] == 0:
                    # Input is no longer needed by any future node!
                    self.release_list(self.assignments[name])
