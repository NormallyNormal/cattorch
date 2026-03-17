from collections import defaultdict


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

    def analyze_lifetimes(self, nodes):
        # Step 1: Count how many times each node is used as an input
        for node in nodes:
            for arg in node.args:
                if hasattr(arg, 'name'):  # If the argument is a previous node
                    self.ref_counts[arg.name] += 1

    def get_list_for_node(self, node):
        # Step 2: Assign a list from the pool or create a new one
        if self.free_pool:
            # Sort to always reuse the lowest-numbered list (cleaner Scratch project)
            self.free_pool.sort()
            assigned_id = self.free_pool.pop(0)
        else:
            self.peak_lists += 1
            assigned_id = self.peak_lists

        self.assignments[node.name] = assigned_id
        return f"T{assigned_id}"

    def release_dependencies(self, node):
        # Step 3: Check if inputs are finished. If so, recycle their lists.
        for arg in node.args:
            if hasattr(arg, 'name'):
                if arg.name in self.assignments:
                    self.ref_counts[arg.name] -= 1
                    if self.ref_counts[arg.name] == 0:
                        # Input is no longer needed by any future node!
                        list_id = self.assignments[arg.name]
                        self.free_pool.append(list_id)
                        # Optional: print(f"Recycling List T{list_id} (last used by {node.name})")