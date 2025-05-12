import threading # For Condition and Lock

class KeyedRemovableQueue:
    """
    A thread-safe Queue implementation that supports enqueue, dequeue,
    and removal of items by a unique key, with blocking operations.
    Items are stored as (key, value) pairs.
    - dequeue() blocks if the queue is empty.
    - peek() blocks if the queue is empty.
    """

    # Inner class for Doubly Linked List Node
    class _Node:
        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None

        def __repr__(self):
            return f"Node(key={self.key}, value={self.value})"

    def __init__(self):
        """
        Initializes an empty KeyedRemovableQueue.
        Uses a dictionary for O(1) key-based access and a doubly linked list for order.
        A threading.Condition is used for blocking and thread safety.
        """
        self._map = {}  # Maps keys to _Node objects
        self._head = self._Node(None, None)  # Sentinel head node
        self._tail = self._Node(None, None)  # Sentinel tail node
        
        self._head.next = self._tail
        self._tail.prev = self._head
        
        self._count = 0 # Number of items in the queue
        
        # Condition variable for managing blocking and thread safety.
        # The Condition object implicitly creates a Lock.
        self._condition = threading.Condition()

    def _remove_node(self, node: _Node):
        """
        Internal helper to remove a node from the list.
        Assumes the lock is already held by the caller.
        """
        if node is None or node == self._head or node == self._tail:
            return
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_node_to_tail(self, node: _Node):
        """
        Internal helper to add a node to the tail of the list.
        Assumes the lock is already held by the caller.
        """
        last_actual_node = self._tail.prev
        last_actual_node.next = node
        node.prev = last_actual_node
        node.next = self._tail
        self._tail.prev = node

    def _evict_if_key_exists(self, key):
        """
        Internal helper to remove an item if its key already exists.
        Assumes the lock is already held by the caller.
        """
        if key in self._map:
            old_node = self._map.pop(key) 
            self._remove_node(old_node)   
            self._count -= 1              

    def enqueue(self, key, value) -> None:
        """
        Adds an item with the given key and value to the rear (tail) of the queue.
        If the key already exists, the old item is removed, and the new item is added to the rear.
        Notifies one waiting thread that an item is available.

        Args:
            key: The unique key for the item.
            value: The value of the item.
        """
        with self._condition: # Acquire lock
            self._evict_if_key_exists(key) 
            
            new_node = self._Node(key, value)
            self._add_node_to_tail(new_node)
            self._map[key] = new_node
            self._count += 1
            
            self._condition.notify() # Wake up one waiting thread (e.g., in dequeue or peek)
            # Lock is released automatically when exiting 'with' block

    def dequeue(self):
        """
        Removes and returns the (key, value) pair of the item at the front (head) of the queue.
        If the queue is empty, this method blocks until an item is available.

        Returns:
            A tuple (key, value) of the item at the front.
        """
        with self._condition: # Acquire lock
            while self.is_empty(): # Must re-check condition after waking up (spurious wakeups)
                self._condition.wait() # Release lock, wait for notify(), then re-acquire lock
            
            # At this point, the queue is not empty and the lock is held
            first_node = self._head.next
            self._remove_node(first_node)
            
            if first_node.key in self._map:
                 del self._map[first_node.key]
            
            self._count -= 1
            return (first_node.key, first_node.value)
            # Lock is released automatically

    def remove_by_key(self, key):
        """
        Removes the item associated with the given key from the queue and returns its value.
        This operation is thread-safe but does not block if the key is not found (raises KeyError).

        Args:
            key: The key of the item to remove.

        Returns:
            The value of the removed item.

        Raises:
            KeyError: If the key is not found in the queue.
        """
        with self._condition: # Acquire lock
            if key not in self._map:
                raise KeyError(f"Key '{key}' not found in queue")
            
            node_to_remove = self._map.pop(key) 
            self._remove_node(node_to_remove)   
            self._count -= 1
            # Note: If this queue had a maxsize, we might notify here
            return node_to_remove.value
            # Lock is released automatically

    def peek(self):
        """
        Returns the (key, value) pair of the item at the front (head) of the queue
        without removing it.
        If the queue is empty, this method blocks until an item is available.

        Returns:
            A tuple (key, value) of the item at the front.
        """
        with self._condition: # Acquire lock
            while self.is_empty():
                self._condition.wait()
            
            first_node = self._head.next
            return (first_node.key, first_node.value)
            # Lock is released automatically

    def is_empty(self) -> bool:
        """
        Checks if the queue is empty. This method is thread-safe.
        Note: For internal checks within locked sections, direct _count access is fine.
        This public method ensures lock acquisition if called externally.
        """
        with self._condition:
            return self._count == 0

    def get_size(self) -> int:
        """Returns the number of items in the queue. Thread-safe."""
        with self._condition:
            return self._count

    def __len__(self) -> int:
        """Allows using len() on the queue instance. Thread-safe."""
        with self._condition:
            return self._count

    def __str__(self) -> str:
        """String representation of the queue for debugging. Thread-safe."""
        with self._condition:
            if self._count == 0: # Use self._count directly as lock is held
                return "KeyedRemovableQueue([])"
            
            items = []
            current = self._head.next
            while current != self._tail:
                items.append(f"({repr(current.key)}: {repr(current.value)})")
                current = current.next
            return f"KeyedRemovableQueue([{', '.join(items)}])"

    def __repr__(self) -> str:
        # For simplicity, __repr__ can be the same as __str__ here,
        # but often __repr__ aims to be a string that could recreate the object.
        with self._condition:
            return self.__str__()

# Example Usage (demonstrating blocking behavior would require threads):
if __name__ == "__main__":
    print("Initializing KeyedRemovableBlockingQueue...")
    krq = KeyedRemovableQueue()
    print(f"Is empty? {krq.is_empty()}, Size: {len(krq)}") 
    print(krq)

    print("\nEnqueuing items...")
    krq.enqueue("task1", "Data for task1")
    krq.enqueue("task2", "Data for task2")
    print(krq)

    print("\nPeeking front item (should not block)...")
    key, value = krq.peek()
    print(f"Peeked: key={key}, value={value}")

    print("\nDequeuing an item (should not block)...")
    key, value = krq.dequeue()
    print(f"Dequeued: key={key}, value={value}") 
    print(krq)

    print("\nRemoving 'task2' by key...")
    removed_value = krq.remove_by_key("task2")
    print(f"Value of removed 'task2': {removed_value}")
    print(krq)
    print(f"Is empty? {krq.is_empty()}, Size: {len(krq)}")

    print("\n--- Demonstrating Blocking (Conceptual - Run this part in separate threads to see actual blocking) ---")
    
    # This part is conceptual for a single-threaded __main__.
    # To truly see blocking, you'd have one thread calling dequeue() on an empty queue,
    # and another thread calling enqueue() later.

    def consumer_thread_func(q, thread_name):
        print(f"[{thread_name}] Attempting to dequeue...")
        try:
            key, value = q.dequeue() # This will block if queue is empty
            print(f"[{thread_name}] Dequeued: key={key}, value={value}")
        except Exception as e:
            print(f"[{thread_name}] Error: {e}")

    def producer_thread_func(q, thread_name, key, value, delay):
        import time
        print(f"[{thread_name}] Will enqueue '{key}' after {delay} seconds...")
        time.sleep(delay)
        q.enqueue(key, value)
        print(f"[{thread_name}] Enqueued: key={key}, value={value}")

    # Example:
    # Create a new empty queue for the threading demo
    blocking_q = KeyedRemovableQueue()

    # Create consumer threads that will block
    consumer1 = threading.Thread(target=consumer_thread_func, args=(blocking_q, "Consumer-1"))
    consumer2 = threading.Thread(target=consumer_thread_func, args=(blocking_q, "Consumer-2"))
    
    print("Starting consumer threads (they should block as queue is empty)...")
    consumer1.start()
    consumer2.start()

    # Create producer threads that will add items after a delay
    producer1 = threading.Thread(target=producer_thread_func, args=(blocking_q, "Producer-1", "itemA", "Value A", 2))
    producer2 = threading.Thread(target=producer_thread_func, args=(blocking_q, "Producer-2", "itemB", "Value B", 4))

    print("Starting producer threads...")
    producer1.start()
    producer2.start()

    # Wait for all threads to complete
    consumer1.join()
    consumer2.join()
    producer1.join()
    producer2.join()

    print("\nBlocking demo finished.")
    print(f"Final queue state: {blocking_q}")
