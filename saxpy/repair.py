"""RePair grammar inference.

A pure-Python port of the RePair (Recursive Pairing) algorithm from the sibling
jMotif C++/R project. RePair (Larsson & Moffat, 1999) is an off-line,
dictionary-based grammar compressor: it repeatedly finds the most frequent pair
of adjacent symbols (a *digram*) in the work string and replaces every
occurrence with a new rule symbol, until no digram occurs more than once. The
resulting grammar's rules are the recurring patterns of the input; positions
never covered by any rule are, conversely, the rare / anomalous ones -- the idea
exploited by :func:`saxpy.rra.find_discords_rra`.

The entry point is :func:`str_to_repair_grammar`.

Determinism note: when several digrams share the maximal frequency, which one is
substituted first is an implementation choice. This port keeps a stable order
(Python dict preserves insertion order), so a given input always yields the same
grammar -- but the concrete *rule numbering* need not match the C++ original,
whose order depends on ``std::unordered_map`` iteration. What IS invariant is the
structure: decompressing R0 always reproduces the input, R0 ends up with no
repeated digram, and the set of rule expansions is determined by the input.
"""

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class RepairRule:
    """A single grammar rule (or R0, the compressed top-level string).

    ``rule_id`` 0 is R0. ``expanded_rule_string`` is the full terminal expansion
    of the rule (space-joined, no trailing space). ``occurrences`` are the
    left-symbol token positions where the rule applies, and ``intervals`` the
    corresponding ``(start, end)`` token spans (``end - start`` equals the number
    of spaces in the expansion, i.e. one less than the token count).
    """

    rule_id: int
    rule_string: str = ""
    expanded_rule_string: str = ""
    occurrences: list = field(default_factory=list)
    intervals: list = field(default_factory=list)


class _Symbol:
    """A terminal token in the work string."""

    __slots__ = ("payload", "str_index")

    def __init__(self, payload, str_index):
        self.payload = payload
        self.str_index = str_index

    def is_guard(self):
        return False

    def to_string(self):
        return self.payload


class _Guard(_Symbol):
    """A non-terminal standing in for a rule; reuses the left-symbol position."""

    __slots__ = ("rule",)

    def __init__(self, rule, str_index):
        # A guard occupies the left symbol's original position, so it stays
        # reachable as r0[str_index] for later digram bookkeeping.
        super().__init__(rule.rule_string, str_index)
        self.rule = rule

    def is_guard(self):
        return True

    def get_expanded_string(self):
        return self.rule.expanded_rule_string


class _SymbolRecord:
    """A node of the R0 work string (a doubly-linked list)."""

    __slots__ = ("payload", "prev", "next")

    def __init__(self, payload):
        self.payload = payload
        self.prev = None
        self.next = None


class _PQNode:
    __slots__ = ("payload", "prev", "next")

    def __init__(self, payload):
        self.payload = payload
        self.prev = None
        self.next = None


class _Digram:
    __slots__ = ("digram", "freq")

    def __init__(self, digram, freq):
        self.digram = digram
        self.freq = freq


class _PriorityQueue:
    """Digrams kept in a doubly-linked list sorted by frequency, descending.

    A ``dict`` maps each digram string to its node for O(1) lookup. Ties use
    ``>=`` (a new/raised entry is placed at the front of its frequency band),
    matching the C++ implementation. Entries dropping below frequency 2 are
    evicted -- a digram occurring once cannot found a rule.
    """

    def __init__(self):
        self.head = None
        self.nodes = {}

    def enqueue(self, digram):
        if digram.digram in self.nodes:
            raise ValueError(f"digram already queued: {digram.digram}")
        node = _PQNode(digram)
        if self.head is None:
            self.head = node
        elif node.payload.freq >= self.head.payload.freq:
            self.head.prev = node
            node.next = self.head
            self.head = node
        else:
            curr = self.head
            placed = False
            while curr.next is not None:
                if node.payload.freq >= curr.payload.freq:
                    prev = curr.prev
                    prev.next = node
                    node.prev = prev
                    curr.prev = node
                    node.next = curr
                    placed = True
                    break
                curr = curr.next
            if not placed:
                # curr is the tail.
                if node.payload.freq >= curr.payload.freq:
                    prev = curr.prev
                    prev.next = node
                    node.prev = prev
                    curr.prev = node
                    node.next = curr
                else:
                    node.prev = curr
                    curr.next = node
        self.nodes[digram.digram] = node
        return node.payload

    def dequeue(self):
        if self.head is None:
            return None
        res = self.head
        self.head = self.head.next
        if self.head is not None:
            self.head.prev = None
        del self.nodes[res.payload.digram]
        return res.payload

    def _remove_node(self, node):
        del self.nodes[node.payload.digram]
        if node.prev is None:
            if node.next is not None:
                self.head = node.next
                self.head.prev = None
            else:
                self.head = None
        elif node.next is None:
            node.prev.next = None
        else:
            node.prev.next = node.next
            node.next.prev = node.prev

    def contains_digram(self, digram_string):
        return digram_string in self.nodes

    def update_digram_frequency(self, digram_string, new_value):
        node = self.nodes.get(digram_string)
        if node is None:
            return None
        if new_value == node.payload.freq:
            return node.payload
        if new_value < 2:
            self._remove_node(node)
            return None

        old_freq = node.payload.freq
        node.payload.freq = new_value
        if len(self.nodes) == 1:
            return node.payload

        if new_value > old_freq:
            # Move the node up toward the head.
            if node.prev is None:
                return node.payload
            current = node.prev
            if node.payload.freq <= current.payload.freq:
                return node.payload
            self._remove_node(node)
            node.next = None
            node.prev = None
            while current is not None and current.payload.freq < node.payload.freq:
                current = current.prev
            if current is None:
                node.next = self.head
                self.head.prev = node
                self.head = node
            elif current.next is None:
                current.next = node
                node.prev = current
            else:
                current.next.prev = node
                node.next = current.next
                current.next = node
                node.prev = current
            self.nodes[node.payload.digram] = node
        else:
            # Move the node down toward the tail.
            if node.next is None:
                return node.payload
            current = node.next
            if node.payload.freq >= current.payload.freq:
                return node.payload
            self._remove_node(node)
            node.next = None
            node.prev = None
            while current.next is not None and current.payload.freq > node.payload.freq:
                current = current.next
            if current.next is None:  # hit the tail
                if node.payload.freq > current.payload.freq:
                    if self.head is current:
                        node.next = current
                        current.prev = node
                        self.head = node
                    else:
                        node.next = current
                        node.prev = current.prev
                        current.prev.next = node
                        current.prev = node
                else:
                    current.next = node
                    node.prev = current
            else:
                node.next = current
                node.prev = current.prev
                if current.prev is None:
                    self.head = node
                else:
                    current.prev.next = node
                    current.prev = node
            self.nodes[node.payload.digram] = node
        return node.payload


def _count_spaces(s):
    """Count literal space characters (matches the C++ ``_count_spaces``)."""
    return s.count(" ")


def _safe_remove(lst, value):
    """Remove the first occurrence of ``value`` from ``lst`` if present.

    Mirrors C++ ``erase(std::remove(...))``, which is a no-op when the value is
    absent.
    """
    try:
        lst.remove(value)
    except ValueError:
        pass


def str_to_repair_grammar(s):
    """Infer a RePair grammar from a space-delimited string of tokens.

    Args:
        s: the input string, tokens separated by single spaces (e.g. a SAX
            string from :func:`saxpy.sax.sax_via_window`).

    Returns:
        A ``dict`` mapping ``rule_id -> RepairRule``. Rule 0 is R0, the
        compressed top-level string; its ``expanded_rule_string`` is the
        original input (with a trailing space, as in the reference).

    Examples:
        >>> g = str_to_repair_grammar("abc abc cba cba bac xxx abc abc cba cba bac")
        >>> g[0].rule_string
        'R4 xxx R4'
        >>> g[4].expanded_rule_string
        'abc abc cba cba bac'
    """
    if not isinstance(s, str):
        raise TypeError("RePair input must be a string of space-delimited tokens.")

    # Append a trailing delimiter, as the reference does, then tokenize.
    s = s + " "
    str_length = _count_spaces(s)
    tokens = s.split(" ")
    # split() on the trailing space yields a final empty token; drop it.
    if tokens and tokens[-1] == "":
        tokens = tokens[:-1]

    if not tokens:
        # Empty input: a degenerate grammar with only R0.
        r0 = RepairRule(rule_id=0, rule_string="", expanded_rule_string=s)
        r0.occurrences.append(-1)
        r0.intervals.append((0, str_length))
        return {0: r0}

    next_rule_id = 1

    # The R0 work string: a fixed-length list of records indexed by original
    # token position, with prev/next links. Records are mutated in place.
    r0 = []
    digram_table = defaultdict(list)  # digram string -> list of left positions

    old_token = None
    for i, token in enumerate(tokens):
        rec = _SymbolRecord(_Symbol(token, i))
        r0.append(rec)
        if i > 0:
            digram_str = old_token + " " + token
            digram_table[digram_str].append(i - 1)
            r0[i - 1].next = rec
            rec.prev = r0[i - 1]
        old_token = token

    n = len(r0)  # fixed original token count; boundary tests use this

    queue = _PriorityQueue()
    for digram_str, occ in digram_table.items():
        if len(occ) > 1:
            queue.enqueue(_Digram(digram_str, len(occ)))

    # Each grammar rule's metadata, keyed by rule id.
    rules = {}

    entry = queue.dequeue()
    while entry is not None:
        # Copy this digram's occurrence positions into a local list (the C++
        # copies before mutating the table). Processed from the END (reverse).
        occurrences = list(digram_table[entry.digram])

        first_record = r0[occurrences[0]]
        second_record = first_record.next

        rule_id = next_rule_id
        next_rule_id += 1
        rule_string = "R" + str(rule_id)

        # Expanded string is computed BEFORE the occurrence loop, while the
        # links at occurrences[0] are still pristine (it is processed last).
        first_sym = first_record.payload
        second_sym = second_record.payload
        left = first_sym.get_expanded_string() if first_sym.is_guard() else first_sym.to_string()
        right = (
            second_sym.get_expanded_string() if second_sym.is_guard() else second_sym.to_string()
        )
        expanded = left + " " + right

        rule = RepairRule(
            rule_id=rule_id, rule_string=rule_string, expanded_rule_string=expanded
        )
        rules[rule_id] = rule

        # Insertion-ordered set: a plain Python ``set`` would iterate in an
        # order that varies across processes (string hash randomization),
        # making the grammar non-deterministic. A dict preserves insertion
        # order, so the same input always yields the same grammar.
        new_digrams = {}

        while occurrences:
            occ = occurrences.pop()

            curr_sym = r0[occ]
            next_sym = curr_sym.next
            old_first = curr_sym.payload

            guard = _Guard(rule, occ)
            rule.occurrences.append(occ)

            # Place the guard, and relink around the consumed second symbol.
            curr_sym.payload = guard
            next_not_null = next_sym.next
            curr_sym.next = next_not_null
            if next_not_null is not None:
                next_not_null.prev = curr_sym

            prev_not_null = curr_sym.prev
            # (curr_sym.prev is already correct; mirror the C++ for clarity.)

            # Fix the OLD LEFT digram (prev | old_first) -> (prev | rule).
            if occ > 0 and prev_not_null is not None:
                ole_left_digram = prev_not_null.payload.to_string() + " " + old_first.to_string()
                table_entry = digram_table[ole_left_digram]
                _safe_remove(table_entry, prev_not_null.payload.str_index)
                new_freq = len(table_entry)
                if ole_left_digram == entry.digram:
                    _safe_remove(occurrences, prev_not_null.payload.str_index)
                queue.update_digram_frequency(ole_left_digram, new_freq)
                if new_freq == 0:
                    del digram_table[ole_left_digram]
                    new_digrams.pop(ole_left_digram, None)

                new_left_digram = prev_not_null.payload.to_string() + " " + rule_string
                digram_table[new_left_digram].append(prev_not_null.payload.str_index)
                new_digrams[new_left_digram] = None

            # Fix the OLD RIGHT digram (next_sym | next_not_null) -> (rule | next_not_null).
            if occ < n - 2 and next_not_null is not None:
                ole_right_digram = (
                    next_sym.payload.to_string() + " " + next_not_null.payload.to_string()
                )
                table_entry = digram_table[ole_right_digram]
                new_freq = len(table_entry) - 1
                _safe_remove(table_entry, next_sym.payload.str_index)
                if ole_right_digram == entry.digram:
                    _safe_remove(occurrences, next_sym.payload.str_index)
                queue.update_digram_frequency(ole_right_digram, new_freq)
                if new_freq == 0:
                    del digram_table[ole_right_digram]
                    new_digrams.pop(ole_right_digram, None)

                new_right_digram = rule_string + " " + next_not_null.payload.to_string()
                digram_table[new_right_digram].append(curr_sym.payload.str_index)
                new_digrams[new_right_digram] = None

        # Done with this digram.
        digram_table.pop(entry.digram, None)

        # Enqueue / refresh the newly created digrams that now occur > once.
        for st in new_digrams:
            if len(digram_table.get(st, ())) > 1:
                if queue.contains_digram(st):
                    queue.update_digram_frequency(st, len(digram_table[st]))
                else:
                    queue.enqueue(_Digram(st, len(digram_table[st])))

        entry = queue.dequeue()

    # Compose the final R0 by walking the linked list from the first record.
    parts = []
    ptr = r0[0]
    while ptr is not None:
        parts.append(ptr.payload.payload)
        ptr = ptr.next
    r0_string = " ".join(parts)

    # Assemble results.
    result = {}
    r0_rule = RepairRule(rule_id=0, rule_string=r0_string, expanded_rule_string=s)
    r0_rule.occurrences.append(-1)
    r0_rule.intervals.append((0, str_length))
    result[0] = r0_rule

    for rid, rule in rules.items():
        spaces = _count_spaces(rule.expanded_rule_string)
        rule.intervals = [(occ, occ + spaces) for occ in rule.occurrences]
        result[rid] = rule

    return result
