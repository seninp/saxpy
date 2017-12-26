from saxpy.visit_registry import VisitRegistry


def my_func():
    arr = VisitRegistry(100000)
    while 0 < arr.get_unvisited_count():
        num = arr.get_next_unvisited()
        arr.set_visited(num)


if __name__ == '__main__':
    my_func()
