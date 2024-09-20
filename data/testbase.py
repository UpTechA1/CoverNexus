from method import double_the_difference




def check(candidate):
    assert (candidate([]) == 0), 'This prints if this assert fails 1 (good for debugging!)'
    assert (candidate([5, 4]) == 25), 'This prints if this assert fails 2 (good for debugging!)'

if __name__ == '__main__':
    check(double_the_difference)