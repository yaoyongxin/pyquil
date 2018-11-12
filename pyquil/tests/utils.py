from pyquil.parser import parse


def parse_equals(quil_string, *instructions):
    expected = list(instructions)
    actual = parse(quil_string)
    assert expected == actual, f'{[i.out() for i in actual]} did not match expected {[i.out() for i in expected]}'
