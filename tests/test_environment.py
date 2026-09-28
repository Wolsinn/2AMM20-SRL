def test_python_environment():
    import sys

    assert sys.version_info.major == 3
    assert sys.version_info.minor == 10
