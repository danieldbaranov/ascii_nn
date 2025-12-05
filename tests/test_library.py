from ascii_nn import ascii_nn

def test_boat():
    ascii_model = ascii_nn.ascii_nn()

    print(ascii_model.convert("tests/boat.png", n_lines=16))
    print(ascii_model.convert("tests/boat.png", n_lines=32))
    print(ascii_model.convert("tests/boat.png", n_lines=48))
    print(ascii_model.convert("tests/boat.png", n_lines=64))
    assert True

def test_horse():
    ascii_model = ascii_nn.ascii_nn()

    print(ascii_model.convert("tests/horse.png", n_lines=16))
    print(ascii_model.convert("tests/horse.png", n_lines=32))
    print(ascii_model.convert("tests/horse.png", n_lines=48))
    print(ascii_model.convert("tests/horse.png", n_lines=64))
    assert True

if __name__ == "__main__":
    test_boat()
    test_horse()