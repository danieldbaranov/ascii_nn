from ascii_nn import ascii_nn
import cv2

def test_boat():
    ascii_model = ascii_nn.ascii_nn()

    # Load Image
    source_img = cv2.imread("tests/boat.png")

    print(ascii_model.convert(source_img, n_lines=16))
    print(ascii_model.convert(source_img, n_lines=32))
    print(ascii_model.convert(source_img, n_lines=48))
    print(ascii_model.convert(source_img, n_lines=64))
    assert True

def test_horse():
    ascii_model = ascii_nn.ascii_nn()

    source_img = cv2.imread("tests/horse.png")

    print(ascii_model.convert(source_img, n_lines=16))
    print(ascii_model.convert(source_img, n_lines=32))
    print(ascii_model.convert(source_img, n_lines=48))
    print(ascii_model.convert(source_img, n_lines=64))
    assert True

if __name__ == "__main__":
    test_boat()
    test_horse()