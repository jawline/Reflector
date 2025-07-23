import matplotlib.pyplot as plt

closed = False


def handle_close(evt):
    global closed
    closed = True


def wait_for_key_press():
    closed = False
    while plt.waitforbuttonpress(0.2) is None:
        if closed:
            return False
    return True
