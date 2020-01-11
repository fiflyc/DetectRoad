import os
import sys
import re
import cv2
from detect_road import detect_line, print_line


def modify_video():
    for filename in os.listdir('./input'):
        name, ext = os.path.splitext(filename)
        source = cv2.VideoCapture('./input/' + filename)

        height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))

        result = cv2.VideoWriter('./output/' + name + "_detected" + ext,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 source.get(cv2.CAP_PROP_FPS),
                                 (width, height))

        while source.isOpened():
            ret, image = source.read()
            if not ret:
                break

            image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            image = print_line(image, detect_line(image))
            result.write(image)

        result.release()
        source.release()
        cv2.destroyAllWindows()


def main(outfile):
    with open(outfile, 'w') as out:
        # TODO
        pass


"""
mode:
    'main'  to find road in the input image. (default)
    'video' to get modified video showing script work.
            In this mode script gets all videos from './input' and saves the output in './output'. All directories must exist!
    'pic'   to get modified picture.
            
in:
    Video frame as image from robot camera.
out:
    File with points of the road border ('main') or
    modified image ('out').
"""


def parse_args():
    mode = 'main'
    infile = None
    outfile = None

    for arg in sys.argv[1:]:
        var, value = re.split(r'=', arg)
        if var == '--mode':
            mode = value
        elif var == '--in':
            infile = value
        elif var == "--out":
            outfile = value
        else:
            exit("No such option: " + var)

    return mode, infile, outfile


if __name__ == '__main__':
    _mode, _infile, _outfile = parse_args()
    if _mode == 'video':
        modify_video()
    elif _infile is None or _outfile is None:
        exit("Use --in= and --out= to choose input and output files")
    elif _mode == 'main':
        main(_outfile)
    elif _mode == 'pic':
        _image = cv2.imread(_infile)
        _image = cv2.cvtColor(cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(_outfile, print_line(_image, detect_line(_image)))
    else:
        exit(_mode + ": No such mode!")
