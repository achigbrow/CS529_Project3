from utils import converter
import argparse


def convert_files(train, train_out, test, test_out):
    """Converts mp3 files to wav files.

    :param train: full path to directory containing mp3 training files
    :param train_out: full path to directory where training .wav files should be populated
    :param test: full path to directory containing mp3 test files
    :param test_out: full path to directory where testing .wav files should be populated
    :return: none
    """
    train_convert = converter.Converter(train)
    train_convert.convert(train_out)
    test_convert = converter.Converter(test)
    test_convert.convert(test_out)


def build_parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="parse ml args")
    parser.add_argument(
        "-train",
        type=str,
        default="D:/proj3_data/project3/train",
        help="directory containing training mp3s",
    )
    parser.add_argument(
        "-test",
        type=str,
        default="D:/proj3_data/project3/test",
        help="directory containing testing mp3s",
    )
    parser.add_argument(
        "-train_out",
        type=str,
        default="D:/proj3_data/project3/trainwav",
        help="output directory for train wav files",
    )
    parser.add_argument(
        "-test_out",
        type=str,
        default="D:/proj3_data/project3/testwav",
        help="output directory for test wav files",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    options, _ = parser.parse_known_args()
    convert_files(options.train, options.train_out, options.test, options.test_out)
