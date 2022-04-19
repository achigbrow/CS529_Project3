import converter
import argparse


def convert_files():
    """Converts mp3 files to wav files."""
    parser = build_parser()
    options, _ = parser.parse_known_args()
    train_convert = converter.Converter(options.train)
    train_convert.convert(options.train_out)
    test_convert = converter.Converter(options.test)
    test_convert.convert(options.test_out)


def build_parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="parse ml args")
    parser.add_argument(
        "-train",
        type=str,
        default=r"D:\proj3_data\project3\train",
        help="directory containing training mp3s",
    )
    parser.add_argument(
        "-test",
        type=str,
        default=r"D:\proj3_data\project3\test",
        help="directory containing testing mp3s",
    )
    parser.add_argument(
        "-train_out",
        type=str,
        default=r"D:\proj3_data\project3\trainwav",
        help="output directory for train wav files",
    )
    parser.add_argument(
        "-test_out",
        type=str,
        default=r"D:\proj3_data\project3\testwav",
        help="output directory for test wav files",
    )
    return parser


if __name__ == "__main__":
    convert_files()
