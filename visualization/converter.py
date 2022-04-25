import os
from pydub import AudioSegment


class Converter:
    def __init__(self, directory):
        self.directory = directory

    def convert(self, output_dir):
        mp3s = os.listdir(self.directory)

        for mp3 in mp3s:
            self.convert_mp3(mp3, self.directory, output_dir)

    def convert_mp3(self, mp3, directory, output_dir):
        name = mp3[:-3]
        source = r"{0:s}\{1:s}".format(directory, mp3)
        destination = r"{0:s}\{1:s}wav".format(output_dir, name)

        # convert mp3 to wave
        sound = AudioSegment.from_mp3(source)
        sound.export(destination, format="wav")
