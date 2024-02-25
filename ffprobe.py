

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple
import os


class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str


def ffprobe(file_path) -> FFProbeResult:
    command_array = ["ffprobe",
                     "-v", "quiet",
                     "-print_format", "json",
                     "-show_format",
                     "-show_streams",
                     file_path]
    cmd = "ffprobe " + "-v " "quiet " +  "-print_format " + "json " +  "-show_format " +  "-show_streams " +  file_path
    result = os.popen(cmd).read()#subprocess.run(command_array, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    return result
                         
def run_ffprobe(src_input):
    
    ffprobe_result = ffprobe(src_input)
    try:
        d = json.loads(ffprobe_result)
        streams = d.get("streams", [])
        return streams[0]

    except:
        print("ERROR")
        #print(ffprobe_result.error, file=sys.stderr)
        return {}