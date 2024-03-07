# Author: jayasingam adhuran

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple
import os
import shutil

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
    ffprobe_settings = shutil.which('ffprobe')
    cmd = str(ffprobe_settings) + " " + "-v " "quiet " +  "-print_format " + "json " +  "-show_format " +  "-show_streams " +  file_path
    result = os.popen(cmd).read()#subprocess.run(command_array, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    return result
                         
def run_ffprobe(src_input):
    
    ffprobe_result = ffprobe(src_input)
    try:
        d = json.loads(ffprobe_result)
        streams = d.get("streams", [])
        print("FFPROBE SUCCESS")
        return streams[0]

    except:
        print("ERROR: FFPROBE NOT WORKING OR FILEPATH IS MISSING")
        #print(ffprobe_result.error, file=sys.stderr)
        return {}