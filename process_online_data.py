import string
import os
import json

FILE_PATH = "data/music_data.json"
OUTPUT_FILE_PATH = "data/music_training_data.txt"
FILE_SIZE = 1250
output_data = ""
count = 0

def is_chinese(text):
    for char in text:
        # Checking if the character is within the basic and extended Chinese character ranges
        if not (
            '\u4e00' <= char <= '\u9fff' or
            '\u3400' <= char <= '\u4dbf' or
            '\u20000' <= char <= '\u2a6df' or
            '\u2a700' <= char <= '\u2b73f' or
            '\u2b740' <= char <= '\u2b81f' or
            '\u2b820' <= char <= '\u2ceaf' or
            '\uf900' <= char <= '\ufaff' or
            '\u2f800' <= char <= '\u2fa1f'
        ):
            return False
    return True
    
with open(FILE_PATH, 'r', encoding="utf8") as f:
    for line in f:
        count += 1
        data_list = json.loads(line).get('geci')
        for data in data_list:
            for d in data.split(" "):
                if is_chinese(d):
                    output_data += d
        output_data += "\n"

        if len(output_data) > 50000:
            with open(OUTPUT_FILE_PATH, 'a', encoding='utf8') as f:
                f.write(output_data)
            print("current complete:", round(count/FILE_SIZE, 1), "%")
            output_data = ""

