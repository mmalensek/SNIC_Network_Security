#!/usr/bin/env python3

# Clean ANSI codes from log files and convert to Markdown formatting.
# Run: python txt2md.py log.txt
# Output: log.md

import re
import argparse


def strip_ansi(text):
    # remove ANSI escape sequences from text
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


def remove_input_section(text):
    # remove INPUT section + following empty/whitespace-only lines
    return re.sub(
        r'-{10,}INPUT-{10,}.*?-{10,}\s*',
        '',
        text,
        flags=re.DOTALL
    )


def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # remove ANSI color codes
    clean_text = strip_ansi(content)

    # remove INPUT section
    clean_text = remove_input_section(clean_text)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(clean_text)

    print(f"Clean Markdown saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ANSI log to clean Markdown")
    parser.add_argument("input_file", help="Input .txt log file")
    parser.add_argument(
        "-o", "--output",
        help="Output .md file (default: input.md)"
    )

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output or input_file.replace('.txt', '.md')

    if not input_file.endswith('.txt'):
        print("Warning: Input doesn't end with .txt")

    process_file(input_file, output_file)
