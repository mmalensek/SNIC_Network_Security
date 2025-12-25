#!/usr/bin/env python3


# Clean ANSI codes from log files and convert to Markdown formatting.
# Run: python txt2md.py log.txt
# Output: log.md


import re
import argparse

def strip_ansi(text):
    # remove ANSI escape sequences from text
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', text)

def ansi_to_markdown(text):
    # convert ANSI colors to markdown formatting
    # green (success ✔) -> **bold**
    text = re.sub(r'\[92m(.*?)(?=\[0m|\Z)', r'**✅ \1**', text, flags=re.DOTALL)
    # red (error ✘) -> *italic*
    text = re.sub(r'\[91m(.*?)(?=\[0m|\Z)', r'*❌ \1*', text, flags=re.DOTALL)
    # reset
    text = re.sub(r'\[0m', '', text)
    return text

def process_file(input_file, output_file):
    # strip ANSI to Markdown conversion
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # strip ANSI codes for clean text
    clean_text = strip_ansi(content)
    
    # apply markdown formatting based on original ANSI positions
    # for simplicity, I pattern-match common test result formats
    markdown_content = clean_text
    
    # convert test results pattern
    markdown_content = re.sub(
        r'Test #(\d+):.*?✔',
        r'**Test #\1: ✔**',
        markdown_content
    )
    markdown_content = re.sub(
        r'Test #(\d+):.*?✘',
        r'*Test #\1: ✘*',
        markdown_content
    )
    
    # headers to markdown headers
    markdown_content = re.sub(r'^-{10,}\s*(.*?)\s*-{10,}', r'### \1', markdown_content)
    
    # results percentages
    markdown_content = re.sub(r'Percentage of correct labels:\s*(\d+\.?\d*)%', r'**✅ Correct: \1%**', markdown_content)
    markdown_content = re.sub(r'Percentage of wrong labels:\s*(\d+\.?\d*)%', r'**❌ Wrong: \1%**', markdown_content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Clean Markdown saved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ANSI log to clean Markdown")
    parser.add_argument("input_file", help="Input .txt log file")
    parser.add_argument("-o", "--output", help="Output .md file (default: input_clean.md)")
    
    args = parser.parse_args()
    
    input_file = args.input_file
    output_file = args.output or input_file.replace('.txt', '.md')
    
    if not input_file.endswith('.txt'):
        print("Warning: Input doesn't end with .txt")
    
    process_file(input_file, output_file)
