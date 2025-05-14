#!/usr/bin/env python3
import re

def fix_indentation_issues(file_path):
    """Fix mixed indentation in Python files by converting all indentation to spaces."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Standard is 4 spaces per indentation level
    tab_size = 4
    fixed_lines = []
    
    # Analyze current line indentation
    for line_num, line in enumerate(lines, 1):
        if line.strip() == '':
            # Keep empty lines as is
            fixed_lines.append(line)
            continue
        
        # Calculate leading whitespace
        leading_space = len(line) - len(line.lstrip())
        
        # Skip lines that are just whitespace
        if leading_space == len(line):
            fixed_lines.append(line)
            continue
        
        # Get the indentation part and the rest of the line
        indent = line[:leading_space]
        rest_of_line = line[leading_space:]
        
        # Convert any tabs to spaces (each tab = tab_size spaces)
        spaces_indent = indent.replace('\t', ' ' * tab_size)
        
        # Ensure indent is a multiple of tab_size (4 spaces per level)
        num_spaces = len(spaces_indent)
        adjusted_spaces = ' ' * (num_spaces - (num_spaces % tab_size) if num_spaces % tab_size < 2 
                                 else num_spaces + (tab_size - num_spaces % tab_size))
        
        # Create fixed line
        fixed_line = adjusted_spaces + rest_of_line
        fixed_lines.append(fixed_line)
        
        # Debug critical lines (around line 725)
        if 720 <= line_num <= 730:
            print(f"Line {line_num}: {repr(line)} -> {repr(fixed_line)}")
    
    # Write fixed content back to file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(fixed_lines)
    
    print(f"Fixed indentation issues in {file_path}")

if __name__ == "__main__":
    app_file = "src/app.py"
    fix_indentation_issues(app_file) 