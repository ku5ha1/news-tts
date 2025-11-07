import os
import json
import re
from datetime import datetime, timezone
from pathlib import Path

def clean_text(text):
    """Clean text by removing unnecessary spaces and symbols while preserving content with meaningful paragraph spacing."""
    if not text:
        return ""
    
    # Fix common corrupted characters - use regex for more comprehensive matching
    # Fix "ಮುಖ್ಯಮಂತ್ರಿಗಳಿAದ" pattern - match full string first
    text = re.sub(r'ಮುಖ್ಯಮಂತ್ರಿಗಳಿAದ', 'ಮುಖ್ಯಮಂತ್ರಿಗಳಿಂದ', text)
    text = re.sub(r'iAದ', 'ದಿಂದ', text)
    text = re.sub(r'ದಿAದ', 'ದಿಂದ', text)
    text = re.sub(r'ದೊAದಿಗೆ', 'ದೊಂದಿಗೆ', text)
    text = re.sub(r'ಒSಆಅ', 'ಒಟಿಐ', text)
    text = re.sub(r'ಗಾAಧಿ', 'ಗಾಂಧಿ', text)
    text = re.sub(r'ಮಂಡಳಿಯಿAದ', 'ಮಂಡಳಿಯಿಂದ', text)
    # Fix more corrupted patterns
    text = re.sub(r'ಹಂಚಿಕೊAಡರು', 'ಹಂಚಿಕೊಂಡರು', text)
    text = re.sub(r'ಸಂಬAಧಪಟ್ಟ', 'ಸಂಬಂಧಪಟ್ಟ', text)
    text = re.sub(r'ಕುಟುಂಬದೊAದಿಗೆ', 'ಕುಟುಂಬದೊಂದಿಗೆ', text)
    text = re.sub(r'ಕುಟುಂಬದಿAದ', 'ಕುಟುಂಬದಿಂದ', text)
    text = re.sub(r'ಇಂಜಿನಿಯರಿAಗ್', 'ಇಂಜಿನಿಯರಿಂಗ್', text)
    text = re.sub(r'ಹಿಂದಿನಿAದಲೂ', 'ಹಿಂದಿನಿಂದಲೂ', text)
    # Fix pattern where Aದ appears after Kannada characters
    text = re.sub(r'([ಕ-ಹ])Aದ', r'\1ದಿಂದ', text)
    # Fix pattern where A appears in compound words
    text = re.sub(r'([ಕ-ಹ])A([ಕ-ಹ])', r'\1ಂ\2', text)
    
    # Remove excessive whitespace (multiple spaces, tabs, etc.)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Split into lines and process
    lines = [line.rstrip() for line in text.split('\n')]
    
    # Remove empty lines at the start and end
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop(-1)
    
    # Combine lines that are part of the same sentence/paragraph
    combined_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            # Empty line - keep as paragraph break only if previous line exists
            if combined_lines and combined_lines[-1]:
                combined_lines.append('')
            i += 1
            continue
        
        # Try to combine with following lines until we hit a sentence end or empty line
        while i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            
            # If next line is empty, stop combining
            if not next_line:
                break
            
            # Check if current line ends with sentence punctuation
            ends_with_punctuation = re.search(r'[.!?।]$', line)
            
            # Check if next line starts with special patterns that indicate new paragraph
            starts_new_paragraph = (
                next_line.startswith(('ಬಾಕ್ಸ್', 'ಕೋಟ್', 'ವಿವಿಧ', 'ವಿವರ:', 'ಯೋಜನೆಯ', 'ಯೋಜನೆಗಳ')) and
                len(line) > 20  # Only if current line is substantial
            )
            
            # If line ends with punctuation and next doesn't start with quote/special, it's a new paragraph
            if ends_with_punctuation and not starts_new_paragraph:
                # Check if next line starts with quote or special continuation
                starts_with_quote = next_line.startswith(('"', "'", '``', '-', '—', '–'))
                # Also check if next line is very short (likely continuation)
                is_short_continuation = len(next_line) < 15 and not next_line.endswith(('.', '।', '!', '?'))
                # Check if next line is a single word or very short (likely continuation)
                is_single_word = len(next_line.split()) <= 2 and not next_line.endswith(('.', '।', '!', '?'))
                if not starts_with_quote and not is_short_continuation and not is_single_word:
                    break
            
            # Combine lines
            line = line + ' ' + next_line
            i += 1
        
        combined_lines.append(line)
        i += 1
    
    # Remove consecutive empty lines (keep only single paragraph breaks)
    cleaned_lines = []
    prev_empty = False
    for line in combined_lines:
        if not line.strip():
            if not prev_empty:
                cleaned_lines.append('')
            prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False
    
    # Join lines back with proper paragraph spacing
    text = '\n'.join(cleaned_lines)
    
    # Remove all unnecessary ASCII characters and symbols that might cause translation issues
    # Fix double backticks to proper quotes
    text = re.sub(r'``', '"', text)
    text = re.sub(r"''", '"', text)
    
    # Remove problematic ASCII characters (but keep essential punctuation and Kannada)
    # Keep: . , ! ? : ; - ( ) [ ] { } " ' / \ and Kannada Unicode range
    # Remove: special unicode characters that might cause translation issues
    # But be careful not to remove valid Kannada characters
    # Remove only truly problematic characters like special symbols, but keep numbers and basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?।:\;\-\(\)\[\]\{\}\"\'\'\/\\\n\u0C80-\u0CFF0-9]', '', text)
    
    # Clean up any remaining excessive spaces (multiple spaces to single space)
    text = re.sub(r' +', ' ', text)
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.!?।,])', r'\1', text)
    
    # Remove spaces after opening quotes and before closing quotes
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r'\s+"', '"', text)
    text = re.sub(r"'\s+", "'", text)
    text = re.sub(r"\s+'", "'", text)
    
    # Remove spaces around dashes (but keep the dash)
    text = re.sub(r'\s+-\s+', ' - ', text)
    
    # Remove multiple spaces between words (keep single space)
    text = re.sub(r'([^\s])\s{2,}([^\s])', r'\1 \2', text)
    
    # Final cleanup: ensure proper paragraph spacing (max 2 newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove trailing whitespace from each line
    final_lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(final_lines)
    
    # Final pass: remove any remaining double spaces
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()

def extract_author(content_lines):
    """Extract author name from the end of the content."""
    # Common patterns for author identification
    author_patterns = [
        r'ಲೇಖನ\s*[:-]\s*(.+?)(?:\n|$)',
        r'ಲೇಖಕರು\s*[:-]\s*(.+?)(?:\n|$)',
        r'ವರದಿ\s*[:-]\s*(.+?)(?:\n|$)',
        r'^[-–—]\s*(.+?)(?:\n|$)',  # Lines starting with dash
    ]
    
    # Check last 10 lines for author
    search_lines = content_lines[-10:] if len(content_lines) > 10 else content_lines
    search_text = '\n'.join(search_lines)
    
    for pattern in author_patterns:
        match = re.search(pattern, search_text, re.IGNORECASE | re.MULTILINE)
        if match:
            author = match.group(1).strip()
            # Clean up author name
            author = re.sub(r'\s+', ' ', author)
            # Remove common suffixes
            author = re.sub(r',\s*(ಸಹಾಯಕ|ಹಿರಿಯ|ಪರ್ತಕರ್ತರು|ನಿರ್ದೇಶಕರು|ಇಲಾಖೆ).*$', '', author, flags=re.IGNORECASE)
            if author and len(author) > 2:
                return author.strip()
    
    # If no pattern found, check if last line looks like an author name
    if content_lines:
        last_line = content_lines[-1].strip()
        # If last line is short and doesn't look like content, might be author
        if last_line and len(last_line) < 100 and not last_line.endswith('.'):
            # Check if it contains common author indicators
            if any(indicator in last_line for indicator in ['ಡಾ.', 'ಶ್ರೀ', 'ಎಂ.', 'ಪಿ.', 'ಆರ್.']):
                return last_line
    
    return "Unknown Author"

def remove_author_from_content(content, author):
    """Remove author information from content if found."""
    if not author or author == "Unknown Author":
        return content
    
    lines = content.split('\n')
    cleaned_lines = []
    
    # Remove author patterns from content (Kannada patterns)
    patterns_to_remove = [
        rf'ಲೇಖನ\s*[:-]\s*{re.escape(author)}.*',
        rf'ಲೇಖಕರು\s*[:-]\s*{re.escape(author)}.*',
        rf'ವರದಿ\s*[:-]\s*{re.escape(author)}.*',
        rf'[-–—]\s*{re.escape(author)}.*',
        rf'^{re.escape(author)}.*',
    ]
    
    # For English articles, author might be at the beginning
    # Remove author line if it appears at the start of content
    author_removed = False
    skip_empty_until_author = True
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip empty lines before author is found (for English articles)
        if not line_stripped:
            if skip_empty_until_author:
                continue
            else:
                cleaned_lines.append(line)
            continue
        
        # Check if this line matches the author name (exact match or starts with)
        if not author_removed and (line_stripped == author or line_stripped.startswith(author)):
            # Skip this author line
            author_removed = True
            skip_empty_until_author = False
            continue
        
        # After we've processed the author section, check for Kannada patterns
        if author_removed or not skip_empty_until_author:
            should_remove = False
            # Check Kannada patterns
            for pattern in patterns_to_remove:
                if re.search(pattern, line, re.IGNORECASE):
                    should_remove = True
                    break
            
            if should_remove:
                # Also remove following lines that might be author details
                break
            else:
                cleaned_lines.append(line)
                skip_empty_until_author = False
        else:
            # Before author is found, keep checking
            should_remove = False
            for pattern in patterns_to_remove:
                if re.search(pattern, line, re.IGNORECASE):
                    should_remove = True
                    break
            
            if not should_remove:
                cleaned_lines.append(line)
            else:
                # Also remove following lines that might be author details
                break
    
    # Remove separator lines (like "-----------------")
    final_lines = []
    for i, line in enumerate(cleaned_lines):
        line_stripped = line.strip()
        # Skip separator lines
        if line_stripped and all(c in ['-', '=', '_', '.'] for c in line_stripped) and len(line_stripped) > 5:
            continue
        # Skip lines that look like author details (short lines after separator)
        if i > 0 and cleaned_lines[i-1].strip() and all(c in ['-', '=', '_', '.'] for c in cleaned_lines[i-1].strip()):
            if len(line_stripped) < 50 and any(word in line_stripped for word in ['ಸಹಾಯಕ', 'ನಿರ್ದೇಶಕರು', 'ಇಲಾಖೆ', 'ಪರ್ತಕರ್ತರು']):
                continue
        final_lines.append(line)
    
    return '\n'.join(final_lines).strip()

def convert_txt_to_json(txt_file_path, output_dir):
    """Convert a single txt file to JSON format."""
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"Warning: {txt_file_path} is empty")
            return None
        
        # Extract title and author
        title = ""
        author = ""
        content_start_idx = 0
        author_from_title = None
        
        # Check if this is an English article (first line contains mostly English characters)
        first_non_empty_line = ""
        for line in lines:
            if line.strip():
                first_non_empty_line = line.strip()
                break
        
        # Detect if English article: check if first line has mostly ASCII characters
        is_english = False
        if first_non_empty_line:
            # Check if line contains mostly English characters (not Kannada Unicode range)
            english_chars = sum(1 for c in first_non_empty_line if ord(c) < 128 or c.isspace())
            total_chars = len([c for c in first_non_empty_line if not c.isspace()])
            if total_chars > 0:
                is_english = (english_chars / len(first_non_empty_line)) > 0.7
        
        if is_english:
            # English article format: Line 1 = Title, Line 2 = Subtitle (optional), Author after empty lines
            non_empty_lines = []
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if line_stripped:
                    non_empty_lines.append((i, line_stripped))
            
            if len(non_empty_lines) >= 1:
                # First non-empty line is title
                title = non_empty_lines[0][1]
                
                # Find author: typically a short line (name) that appears before content starts
                # Author is usually one of the first few non-empty lines, short, and doesn't look like content
                author_found = False
                content_start_idx = non_empty_lines[0][0] + 1
                
                # Check lines 2-5 for author (skip line 1 which is title)
                for idx in range(1, min(6, len(non_empty_lines))):
                    line_text = non_empty_lines[idx][1]
                    # Author is typically short (< 100 chars), doesn't end with punctuation, and looks like a name
                    if (len(line_text) < 100 and 
                        not line_text.endswith(('.', '!', '?', ':')) and
                        (any(pattern in line_text for pattern in ['.', 'Dr', 'Mr', 'Mrs', 'Ms', 'Prof']) or 
                         len(line_text.split()) <= 4)):
                        # This looks like an author name
                        author = line_text
                        author_found = True
                        # Content starts after this author line
                        content_start_idx = non_empty_lines[idx][0] + 1
                        break
                
                # If no author found, content starts after title (or subtitle if line 2 exists)
                if not author_found and len(non_empty_lines) >= 2:
                    # Line 2 might be subtitle, content starts after it
                    content_start_idx = non_empty_lines[1][0] + 1
        else:
            # Kannada article format: Line 1 = Title, author at end
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if line_stripped:
                    # Check if first line contains author pattern (like "ವರದಿ :ಡಾ.ವರ ಪ್ರಸಾದ್ ರಾವ್ ಪಿ ವಿ")
                    author_match = re.search(r'ವರದಿ\s*[:-]\s*(.+?)$', line_stripped, re.IGNORECASE)
                    if author_match:
                        author_from_title = author_match.group(1).strip()
                        # Skip this line and use next non-empty line as title
                        continue
                    
                    title = line_stripped
                    content_start_idx = i + 1
                    break
        
        # Clean title: remove extra spaces
        title = re.sub(r'\s+', ' ', title).strip()
        
        if not title:
            print(f"Warning: No title found in {txt_file_path}")
            return None
        
        # Get remaining content
        content_lines = [line.rstrip('\n\r') for line in lines[content_start_idx:]]
        full_content = '\n'.join(content_lines)
        
        # Extract author for Kannada articles
        if not is_english:
            extracted_author = extract_author(content_lines)
            # If author was found in title, use that instead
            if author_from_title:
                author = author_from_title
            else:
                author = extracted_author if extracted_author else "Unknown Author"
        else:
            # For English articles, author is already extracted from line 2
            if not author:
                author = "Unknown Author"
        
        # Remove author from content
        description = remove_author_from_content(full_content, author)
        
        # Clean description
        description = clean_text(description)
        
        # Create JSON structure
        json_data = {
            "title": title,
            "description": description,
            "author": author,
            "newsImage": "https://diprstorageindia.blob.core.windows.net/newsarticles/1761711930490-Screenshot_2025-10-29_095337.png",
            "publishedAt": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "magazineType": "magazine",
            "newsType": "articles"
        }
        
        # Save JSON file
        txt_filename = Path(txt_file_path).stem
        json_filename = f"{txt_filename}.json"
        json_path = os.path.join(output_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"Converted: {txt_filename}.txt -> {json_filename}")
        return json_path
        
    except Exception as e:
        print(f"Error converting {txt_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Convert all txt files in json_input directory to JSON files."""
    # Get the script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Input and output directories
    input_dir = project_root / "app" / "json_input"
    output_dir = project_root / "app" / "json_input"
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all txt files
    txt_files = sorted(input_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Found {len(txt_files)} .txt files to convert\n")
    
    converted_count = 0
    for txt_file in txt_files:
        result = convert_txt_to_json(txt_file, output_dir)
        if result:
            converted_count += 1
    
    print(f"\nConversion complete: {converted_count}/{len(txt_files)} files converted successfully")

if __name__ == "__main__":
    main()

