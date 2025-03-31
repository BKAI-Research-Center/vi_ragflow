import re

MAX_WORD_BY_CHUNK = 150
OVERLAP_SENTENCES = 2


def clean_text(text):
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"\n{2,}", "\n", text)

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if line and line[0].islower() and cleaned_lines:
            cleaned_lines[-1] += " " + line
        else:
            cleaned_lines.append(line)
    cleaned_line_txt = "\n".join(cleaned_lines)
    cleaned_line_txt = reformat_text_with_newlines(cleaned_line_txt)
    return cleaned_line_txt


def split_text_by_dot(text, max_words_per_segment):
    sentences = text.split(". ")
    segments = []
    current_segment = []

    for sentence in sentences:
        words = sentence.split()

        if len(current_segment) + len(words) <= max_words_per_segment:
            current_segment.extend(words + [". "])
        else:
            segments.append(" ".join(current_segment))
            current_segment = words

    if current_segment:
        segments.append(" ".join(current_segment))
    segments = [segment for segment in segments if segment.strip()]
    return segments


def split_text_to_chunks(text, max_word=MAX_WORD_BY_CHUNK, overlap_sentences=OVERLAP_SENTENCES):
    sentences_now = text.split("\n")
    sentences = []
    for i in range(len(sentences_now)):
        if len(sentences_now[i].split()) > max_word:
            segments = split_text_by_dot(sentences_now[i], max_word)
        else:
            segments = [sentences_now[i]]
        sentences.extend(segments)

        # sentences[(i + len(segments)):] = sentences[i:]
        # sentences[i:(i+ len(segments))] = segments

    chunks = []
    current_chunk = []
    current_word_count = 0
    sentence_index = 0

    while sentence_index < len(sentences):
        # print(sentence_index, len(sentences))
        current_chunk = []
        current_word_count = 0
        start_index = sentence_index

        if len(sentences[sentence_index].split()) > max_word:
            chunks.extend(split_text_by_dot(sentences[sentence_index], max_word))
            sentence_index += 1

        while (
            sentence_index < len(sentences)
            and current_word_count + len(sentences[sentence_index].split()) <= max_word
        ):
            current_chunk.append(sentences[sentence_index])
            current_word_count += len(sentences[sentence_index].split())
            sentence_index += 1

        if overlap_sentences > 0 and len(chunks) > 0:
            overlap_chunk = chunks[-1].split("\n")[-overlap_sentences:]
            current_chunk = overlap_chunk + current_chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        if overlap_sentences == 0:
            sentence_index = start_index + len(current_chunk)

    return [("Mở đầu", chunk) for chunk in chunks if chunk.strip()]


HEADING_PRIORITY = {
    "roman_numeral": 1,
    "Điểm": 2,
    "Điều": 3,
    "Khoản": 4,
    # "roman_numeral": 4,
    "decimal_1": 5,
    "decimal_2": 6,
    "decimal_3": 7,
    "decimal_4": 8,
    "letter_dot": 9,
    "letter_paren": 9,
    "letter_slash": 9,
}


def extract_headings_and_chunks(text, max_words, overlap_sentences):
    patterns = [
        (r"^\s*([IVXLCDM]+\.)\s+(.*)", "roman_numeral"),
        (r"^\s*(Điểm\s[^\W\d_].*)", "Điểm"),
        (r"^\s*(Điều\s[IVX\d]+.*)", "Điều"),
        (r"^\s*(Khoản\s\d+(\.\d+)?\s.*)", "Khoản"),
        (r"^\s*(\d+\.)\s+(.*)", "decimal_1"),
        (r"^\s*(\d+\.\d+)\s+(.*)", "decimal_2"),
        (r"^\s*(\d+\.\d+\.\d+)\s+(.*)", "decimal_3"),
        (r"^\s*(\d+\.\d+\.\d+\.\d+)\s+(.*)", "decimal_4"),
        (r"^\s*([a-z]\.)\s+(.*)", "letter_dot"),
        (r"^\s*(/[a-z])\s+(.*)", "letter_slash"),
        (r"^\s*(/[a-z]\))\s+(.*)", "letter_paren"),
        (r"^\s*([a-z]\))\s+(.*)", "letter_paren")
    ]

    headings = []
    lines = text.splitlines()

    for i, line in enumerate(lines):
        for pattern, htype in patterns:
            match = re.match(pattern, line)
            if match:
                headings.append((match.group(1), i, htype, line.strip()))
                break

    chunks = []

    if headings and headings[0][1] > 0:
        intro_text = "\n".join(lines[: headings[0][1]]).strip()
        intro_chunks = split_text_to_chunks(intro_text, max_words, overlap_sentences)
        for chunk in intro_chunks:
            chunks.append(("Mở đầu", "intro", 0, chunk[1]))

    for idx, (heading, start_idx, htype, full_line) in enumerate(headings):
        end_idx = headings[idx + 1][1] if idx + 1 < len(headings) else len(lines)
        chunk_text = "\n".join(lines[start_idx:end_idx]).strip()
        chunks.append((heading, htype, start_idx, chunk_text))

    return chunks, headings


def merge_chunks(chunks, max_words, headings):
    merged_chunks = []
    current_chunk = []
    current_word_count = 0

    for heading, htype, line_number, content in chunks:
        word_count = len(content.split())

        if current_word_count + word_count > max_words and current_chunk:
            merged_chunks.append(current_chunk)
            current_chunk = []
            current_word_count = 0

        current_chunk.append((heading, htype, line_number, content))
        current_word_count += word_count

    if current_chunk:
        merged_chunks.append(current_chunk)

    final_chunks = []

    for chunk_group in merged_chunks:
        merged_heading_list = []
        first_line_is_heading = chunk_group[0][0] is not None
        first_line_heading_level = (
            HEADING_PRIORITY.get(chunk_group[0][1], float("inf"))
            if first_line_is_heading
            else float("inf")
        )
        seen_levels = set()

        for heading, line_idx, htype, full_line in reversed(headings):
            heading_level = HEADING_PRIORITY.get(htype, float("inf"))
            if (
                line_idx < chunk_group[0][2]
                and heading_level < first_line_heading_level
            ):
                if heading_level not in seen_levels:
                    merged_heading_list.append((heading, htype, line_idx, full_line))
                    seen_levels.add(heading_level)

        merged_heading_list.reverse()
        meaningful_chunk = (merged_heading_list, chunk_group)
        final_chunks.append(meaningful_chunk)

    return final_chunks


def table_to_text(table):
    return "\n".join([", ".join(map(str, row)) for row in table])

detect_pattern = re.compile(r"(\s[a-z]\))\s+")

def reformat_text_with_newlines(text):
    reformatted_lines = []
    
    for line in text.split("\n"):
        parts = detect_pattern.split(line)
        rebuilt_line = ""
        
        for part in parts:
            if re.match(r"\s[a-z]\)", part): 
                if rebuilt_line:
                    reformatted_lines.append(rebuilt_line.strip())  
                rebuilt_line = part.strip()  
            else:
                rebuilt_line += " " + part.strip()  
        
        if rebuilt_line:
            reformatted_lines.append(rebuilt_line.strip())
    
    return "\n".join(reformatted_lines)

def main_chunking_law(
    text,
    max_words=MAX_WORD_BY_CHUNK,
    overlap_sentences=OVERLAP_SENTENCES,
    is_clean_text=True,depth=0,
    max_depth=20
):

    if depth >= max_depth or len(text.strip()) < max_words:
        return [text]
    if is_clean_text:
        text = clean_text(text)

    chunks, headings = extract_headings_and_chunks(text, max_words, overlap_sentences)
    final_chunks = merge_chunks(chunks, max_words, headings=headings)

    lst_chunk_final = []
    for merged_heading_list, chunk_group in final_chunks:
        chunk_final = ""
        for heading, htype, line_idx, full_line in merged_heading_list:
            chunk_final += full_line + "\n"

        chunk_final_group = ""
        for chunk in chunk_group:
            chunk_final_group += chunk[3] + "\n"

        if len(chunk_final_group.split()) <= max_words:
            lst_chunk_final.append(chunk_final + chunk_final_group)
            continue

        list_chunk_final_group = main_chunking_law(
            " ".join(chunk_final_group.split()[:]), is_clean_text=False, depth=depth+1
        )

        for chunk_final_sub in list_chunk_final_group:
            lst_chunk_final.append(chunk_final + chunk_final_sub)
    filtered_lst_chunk_final = []
    for i in range(len(lst_chunk_final)):
        if i < len(lst_chunk_final) - 1 and lst_chunk_final[i] in lst_chunk_final[i + 1]:
            continue  
        filtered_lst_chunk_final.append(lst_chunk_final[i])
        
    return filtered_lst_chunk_final