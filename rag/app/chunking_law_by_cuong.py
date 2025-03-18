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

    return "\n".join(cleaned_lines)


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


def split_text_to_chunks(text, max_word, overlap_sentences):
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
    "Điểm": 1,
    "Điều": 2,
    "Khoản": 3,
    "decimal_1": 4,
    "decimal_2": 5,
    "decimal_3": 6,
    "decimal_4": 7,
    "letter_dot": 8,
    "letter_paren": 8,
    "letter_slash": 8,
}


def extract_headings_and_chunks(text, max_words, overlap_sentences):
    patterns = [
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


def main_chunking_law(
    text,
    max_words=MAX_WORD_BY_CHUNK,
    overlap_sentences=OVERLAP_SENTENCES,
    is_clean_text=True,
):
    if is_clean_text:
        text = clean_text(text)
    chunks, headings = extract_headings_and_chunks(text, max_words, overlap_sentences)
    final_chunks = merge_chunks(chunks, max_words, headings=headings)

    lst_chunk_final = []
    for merged_heading_list, chunk_group in final_chunks:
        chunk_final = ""
        for heading, htype, line_idx, full_line in merged_heading_list:
            chunk_final = chunk_final + full_line + "\n"
        for chunk in chunk_group:
            chunk_final = chunk_final + chunk[3] + "\n"
        lst_chunk_final.append(chunk_final)

    return lst_chunk_final
