import py_vncorenlp
import os
from typing import Dict, Union
from api.utils.file_utils import get_project_base_directory

os.environ["JAVA_HOME"] = r"/usr/lib/jvm/java-11-openjdk-amd64"

POS_MAPPING = {
    "Np": "NNP",
    "Nc": "NN",
    "Nu": "NN",
    "N": "NN",
    "Ny": "NN",
    "Nb": "FW",
    "V": "VB",
    "Vb": "FW",
    "A": "JJ",
    "P": "PRP",
    "R": "RB",
    "L": "DT",
    "M": "CD",
    "E": "IN",
    "C": "IN",
    "Cc": "CC",
    "I": "UH",
    "T": "RP",
    "Y": "NN",
    "Z": "NN",
    "X": "FW",
    "CH": "PUNCT",
}


class VNCoreNLPTokenizer:
    vn_core_nlp: py_vncorenlp.VnCoreNLP

    def __init__(self):
        self.vn_core_nlp = py_vncorenlp.VnCoreNLP(
            save_dir=get_project_base_directory() + "/rag/res/vn_core_nlp",
            annotators=["wseg", "pos", "ner", "parse"],
        )

    def word_segment(self, line) -> str:
        return " ".join(self.vn_core_nlp.word_segment(line))

    def pos_tag(self, token) -> str:
        annotated_text = self.vn_core_nlp.annotate_text(token)
        for sent in annotated_text.keys():
            list_dict_words = annotated_text[sent]
            for word in list_dict_words:
                if word["wordForm"] == token.replace(" ", "_"):
                    vn_tag = word["posTag"]
                    return POS_MAPPING.get(vn_tag, "FW")
        return "FW"  # Return 'FW' for undefined or unmatched tokens


vn_core_nlp = VNCoreNLPTokenizer()
if __name__ == "__main__":
    line = "Tôi là sinh viên của Đại học Bách khoa Hà Nội. Hiện tại tôi đang làm kỹ sư tại Tập đoàn Công nghiệp - Viễn thông quân đội Viettel."
    print(vn_core_nlp.word_segment(line))
