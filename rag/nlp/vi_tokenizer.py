# import py_vncorenlp
import os
from typing import Dict, Union
from api.utils.file_utils import get_project_base_directory
from underthesea import word_tokenize, pos_tag
import logging

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


# class VNCoreNLPTokenizer:
#     vn_core_nlp: py_vncorenlp.VnCoreNLP

#     def __init__(self):
#         self.vn_core_nlp = py_vncorenlp.VnCoreNLP(
#             save_dir=get_project_base_directory() + "/rag/res/vn_core_nlp",
#             annotators=["wseg", "pos", "ner", "parse"],
#         )

#     def word_segment(self, line) -> str:
#         return " ".join(self.vn_core_nlp.word_segment(line))

#     def pos_tag(self, token) -> str:
#         try:
#             annotated_text = self.vn_core_nlp.annotate_text(token)
#             for sent in annotated_text.keys():
#                 list_dict_words = annotated_text[sent]
#                 for word in list_dict_words:
#                     if word["wordForm"] == token.replace(" ", "_"):
#                         vn_tag = word["posTag"]
#                         return POS_MAPPING.get(vn_tag, "FW")
#             return "FW"  # Return 'FW' for undefined or unmatched tokens
#         except Exception as e:
#             logging.exception(f"Current error token: {token}.\nException: {e}")
#             return "FW"

class UndertheseaTokenizer:
    def word_segment(self, line: str) -> str:
        return word_tokenize(line,format="text")

    def pos_tag(self, token: str) -> str:
        try:
            annotated = pos_tag(token)
            for word, tag in annotated:
                if word == token:
                    return POS_MAPPING.get(tag, "FW")
            return "FW"
        except Exception as e:
            logging.exception(f"Current error token: {token}.\nException: {e}")
            return "FW"


if __name__ == "__main__":
    line = "4. Chính phủ quyết định:a) Gia nhập điều ước quốc tế nhiều bên nhân danh Chính phủ trong thời hạn mười lăm ngày, kể từ ngày nhận được hồ sơ do cơ quan đề xuất trình hoặc kể từ ngày nhận được ý kiến của Quốc hội, Uỷ ban thường vụ Quốc hội về việc gia nhập điều ước quốc tế nhiều bên có điều khoản trái hoặc chưa được quy định trong văn bản quy phạm pháp luật của Quốc hội, Uỷ ban thường vụ Quốc hội hoặc điều ước quốc tế mà để thực hiện cần sửa đổi, bổ sung, bãi bỏ hoặc ban hành văn bản quy phạm pháp luật của Quốc hội, Uỷ ban thường vụ Quốc hội;"
    print(underthesea.word_segment(line))
    # print(underthesea.pos_tag("Đại học Bách khoa Hà Nội"))
    # print(underthesea.pos_tag("sinh viên"))
    # print(underthesea.pos_tag("làm"))
    # print(underthesea.pos_tag("tại"))
    # print(underthesea.pos_tag("Tập đoàn Công nghiệp - Viễn thông quân đội Viettel"))
