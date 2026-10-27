print("Rag入库开始")
"""Rag入库"""
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)
import os
from langchain_community.document_loaders import PyMuPDFLoader

"""加载文档"""


# def _pdf_loader(file_path):
#     """PDF：逐页提取文本"""
#     loader = UnstructuredPDFLoader(file_path)
#     return loader.load()


def _pdf_loader(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()


def _md_loader(file_path):
    """Markdown：逐页提取文本"""
    loader = UnstructuredMarkdownLoader(file_path)
    return loader.load()


def _excel_loader(file_path):
    """Excel：逐页提取文本"""
    loader = UnstructuredExcelLoader(file_path)
    return loader.load()


def _word_loader(file_path):
    """Word：逐页提取文本"""
    loader = UnstructuredWordDocumentLoader(file_path)
    return loader.load()


def _txt_loader(file_path):
    """文本：逐行提取文本"""
    loader = TextLoader(file_path)
    return loader.load()


LOADER_REGISTRY = {
    ".pdf": _pdf_loader,
    ".md": _md_loader,
    ".xlsx": _excel_loader,
    ".docx": _word_loader,
    ".txt": _txt_loader,
    ".doc": _word_loader,
}


def load_document(file_path):
    """根据文件类型加载文档"""
    ext = os.path.splitext(file_path)[1].lower()
    loader = LOADER_REGISTRY.get(ext, _txt_loader)
    return loader(file_path)
