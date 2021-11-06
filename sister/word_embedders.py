from pathlib import Path
from typing import List

import gensim
import numpy as np
from fasttext import load_model

from sister import download


def get_fasttext(lang: str = "en"):
    # Download.
    urls = {
        "be": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.be.zip",
        "bg": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.bg.zip",
        "cs": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.cs.zip",
        "en": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip",
        "et": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.et.zip",
        "fr": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fr.zip",
        "hr": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hr.zip",
        "hu": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hu.zip",
        "hy": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hy.zip",
        "ja": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ja.zip",
        "lt": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.lt.zip",
        "mk": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.mk.zip",
        "pl": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.pl.zip",
        "ro": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ro.zip",
        "ru": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ru.zip",
        "sl": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sl.zip",
        "sr": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sr.zip",
        "uk": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.uk.zip",
    }
    path = download.cached_download(urls[lang])
    path = Path(path)
    dirpath = path.parent / "fasttext" / lang
    download.cached_unzip(path, dirpath)

    print("Loading model...")
    filename = Path(urls[lang]).stem + ".bin"
    model = load_model(str(dirpath / filename))
    return model


def get_word2vec(lang: str = "en"):
    # Download.
    urls = {
        "en": "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz",
        "ja": "http://public.shiroyagi.s3.amazonaws.com/latest-ja-word2vec-gensim-model.zip",
    }
    path = download.cached_download(urls[lang])
    path = Path(path)

    filename = "word2vec.gensim.model"

    print("Loading model...")

    if lang == "ja":
        dirpath = Path(download.get_cache_directory(str(Path("word2vec"))))
        download.cached_unzip(path, dirpath / lang)
        model_path = dirpath / lang / filename
        model = gensim.models.Word2Vec.load(str(model_path))

    if lang == "en":
        dirpath = Path(download.get_cache_directory(str(Path("word2vec") / "en")))
        model_path = dirpath / filename
        download.cached_decompress_gzip(path, model_path)
        model = gensim.models.KeyedVectors.load_word2vec_format(
            str(model_path), binary=True
        )

    return model


class WordEmbedding(object):
    def get_word_vector(self, word: str) -> np.ndarray:
        raise NotImplementedError

    def get_word_vectors(self, words: List[str]) -> np.ndarray:
        vectors = []
        for word in words:
            vectors.append(self.get_word_vector(word))
        return np.array(vectors)


class FasttextEmbedding(WordEmbedding):
    def __init__(self, lang: str = "en", model_path: str = None) -> None:
        """Word embedding model.
        Args:
            lang (str): Target language code. Default: en
            model_path (str): Path to local model. Default: None
        """
        if model_path:
            model = load_model(model_path)
        else:
            model = get_fasttext(lang)
        self.model = model

    def get_word_vector(self, word: str) -> np.ndarray:
        return self.model.get_word_vector(word)


class Word2VecEmbedding(WordEmbedding):
    def __init__(self, lang: str = "en", model_path: str = None) -> None:
        """Word embedding model.
        Args:
            lang (str): Target language code. Default: en
            model_path (str): Path to local model. Default: None
        """
        if model_path:
            model = gensim.models.KeyedVectors.load(model_path)
        else:
            model = get_word2vec(lang)
        self.model = model

    def get_word_vector(self, word: str) -> np.ndarray:
        if word in self.model:
            return self.model[word]
        else:
            return np.random.rand(
                self.model.vector_size,
            )
