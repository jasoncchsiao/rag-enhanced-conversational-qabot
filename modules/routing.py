from typing import List, Dict
import json
import copy
from collections import defaultdict
from datetime import datetime
import pickle

STOPWORDS = ['.txt','.csv']

class Routing:
    def __init__(self, file_path: str=Noen) -> None:
        self.file_path = file_path
        self.hashmap = defaultdict(set)

    def _build_hashmap(self, exist: bool=True, threshold: int=200,
                       output_path: str=None) -> None:
        """Build a hashmap w/ categories as keys and corresponding docs as value lists"""
        countmap: Dict[str, int] = {}
        if exist:
            self._get_existing_hashmap(output_path)
        else:
            with open(self.file_path, 'r') as f:
                embeds = json.load(f)
                n = embeds['embeddings'].__len__()
                unique_filenames = set()
                for i in range(n):
                    if embeds['embeddings'][i]['source_doc'] not in unique_filenames:
                        unique_filenames.add(embeds['embeddings'][i]['source_doc'])
                for filename in unique_filenames:
                    words = set()
                    for word in filename.split('--'):
                        if word.find('-') != -1:
                            split_words = word.split('-')
                            for single_word in split_words:
                                words.add(single_word)
                        else:
                            words.add(word)
                # for filename in unique_filenames:
                #     words = filename.split('--')
                    words = [self._remove_stopwords(word).lower() for word in words]
                    for word in words:
                        if word not in self.hashmap:
                            countmap[word] = 1
                        else:
                            countmap[word] += 1
                    for word in words:
                        self.hashmap[word].add(filename)
                keys = copy.deepcopy([key for key in countmap.keys()])
                for k in keys:
                    if k == '' or countmap[k] < threshold:
                        del countmap[k]
                        del self.hashmap[k]
                for i, filename in enumerate(unique_filenames):
                    self.hashmap['others'].add(filename)
                self._write_to_file(self.hashmap, output_path)

    def _write_to_file(self, mapping: Dict, output_path: str=None) -> None:
        with open(output_path, 'wb') as f:
            pickle.dump(mapping, f)

    def _remove_stopwords(self, text: str) -> str:
        for stopword in STOPWORDS:
            if text.find(stopword) != -1:
                text = text.replace(stopword, '')
        return text

    def _get_existing_hashmap(self, hashmap_path: str=None) -> None:
        with open(hashmap_path, 'rb') as f:
            self.hashmap = pickle.load(f)

    def extract_categories(self, user_query: str=None) -> List[str]:
        """The only external interface open to client end"""
        words = [word.lower() for word in user_query.strip().split()]
        visited = set()
        for word in words:
            if word in self.hashmap:
                visited.add(word)
        if len(visited) == 0:
            return self.hashmap['others']
        else:
            multi_cats = [self.hashmap[key] for key in visited]
            union_sets = self._union_sets(multi_cats)
            return union_sets

    def _union_sets(self, multi_cats: List[List[str]]) -> List[str]:
        unique_sets = set()
        for lst in multi_cats:
            for ele in lst:
                if ele not in unique_sets:
                    unique_sets.add(ele)
        return list(unique_sets)

if __name__ == '__main__':
    t0 = datetime.now()
    file_path = '../data/samples.json'
    hashmap_path = '../data/catmap.pkl'
    r = Routing(file_path)
    r._build_hashmap(exist=False, threshold=1, output_path=hashmap_path)
    # r._build_hashmap(exist=True, threshold=1, output_path=hashmap_path)
    print('total time to build & load hashmap='.upper(), datetime.now() - t0)
    print('total categories=', r.hashmap.__len__())
    test_cases = [....]
    t1 = datetime.now()
    for i, test_case in enumerate(test_cases):
        print(r.extract_categories(test_case).__len__())
        if i == len(test_cases) - 1:
            print(r.extract_categories(test_case))
    print('total time to get responses for 10 testcases='.upper(), datetime.now() - t1)
